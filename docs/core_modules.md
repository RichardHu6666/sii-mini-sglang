# MiniSGL 核心模块架构与实现

## 一、核心模块概述

MiniSGL 的单卡推理架构（TP=1）由三个核心模块组成：

| 模块 | 职责 | 核心组件 |
|------|------|----------|
| **Engine** | 模型执行引擎 | 模型加载、KV Cache 分配、采样 |
| **Scheduler** | 请求调度器 | Prefill/Decode 调度、资源管理、ZMQ 通信 |
| **KVCache** | KV 缓存管理 | KV Cache 池、前缀缓存（Radix/Naive） |

这三个模块协同工作，实现了高效的 LLM 推理服务。

---

## 二、Engine 模块：模型执行引擎

### 2.1 Engine 的核心职责

Engine 是 MiniSGL 的底层执行引擎，负责：

1. **初始化阶段**：加载模型权重、分配 KV Cache 池、初始化 Sampler
2. **执行阶段**：执行模型 Forward 计算、采样生成下一个 Token

### 2.2 Engine 初始化流程

```sequence
participant User as 用户代码
participant E as Engine
participant Model as 模型
participant KVC as KVCache 池

User->>E: Engine.__init__(config)
E->>E: 设置 CUDA 设备
E->>E: 查询 GPU 可用显存

note over E: 模型加载阶段
E->>Model: 在 meta 设备创建模型空壳
E->>Model: 加载权重到 GPU
E->>E: 可选：int8 动态量化

note over E: KV Cache 分配阶段
E->>E: 计算可用显存 = 总显存 - 模型占用
E->>KVC: 创建 KV Cache 池 (num_pages × page_size)
E->>E: 创建 Page Table (请求→物理地址映射)

note over E: Sampler 初始化
E->>E: 初始化 Sampler
```

### 2.3 Engine 执行流程

```sequence
participant S as Scheduler
participant E as Engine
participant M as Model
participant Sampler as 采样器

S->>E: forward_batch(batch, sample_args)

note over E: Forward 计算
E->>M: model.forward()
M-->>E: 返回 logits

E->>Sampler: sample(logits, sample_args)
Sampler->>Sampler: 贪婪采样 或 Top-K/Top-P 采样
Sampler-->>E: 返回 next_tokens

note over E: 异步拷贝到 CPU
E->>E: next_tokens_gpu → next_tokens_cpu (非阻塞)
E->>E: 记录 copy_done_event
E-->>S: ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

### 2.4 关键设计

| 设计点 | 说明 |
|--------|------|
| **Meta 设备创建模型** | 先在 meta 设备创建模型结构，再加载权重，避免中间状态占用显存 |
| **动态 KV Cache 计算** | 根据 GPU 可用显存自动计算可分配的 KV Cache 页数 |
| **异步 H2D 拷贝** | 采样结果异步拷贝到 CPU，使用 Event 同步，隐藏传输延迟 |

---

## 三、Scheduler 模块：请求调度器

### 3.1 Scheduler 的核心职责

Scheduler 是 MiniSGL 的调度中枢，负责：

1. **接收请求**：通过 ZMQ 从 Tokenizer 接收 UserMsg
2. **资源调度**：管理 Token Table 和 KV Cache 分配
3. **批次调度**：Prefill/Decode 分离调度，支持 Chunked Prefill
4. **结果处理**：发送 Detokenize 请求、释放完成的请求资源
5. **指标收集**：收集吞吐量、延迟、缓存命中率等指标

### 3.2 Scheduler 初始化

```sequence
participant S as Scheduler
participant E as Engine
participant TM as TableManager
participant CM as CacheManager
participant PM as PrefillManager
participant DM as DecodeManager
participant Metrics as MetricsCollector

S->>E: Engine.__init__(config)
S->>TM: TableManager(max_running_req, page_table)
S->>CM: CacheManager(num_pages, page_size, cache_type)
CM->>CM: 创建前缀缓存 (RadixPrefixCache / NaivePrefixCache)

S->>DM: DecodeManager(page_size)
S->>Metrics: MetricsCollector()
S->>PM: PrefillManager(CM, TM, DM, metrics_collector)

note over S: 初始化完成
S->>S: 进入 run_forever() 主循环
```

### 3.3 Scheduler 主循环

```sequence
participant Loop as run_forever()
participant Recv as receive_msg()
participant Sched as _schedule_next_batch()
participant Fwd as _forward()
participant Proc as _process_last_data()

note over Loop: 每轮迭代
Loop->>Recv: 接收 ZMQ 消息 (非阻塞/阻塞)

alt 有用户请求
    Recv-->>Loop: UserMsg
    Loop->>Loop: _process_one_msg(UserMsg)
    Loop->>Loop: PrefillManager.add_one_req()
end

Loop->>Sched: _schedule_next_batch()

alt 可以调度新批次
    Sched->>Sched: PrefillManager.schedule_next_batch() (优先)
    alt Prefill 有请求
        Sched-->>Loop: 返回 prefill batch
    else Prefill 无请求
        Sched->>Sched: DecodeManager.schedule_next_batch()
        Sched-->>Loop: 返回 decode batch
    end
else 无法调度
    Sched-->>Loop: None
end

alt 有新批次
    Loop->>Fwd: _forward(forward_input)
    Fwd-->>Loop: ForwardOutput
end

Loop->>Proc: _process_last_data(last_data)
Proc->>Proc: 处理上一批结果
Proc->>Proc: 发送 DetokenizeMsg
```

### 3.4 Prefill 调度流程

```sequence
participant PM as PrefillManager
participant CM as CacheManager
participant TM as TableManager
participant Adder as PrefillAdder
participant Req as PendingReq

PM->>PM: schedule_next_batch(prefill_budget)
PM->>Adder: PrefillAdder(token_budget, reserved_size)

loop 遍历 pending_list
    PM->>Adder: try_add_one(pending_req)

    alt 有剩余 token budget
        Adder->>CM: match_req(req)
        CM-->>Adder: 返回缓存匹配结果 (cached_len)

        Adder->>Adder: 计算 estimated_len = extend_len + output_len
        alt 资源足够
            Adder->>TM: allocate()
            TM-->>Adder: 返回 table_idx
            Adder->>CM: lock(handle)
            Adder->>Adder: 创建 Req 或 ChunkedReq
            Adder-->>PM: 返回 Req
        else 资源不足
            Adder-->>PM: None
        end
    else budget 耗尽
        Adder-->>PM: None
    end
end

PM-->>PM: 返回 Batch(reqs, phase="prefill")
```

### 3.5 请求处理完整链路

```sequence
participant Tokenizer
participant Sched as Scheduler
participant Engine
participant Cache as CacheManager
participant Table as TableManager

note over Tokenizer: 阶段 1: 请求到达
Tokenizer->>Sched: UserMsg(input_ids, sampling_params)
Sched->>Sched: 记录 metrics (request_arrived)
Sched->>Sched: PrefillManager.add_one_req()

note over Sched: 阶段 2: Prefill 调度
Sched->>Cache: match_req() → 前缀缓存匹配
Cache-->>Sched: cached_len (命中长度)

Sched->>Table: allocate() → 分配请求槽位
Table-->>Sched: table_idx

Sched->>Cache: lock(handle) → 锁定缓存句柄
Sched->>Cache: allocate_paged() → 分配 KV Cache 页

note over Sched,Engine: 阶段 3: Forward 计算
Sched->>Engine: forward_batch(batch)
Engine->>Engine: model.forward()
Engine->>Engine: Sampler.sample()
Engine-->>Sched: next_tokens

note over Sched: 阶段 4: 结果处理
Sched->>Sched: req.append_host(next_token)
Sched->>Sched: req.device_len = req.cached_len + 1

alt 可以 Decode (未完成)
    Sched->>Sched: DecodeManager.running_reqs.add(req)
    Sched->>Cache: cache_req(req, finished=False)
else 请求完成
    Sched->>Table: free(table_idx)
    Sched->>Cache: cache_req(req, finished=True)
end

Sched->>Tokenizer: DetokenizeMsg(next_token)
```

---

## 四、KVCache 模块：KV 缓存管理

### 4.1 KVCache 模块结构

```
KVCache 模块
├── MHAKVCache          # KV Cache 池，管理物理显存
├── BasePrefixCache     # 前缀缓存基类
│   ├── RadixPrefixCache    # 基数树前缀缓存（默认）
│   └── NaivePrefixCache    # 简单前缀缓存
├── BaseCacheHandle     # 缓存句柄，表示缓存状态
└── CacheManager        # 缓存管理器（Scheduler 子模块）
```

### 4.2 MHAKVCache：KV Cache 池

**内存布局**：
```
KV Cache 池 = [num_pages] × [page_size] × [num_layers] × [num_kv_heads] × [head_dim]

实际存储：
  _kv_buffer: [2, num_layers, num_pages, page_size, local_kv_heads, head_dim]
    ├── [0] = k_buffer (Key 缓存)
    └── [1] = v_buffer (Value 缓存)
```

**核心操作**：
| 操作 | 说明 |
|------|------|
| `k_cache(index)` / `v_cache(index)` | 获取指定位置的 KV 缓存张量 |
| `store_kv(k, v, out_loc, layer_id)` | 将 Forward 计算的 KV 写入缓存池 |

### 4.3 RadixPrefixCache：基数树前缀缓存

**数据结构**：
```
RadixTreeNode:
  ├── key: 该节点对应的 token_ids (按 page_size 分块)
  ├── value: 该节点对应的物理页地址 (indices)
  ├── children: Dict[key_fn(key), RadixTreeNode]
  ├── parent: 父节点
  ├── ref_count: 引用计数 (被多少请求共享)
  └── timestamp: 最后访问时间 (用于 LRU 淘汰)
```

**核心操作时序图**：

```sequence
participant Req as 请求
participant Cache as RadixPrefixCache
participant Tree as RadixTree
participant Node as RadixTreeNode

note over Req,Cache: 阶段 1: 前缀匹配
Req->>Cache: match_prefix(input_ids)
Cache->>Tree: _tree_walk(input_ids)

loop 遍历树
    Tree->>Node: 查找匹配的子节点
    alt 找到匹配
        Node->>Node: get_match_len() → match_len
        Node->>Node: timestamp = now (更新时间戳)
        Tree->>Tree: 继续向下遍历
    else 无匹配
        Tree-->>Cache: 返回 (当前节点，prefix_len)
    end
end

Cache-->>Req: MatchResult(cuda_handle)

note over Req,Cache: 阶段 2: 插入新前缀
Req->>Cache: insert_prefix(input_ids, indices)
Cache->>Tree: _tree_walk(input_ids)

alt 需要插入新节点
    Cache->>Node: 创建新 RadixTreeNode
    Node->>Node: set_key_value(key, value)
    Node->>Node: set_parent(parent)
    Cache->>Cache: evictable_size += new_node.length
end

Cache-->>Req: InsertResult(cached_len, cuda_handle)

note over Req,Cache: 阶段 3: 缓存淘汰
Req->>Cache: evict(size)
Cache->>Cache: _collect_leave_nodes_for_evict()
Cache->>Cache: 按 timestamp 排序 (LRU)

loop 淘汰直到满足 size
    Cache->>Node: 选择 ref_count=0 的叶子节点
    Cache->>Cache: evicted_indices.append(node.value)
    Cache->>Cache: 从父节点删除该子节点
    alt 父节点变为叶子且 ref_count=0
        Cache->>Cache: 将父节点加入待淘汰队列
    end
end

Cache-->>Req: evicted_indices
```

### 4.4 CacheManager：缓存管理器

CacheManager 是 Scheduler 的子模块，负责协调 KV Cache 的分配与回收。

**核心职责**：
| 操作 | 说明 |
|------|------|
| `match_req(req)` | 前缀缓存匹配，返回已缓存长度 |
| `allocate_paged(reqs)` | 为批次中的请求分配 KV Cache 页 |
| `cache_req(req, finished)` | 请求完成后将 KV 插入前缀缓存 |
| `lock(handle)` / `unlock(handle)` | 锁定/解锁缓存句柄（引用计数） |
| `evict(size)` | 淘汰 LRU 缓存，释放空间 |

**Lazy Free 机制**：
```sequence
participant Proc as _process_last_data()
participant CM as CacheManager
participant LR as lazy_free_region()

Proc->>CM: with lazy_free_region():
CM->>LR: 设置 _free = lazy_free (延迟回收)

loop 处理每个请求
    alt 请求完成
        Proc->>CM: cache_req(req, finished=True)
        CM->>LR: lazy_free(tail_indices) → 暂存到列表
    end
    alt 插入前缀缓存
        Proc->>CM: cache_req(req, finished=False)
        CM->>LR: lazy_free(already_cached_indices) → 暂存到列表
end

Proc->>CM: with 块结束
CM->>CM: free_slots = cat([free_slots] + lazy_free_list)
```

---

## 五、模块间协作：完整推理流程

### 5.1 单请求完整生命周期

```sequence
participant User as 用户
participant API as API Server
participant TZ as Tokenizer
participant Sched as Scheduler
participant CM as CacheManager
participant TM as TableManager
participant Engine
participant Sampler

note over User: 1. 发送请求
User->>API: POST /v1/chat/completions
API->>TZ: TokenizeMsg (ZMQ)
TZ->>TZ: tokenize()
TZ->>Sched: UserMsg(input_ids)

note over Sched: 2. Prefill 调度
Sched->>CM: match_req() → 前缀匹配
CM-->>Sched: cached_len
Sched->>TM: allocate() → 分配请求槽位
TM-->>Sched: table_idx
Sched->>CM: lock(handle)
Sched->>CM: allocate_paged()

note over Sched,Engine: 3. Prefill Forward
Sched->>Engine: forward_batch(prefill_batch)
Engine->>Engine: model.forward()
Engine-->>Sched: logits
Engine->>Sampler: sample(logits)
Sampler-->>Engine: next_tokens

note over Sched: 4. 更新请求状态
Sched->>Sched: req.device_len = req.cached_len + 1
Sched->>Sched: DecodeManager.add(req)
Sched->>CM: cache_req(req, finished=False)
Sched->>TZ: DetokenizeMsg(next_token)
TZ->>TZ: detokenize()
TZ->>API: UserReply(incremental_output)
API-->>User: 流式返回 (第一个 token)

note over Sched,Engine: 5. Decode 循环
loop 生成每个 token
    Sched->>CM: allocate_paged() (按需)
    Sched->>Engine: forward_batch(decode_batch)
    Engine->>Engine: model.forward()
    Engine-->>Sched: next_tokens
    Sched->>Sched: req.device_len += 1
    Sched->>TZ: DetokenizeMsg(next_token)
    TZ->>API: UserReply(incremental_output)
    API-->>User: 流式返回
end

note over Sched: 6. 请求完成
Sched->>TM: free(table_idx)
Sched->>CM: cache_req(req, finished=True)
Sched->>Sched: DecodeManager.remove(req)
```

### 5.2 关键设计总结

| 设计点 | 模块 | 优势 |
|--------|------|------|
| **Prefill/Decode 分离调度** | Scheduler | 优先调度 Prefill，减少排队延迟 |
| **Chunked Prefill** | Scheduler | 大 Prefill 请求分块处理，避免阻塞 Decode |
| **Radix Tree 前缀缓存** | KVCache | 支持多请求共享前缀，提高缓存命中率 |
| **Lazy Free 机制** | CacheManager | 批量回收缓存，减少碎片 |

---

## 六、配置与调优

### 6.1 关键配置项

| 配置项 | 模块 | 默认值 | 说明 |
|--------|------|--------|------|
| `--memory-ratio` | Engine | 0.9 | KV Cache 可用显存比例 |
| `--page-size` | KVCache | 64 | KV Cache 页大小 (token 数) |
| `--cache-type` | KVCache | radix | 前缀缓存类型 (radix/naive) |
| `--max-running-requests` | Scheduler | 100 | 最大并发请求数 |
| `--max-prefill-length` | Scheduler | 4096 | 最大 Prefill token 数 |

### 6.2 调优建议

1. **提高缓存命中率**：使用 `--cache-type radix`，适用于多请求共享前缀场景
2. **降低首 token 延迟**：减小 `--max-prefill-length`，避免大 Prefill 阻塞
3. **显存受限场景**：减小 `--memory-ratio` 或 `--page-size`，降低 KV Cache 占用
