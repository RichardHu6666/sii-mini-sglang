# MiniSGL Radix Tree 前缀缓存实现详解

## 一、概述

### 1.1 什么是前缀缓存

在多请求并发场景中，不同请求之间经常共享相同的前缀。例如：

- **系统提示词共享**：多个用户请求使用相同的系统提示词
- **多轮对话**：同一对话历史被多个候选回复共享
- **批量推理**：相同输入的不同参数生成

前缀缓存通过复用已计算的 KV Cache，避免重复计算，显著提升推理效率。

### 1.2 MiniSGL 的两种前缀缓存

| 缓存类型 | 数据结构 | 适用场景 |
|----------|----------|----------|
| **NaivePrefixCache** | 简单列表 | 单请求缓存，无共享 |
| **RadixPrefixCache** | 基数树（Radix Tree） | 多请求共享前缀（默认） |

### 1.3 Radix Tree 的优势

```
场景：3 个请求共享前缀 "The cat sat"

请求 1: "The cat sat on the mat"
请求 2: "The cat sat on the hat"
请求 3: "The cat ran away"

Naive 缓存（无共享）：
  请求 1: [The, cat, sat, on, the, mat]  - 6 个 token 的 KV
  请求 2: [The, cat, sat, on, the, hat]  - 6 个 token 的 KV
  请求 3: [The, cat, ran, away]          - 4 个 token 的 KV
  总计：16 个 token 的 KV 存储

Radix 缓存（共享前缀）：
           [根]
            |
          "The"
            |
          "cat"
           / \
        "sat" "ran"
         |      |
        "on"   "away"
       /    \
    "the"  "the"
     |      |
   "mat"  "hat"

  总计：10 个 token 的 KV 存储（节省 37.5%）
```

---

## 二、RadixPrefixCache 数据结构

### 2.1 核心组件

```
RadixPrefixCache
├── RadixTreeNode (树节点)
│   ├── key: token_ids (按 page_size 分块)
│   ├── value: 物理页地址 (indices)
│   ├── children: Dict[key_fn(key), RadixTreeNode]
│   ├── parent: 父节点引用
│   ├── ref_count: 引用计数
│   └── timestamp: 最后访问时间 (LRU 淘汰)
├── root_node: 根节点 (ref_count=1，永不被淘汰)
├── key_fn: 分块函数 (按 page_size 切分)
├── evictable_size: 可淘汰的 token 数
└── protected_size: 受保护的 token 数 (有引用)
```

### 2.2 RadixTreeNode 详解

每个树节点代表一段连续的 token 序列：

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | Tensor | 该节点对应的 token ID 序列 |
| `value` | Tensor | 该节点对应的物理 KV Cache 页地址 |
| `children` | Dict | 子节点字典，key 由 `key_fn(key)` 生成 |
| `parent` | RadixTreeNode | 父节点引用 |
| `ref_count` | int | 引用计数，>0 表示正在使用，不可淘汰 |
| `timestamp` | int | 最后访问时间戳，用于 LRU 淘汰 |
| `length` | int | 该节点包含的 token 数 |

### 2.3 关键设计

| 设计点 | 说明 |
|--------|------|
| **Page 对齐** | 所有操作按 `page_size` 对齐，避免细粒度碎片 |
| **懒分裂** | 只在需要时分裂节点，减少树节点数量 |
| **引用计数** | 精确跟踪共享状态，安全淘汰 |
| **LRU 淘汰** | 按时间戳淘汰最久未使用的叶子节点 |

---

## 三、核心操作详解

### 3.1 前缀匹配（match_prefix）

**目标**：给定输入 token 序列，找到树中最长匹配前缀。

**流程**：

```sequence
participant Req as 请求
participant Cache as RadixPrefixCache
participant Tree as RadixTree

Req->>Cache: match_prefix(input_ids)
Cache->>Cache: node=root, prefix_len=0

loop 向下遍历树
    Cache->>Cache: 根据 input_ids[prefix_len:] 查找子节点

    alt 找到匹配子节点
        Cache->>Cache: 走到子节点
        Cache->>Cache: fast_compare_key() 计算匹配长度
        Cache->>Cache: 按 page_size 向下对齐 match_len
        Cache->>Cache: prefix_len += match_len

        alt 未完全匹配子节点
            Cache->>Cache: 在 match_len 处分裂节点
            Cache-->>Req: 返回分裂后的节点
        else 完全匹配
            Cache->>Cache: 更新节点 timestamp
        end
    else 无匹配子节点
        Cache-->>Req: 返回当前节点和 prefix_len
    end
end

Cache-->>Req: MatchResult(handle)
```

**关键点**：

1. **fast_compare_key**：快速比较两个 Tensor，返回首个不同位置
2. **Page 对齐**：匹配长度按 `page_size` 向下对齐，确保 KV Cache 页完整
3. **懒分裂**：只有部分匹配时才分裂节点，避免不必要操作

### 3.2 前缀插入（insert_prefix）

**目标**：将新请求的 KV Cache 插入前缀树。

**流程**：

```sequence
participant Req as 请求
participant Cache as RadixPrefixCache
participant Tree as RadixTree
participant Node as RadixTreeNode

Req->>Cache: insert_prefix(input_ids, indices)
Cache->>Cache: 按 page_size 对齐 input_ids 和 indices

Cache->>Tree: _tree_walk(input_ids)
Tree-->>Cache: 返回 (node, prefix_len)

alt prefix_len < insert_len (需要插入新节点)
    Cache->>Node: 创建新 RadixTreeNode
    Node->>Node: key = input_ids[prefix_len:]
    Node->>Node: value = indices[prefix_len:]
    Node->>Node: set_parent(node)
    Cache->>Cache: evictable_size += new_node.length
    Cache->>Cache: node = new_node
end

Cache-->>Req: InsertResult(prefix_len, handle)
```

**关键点**：

1. **尾部分割**：只插入 page 对齐的部分，尾部不足一页的 token 被丢弃
2. **引用继承**：新节点初始 `ref_count=0`，等待后续锁定

### 3.3 缓存淘汰（evict）

**目标**：当显存不足时，淘汰 LRU 缓存，释放空间。

**淘汰条件**：
- `ref_count == 0`：无引用，不在使用中
- `is_leaf()`：叶子节点，没有子节点依赖
- `not is_root()`：非根节点

**流程**：

```sequence
participant Req as 请求
participant Cache as RadixPrefixCache
participant Nodes as 待淘汰队列

Req->>Cache: evict(size)

Cache->>Cache: _collect_leave_nodes_for_evict()
Cache->>Cache: 收集所有 ref_count=0 的叶子节点
Cache->>Nodes: 用淘汰节点创建最小堆 (按 timestamp)

loop 淘汰直到满足 size
    Nodes->>Cache: 弹出 timestamp 最小的节点
    Cache->>Cache: evicted_indices.append(node.value)
    Cache->>Cache: 从父节点的 children 中删除该节点
    Cache->>Cache: evictable_size -= node.length

    alt 父节点变为可淘汰
        Cache->>Cache: 父节点变为叶子且 ref_count=0
        Cache->>Nodes: 将父节点加入堆
    end
end

Cache-->>Req: evicted_indices
```

**关键点**：

1. **最小堆**：按 timestamp 排序，快速找到 LRU 节点
2. **级联淘汰**：子节点淘汰后，父节点可能变为可淘汰
3. **根节点保护**：`root_node.ref_count=1`，永不被淘汰

### 3.4 锁定/解锁（lock/unlock）

**目标**：管理引用计数，保护正在使用的缓存。

**流程**：

```sequence
participant Cache as RadixPrefixCache
participant Node as RadixTreeNode

alt 锁定 (lock)
    Cache->>Cache: 从节点向上遍历到根
    loop 遍历路径上的每个节点
        alt ref_count == 0
            Cache->>Cache: evictable_size -= node.length
            Cache->>Cache: protected_size += node.length
        end
        Cache->>Node: ref_count += 1
    end
else 解锁 (unlock)
    Cache->>Cache: 从节点向上遍历到根
    loop 遍历路径上的每个节点
        Cache->>Node: ref_count -= 1
        alt ref_count == 0
            Cache->>Cache: evictable_size += node.length
            Cache->>Cache: protected_size -= node.length
        end
    end
end
```

**关键点**：

1. **路径锁定**：锁定节点时，锁定从根到该节点的整条路径
2. **动态统计**：`evictable_size` 和 `protected_size` 实时更新

---

## 四、CacheManager：缓存管理器

### 4.1 CacheManager 的职责

`CacheManager` 是 Scheduler 的子模块，负责协调 `RadixPrefixCache` 与 KV Cache 池的交互：

| 操作 | 说明 |
|------|------|
| `match_req(req)` | 前缀缓存匹配，返回已缓存长度 |
| `allocate_paged(reqs)` | 为批次中的请求分配 KV Cache 页 |
| `cache_req(req, finished)` | 请求完成后将 KV 插入前缀缓存 |
| `lock(handle)` / `unlock(handle)` | 锁定/解锁缓存句柄 |
| `lazy_free_region()` | 延迟回收上下文管理器 |

### 4.2 请求缓存完整流程

```sequence
participant PM as PrefillManager
participant CM as CacheManager
participant Tree as RadixPrefixCache
participant Pool as KVCachePool
participant TM as TableManager

note over PM: 阶段 1: 匹配已有缓存
PM->>CM: match_req(req)
CM->>Tree: match_prefix(input_ids)
Tree-->>CM: MatchResult(handle, cached_len)
CM-->>PM: cached_len
PM->>CM: lock(handle)

note over PM,Pool: 阶段 2: 分配新页面
PM->>Pool: allocate_paged(req)
Pool->>Pool: 计算需要页数 = ceil((device_len - cached_len) / page_size)
Pool->>Pool: 从 free_slots 分配或触发 evict
Pool-->>PM: allocated_pages

note over CM,Tree: 阶段 3: 请求完成，插入缓存
PM->>CM: cache_req(req, finished=False)
CM->>Tree: insert_prefix(input_ids[:cached_len], page_indices)
Tree-->>CM: (new_cached_len, new_handle)

CM->>CM: 释放重复区域 [old_handle.cached_len, new_cached_len)
alt 请求未完成
    CM->>CM: 保留尾部 [new_handle.cached_len, req.cached_len)
    CM->>Tree: lock(new_handle)
else 请求完成
    CM->>CM: 释放尾部 [new_handle.cached_len, req.cached_len)
end

CM->>CM: unlock(old_handle)
```

### 4.3 Lazy Free 机制

**问题**：频繁的单页回收会导致 `free_slots` 碎片化。

**解决**：批量延迟回收。

```sequence
participant Proc as _process_last_data()
participant CM as CacheManager
participant LR as lazy_free_list

Proc->>CM: with lazy_free_region():
CM->>CM: 临时替换 _free 方法
CM->>LR: 待回收页面暂存到 lazy_free_list

loop 处理每个请求
    alt 释放重复缓存区域
        Proc->>CM: _free(already_cached_indices)
        CM->>LR: lazy_free_list.append(indices)
    end
    alt 释放尾部 (请求完成)
        Proc->>CM: _free(tail_indices)
        CM->>LR: lazy_free_list.append(indices)
    end
end

Proc->>CM: with 块结束
CM->>CM: free_slots = cat([free_slots] + lazy_free_list)
CM->>CM: 恢复 _free 方法
```

---

## 五、Key Function：分块策略

### 5.1 Key Function 的作用

`key_fn` 将 token 序列转换为字典键，用于快速查找子节点。

### 5.2 分块规则

```python
# page_size = 1 时
key_fn = lambda x: x[0].item()  # 单个 token 作为 key

# page_size > 1 时 (默认 page_size=64)
key_fn = lambda x: tuple(x[:page_size].tolist())  # page_size 个 token 作为 key
```

### 5.3 设计考量

| 策略 | 优势 | 劣势 |
|------|------|------|
| **Page 对齐** | 与 KV Cache 页管理一致，避免细粒度操作 | 可能浪费少量缓存（< page_size 的部分） |
| **Tuple 作为 key** | 哈希查找 O(1)，速度快 | page_size 过大时 key 较长 |

---

## 六、示例：多请求共享前缀

### 6.1 初始状态

```
树结构：
     [根] (ref_count=1, protected)
```

### 6.2 请求 1 到达

```
请求 1: "Hello world, how are you?"

匹配结果：prefix_len = 0 (无匹配)
分配页面：[0, 1, 2, 3] (假设 page_size=8, 需要 4 页)
插入缓存后：

     [根] (ref_count=1)
       |
    "Hello" (ref_count=1, locked)
```

### 6.3 请求 2 到达（共享前缀 "Hello"）

```
请求 2: "Hello world, what's up?"

匹配结果：prefix_len = len("Hello world, ") = 14
         (假设 page_size=8, 对齐后为 8)
匹配到节点："Hello"

分配页面：[4, 5, 6] (从 cached_len=8 开始)

插入缓存后：

     [根] (ref_count=1)
       |
    "Hello" (ref_count=2, 被两个请求共享)
       |
    "world, " (ref_count=1)
```

### 6.4 请求 3 到达（无共享）

```
请求 3: "Goodbye, see you!"

匹配结果：prefix_len = 0
分配页面：[7, 8, 9]

插入缓存后：

         [根] (ref_count=1)
        /     \
    "Hello"   "Goodbye"
       |
    "world, "
```

### 6.5 请求 1 完成

```
请求 1 完成，解锁路径：

"Hello" 的 ref_count: 2 → 1 (仍被请求 2 共享)

淘汰触发（假设需要释放空间）：
- "Goodbye" ref_count=0, 可淘汰
- "Hello" ref_count=1, 受保护
- "world, " ref_count=1, 受保护

淘汰结果："Goodbye" 节点被淘汰，页面回收
```

---

## 七、性能与优化

### 7.1 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `match_prefix` | O(L / page_size) | L 为输入长度，按页遍历 |
| `insert_prefix` | O(L / page_size) | 同上 |
| `evict` | O(N log M) | N 为淘汰节点数，M 为叶子节点数 |
| `lock/unlock` | O(depth) | depth 为树深度 |

### 7.2 空间优化

| 技术 | 效果 |
|------|------|
| **Page 对齐** | 减少树节点数量，降低内存开销 |
| **懒分裂** | 避免不必要的节点创建 |
| **引用计数** | 精确共享，避免冗余 |

### 7.3 缓存命中率

缓存命中率 = 命中 token 数 / 总输入 token 数

**高命中率场景**：
- 系统提示词长且共享
- 多轮对话历史共享
- 批量推理相同输入

**低命中率场景**：
- 短提示词
- 请求之间无共享
- 频繁淘汰导致缓存失效

---

## 八、配置与调优

### 8.1 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--cache-type` | radix | 前缀缓存类型 (radix/naive) |
| `--page-size` | 64 | KV Cache 页大小，影响分块粒度 |
| `--memory-ratio` | 0.9 | KV Cache 可用显存比例 |

### 8.2 调优建议

| 场景 | 建议 |
|------|------|
| **高共享场景** | 使用 `--cache-type radix`，增大 `--memory-ratio` |
| **低共享场景** | 使用 `--cache-type naive`，减少 overhead |
| **显存受限** | 减小 `--page-size`，提高缓存利用率 |
| **长上下文** | 增大 `--page-size`，减少树节点数量 |

---

## 九、总结

### 9.1 RadixPrefixCache 核心优势

1. **高效共享**：多请求共享前缀，显著减少重复计算
2. **精确淘汰**：引用计数 + LRU，安全高效
3. **懒分裂**：按需分裂，减少树节点数量
4. **Page 对齐**：与 KV Cache 管理一致，避免碎片

### 9.2 与其他系统的对比

| 系统 | 前缀缓存 | 特点 |
|------|----------|------|
| **MiniSGL** | Radix Tree | Page 对齐、引用计数、懒分裂 |
| **vLLM** | Radix Tree | 类似实现，细节略有不同 |
| **SGLang** | Radix Tree | 支持更复杂的缓存策略 |

### 9.3 未来方向

- **多层缓存**：Hot/Cold 分离，进一步提高命中率
- **分布式缓存**：跨 GPU 共享前缀缓存
- **自适应淘汰**：基于访问模式智能淘汰
