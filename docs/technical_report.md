# Mini-SGLang 技术报告

## 摘要

Mini-SGLang 是一个轻量级但高性能的大语言模型推理服务框架，作为 SGLang 项目的精简实现，其代码库仅约 5,000 行 Python 代码。本报告详细阐述 Mini-SGLang 的系统架构、核心调用链路以及关键技术实现，包括 Chunked Prefill、Radix Attention、INT8 动态量化和 Metrics Endpoint 等核心特性。

---

## 目录

1. [系统架构](#1-系统架构)
2. [调用链路](#2-调用链路)
3. [Chunked Prefill 技术](#3-chunked-prefill-技术)
4. [Radix Attention 技术](#4-radix-attention-技术)
5. [INT8 动态量化技术](#5-int8-动态量化技术)
6. [Metrics Endpoint 实现](#6-metrics-endpoint-实现)
7. [总结](#7-总结)

---

## 1. 系统架构

### 1.1 整体设计

Mini-SGLang 采用分布式系统设计，通过多个独立进程协作处理 LLM 推理任务。系统基于多进程架构，利用 ZeroMQ (ZMQ) 进行控制消息通信，通过 NCCL（`torch.distributed`）实现 GPU 间的张量数据交换。

### 1.2 核心组件

系统由以下四个核心组件构成：

| 组件 | 职责 |
|------|------|
| **API Server** | 系统入口，提供 OpenAI 兼容的 API 接口（如 `/v1/chat/completions`） |
| **Tokenizer Worker** | 将输入文本转换为模型可处理的 token 序列 |
| **Detokenizer Worker** | 将模型输出的 token 转换回人类可读的文本 |
| **Scheduler Worker** | 核心计算组件，每个 GPU 对应一个 Scheduler，管理计算和资源分配 |

### 1.3 代码组织结构

```
python/minisgl/
├── core.py              # 核心数据结构：Req, Batch, Context
├── engine/              # 推理引擎，管理模型、KVCache、CUDA Graph
├── scheduler/           # 调度器，管理请求调度和资源分配
├── server/              # API 服务器和启动逻辑
├── attention/           # Attention 后端接口和实现
├── kvcache/             # KVCache 池和管理器
├── layers/              # 神经网络层（Linear, LayerNorm, RoPE 等）
├── models/              # 模型实现（Llama, Qwen 等）
├── kernel/              # 自定义 CUDA 核函数
├── message/             # 进程间消息定义
├── metrics/             # 监控指标收集
├── tokenizer/           # Tokenizer/Detokenizer 实现
├── distributed/         # 张量并行通信
└── utils/               # 工具函数
```

### 1.4 核心数据结构

**Req 类**：表示单个请求的状态
```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor      # 输入 token IDs
    table_idx: int               # 在 token 池中的索引
    cached_len: int              # 已缓存的长度
    output_len: int              # 期望输出长度
    uid: int                     # 请求唯一标识
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
```

**Batch 类**：表示一批请求的集合
```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_loc: torch.Tensor
    padded_reqs: List[Req]
    attn_metadata: BaseAttnMetadata
```

**Context 类**：全局推理上下文
```python
@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor
    attn_backend: BaseAttnBackend
    moe_backend: BaseMoeBackend
    kv_cache: BaseKVCachePool
```

---

## 2. 调用链路

### 2.1 请求处理流程

```
用户请求 → API Server → Tokenizer → Scheduler (Rank 0)
                                      ↓
                              广播到其他 Schedulers
                                      ↓
                              各 Scheduler 触发 Engine 计算
                                      ↓
Scheduler (Rank 0) 收集输出 → Detokenizer → API Server → 用户
```

### 2.2 详细调用链路

#### 2.2.1 API 层 (`minisgl/server/api_server.py`)

API Server 接收用户请求，主要端点包括：
- `/v1/chat/completions`：OpenAI 兼容的聊天完成接口
- `/generate`：原生文本生成接口
- `/metrics`：Prometheus 格式监控指标接口

```python
@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(...),
        )
    )
    return StreamingResponse(
        state.stream_chat_completions(uid),
        media_type="text/event-stream",
    )
```

#### 2.2.2 消息传递层 (`minisgl/message/`)

进程间通信基于 ZeroMQ，消息类型包括：
- `TokenizeMsg`：前端发送到 Tokenizer 的文本转 Token 请求
- `UserReply`：Tokenizer/Detokenizer 返回给前端的响应
- `BatchFrontendMsg`：批量响应消息
- `MetricsReportMsg`：调度器向 API Server 报告指标

所有消息类型支持自动序列化和反序列化。

#### 2.2.3 调度层 (`minisgl/scheduler/`)

Scheduler 是每个 TP Worker 进程的核心，主要职责：
1. 接收来自 Tokenizer 的请求
2. 管理 PrefillManager 和 DecodeManager
3. 与 Engine 交互触发模型计算
4. 协调多 GPU 间的通信

调度器的核心循环：
```
1. 从 pending list 中选择请求 → PrefillManager.schedule_next_batch()
2. 执行 Prefill 阶段计算
3. 将完成的请求转移到 DecodeManager
4. 执行 Decode 阶段计算（自回归生成）
5. 返回输出 token 给 Detokenizer
```

#### 2.2.4 引擎层 (`minisgl/engine/`)

Engine 在单个 GPU 进程上执行实际计算：
- 管理模型权重加载
- 管理 KVCache 分配
- 执行 Attention 计算
- 支持 CUDA Graph 捕获和回放

### 2.3 系统启动流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     用户执行启动命令                              │
│              python -m minisgl --model <model_path>              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    launch_server() [launch.py]                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ start_subprocess() - 启动所有子进程                       │    │
│  │                                                          │    │
│  │  for i in range(world_size):  # Scheduler 进程           │    │
│  │      Process(target=_run_scheduler, ...)                 │    │
│  │          ├─ Scheduler.__init__()                         │    │
│  │          │   ├─ Engine(config)                           │    │
│  │          │   │   ├─ 初始化通信 (NCCL/Gloo)                │    │
│  │          │   │   ├─ 加载模型权重                          │    │
│  │          │   │   ├─ 分配 KV Cache                         │    │
│  │          │   │   ├─ 创建 Attention Backend               │    │
│  │          │   │   └─ 捕获 CUDA Graphs                     │    │
│  │          │   ├─ TableManager(max_running_reqs)           │    │
│  │          │   ├─ CacheManager(num_pages, page_size)       │    │
│  │          │   ├─ DecodeManager(page_size)                 │    │
│  │          │   ├─ PrefillManager(...)                      │    │
│  │          │   └─ MetricsCollector()                       │    │
│  │          └─ scheduler.run_forever()                      │    │
│  │                                                          │    │
│  │  Process(target=tokenize_worker, ...)  # Tokenizer       │    │
│  │      └─ ZMQ sockets 建立                                  │    │
│  │                                                          │    │
│  │  Process(target=tokenize_worker, ...)  # Detokenizer     │    │
│  │      └─ ZMQ sockets 建立                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  等待 ACK: for _ in range(num_tokenizers + 2): ack_queue.get()  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  run_api_server() [api_server.py]               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ FrontendManager 初始化                                    │    │
│  │   ├─ ZmqAsyncPushQueue(tokenizer_addr)                  │    │
│  │   ├─ ZmqAsyncPullQueue(frontend_addr)                   │    │
│  │   └─ MetricsState()                                     │    │
│  │                                                          │    │
│  │ Uvicorn 启动 uvicorn.run(app, host, port)                │    │
│  │   ├─ FastAPI 应用注册                                     │    │
│  │   │   ├─ POST /v1/chat/completions                      │    │
│  │   │   ├─ POST /generate                                 │    │
│  │   │   ├─ GET  /metrics                                  │    │
│  │   │   └─ GET  /v1/models                                │    │
│  │   └─ lifespan 事件处理                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    系统就绪，等待请求
```

### 2.4 端到端请求处理详细流程

```
┌──────────────┐
│  User Request │
└──────┬───────┘
       │ HTTP POST /v1/chat/completions
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ API Server (FastAPI)                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ v1_completions()                                        │    │
│  │   uid = state.new_user()                                │    │
│  │   sampling_params = SamplingParams(...)                 │    │
│  │   send_one(TokenizeMsg(uid, text, sampling_params))     │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │ ZMQ Push (zmq_tokenizer_addr)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Tokenizer Worker                                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ tokenize_worker()                                       │    │
│  │   for msg in recv_zmq():                                │    │
│  │       input_ids = tokenizer.encode(msg.text)            │    │
│  │       send_to_backend(UserMsg(uid, input_ids, ...))     │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │ ZMQ Push (zmq_backend_addr)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Scheduler (Rank 0) - 接收并广播                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ _process_one_msg(UserMsg)                               │    │
│  │   ├─ metrics_collector.request_arrived(uid, input_len)  │    │
│  │   └─ prefill_manager.add_one_req(msg)                   │    │
│  │       └─ pending_list.append(PendingReq)                │    │
│  │                                                          │    │
│  │ sync_all_ranks() - 广播到其他 TP ranks                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Scheduler Loop (normal_loop / overlap_loop)                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ _schedule_next_batch()                                  │    │
│  │   │                                                      │    │
│  │   ├─ prefill_manager.schedule_next_batch(prefill_budget)│    │
│  │   │   └─ PrefillAdder.try_add_one(pending_req)          │    │
│  │   │       ├─ cache_manager.match_req()                  │    │
│  │   │       │   └─ prefix_cache.match_prefix(input_ids)   │    │
│  │   │       ├─ table_manager.allocate()                   │    │
│  │   │       └─ ChunkedReq / Req 创建                       │    │
│  │   │                                                      │    │
│  │   └─ decode_manager.schedule_next_batch()               │    │
│  │       └─ Batch(running_reqs, phase="decode")            │    │
│  │                                                          │    │
│  │ _prepare_batch(batch)                                   │    │
│  │   ├─ graph_runner.pad_batch(batch)                      │    │
│  │   ├─ cache_manager.allocate_paged(reqs)                 │    │
│  │   │   └─ _allocate(needed_pages)                        │    │
│  │   │       └─ (evict if needed) → prefix_cache.evict()   │    │
│  │   ├─ _make_positions(batch)                             │    │
│  │   ├─ _make_input_tuple(batch)                           │    │
│  │   ├─ attn_backend.prepare_metadata(batch)               │    │
│  │   └─ sampler.prepare(batch)                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Engine.forward_batch(batch, sample_args)                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ with ctx.forward_batch(batch):                          │    │
│  │   if graph_runner.can_use_cuda_graph(batch):            │    │
│  │       logits = graph_runner.replay(batch)               │    │
│  │   else:                                                 │    │
│  │       logits = model.forward()                          │    │
│  │                                                          │    │
│  │ next_tokens_gpu = sampler.sample(logits, args)          │    │
│  │ next_tokens_cpu = next_tokens_gpu.to("cpu")             │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Scheduler._process_last_data()                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ for i, req in enumerate(batch.reqs):                    │    │
│  │   next_token = next_tokens_cpu[i]                       │    │
│  │   req.append_host(next_token)                           │    │
│  │   finished = not req.can_decode or next_token == eos    │    │
│  │   reply.append(DetokenizeMsg(uid, next_token, finished))│    │
│  │                                                          │    │
│  │   metrics_collector.request_first_token(uid)  # 首个 token │    │
│  │   metrics_collector.request_output_token(uid)           │    │
│  │                                                          │    │
│  │   if finished:                                          │    │
│  │     decode_manager.remove_req(req)                      │    │
│  │     cache_manager.cache_req(req, finished=True)         │    │
│  │     metrics_collector.request_completed(uid)            │    │
│  │   elif batch.is_prefill:                                │    │
│  │     decode_manager.running_reqs.add(req)                │    │
│  │     cache_manager.cache_req(req, finished=False)        │    │
│  │                                                          │    │
│  │ send_result(reply) → ZMQ Push                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │ ZMQ Push (zmq_detokenizer_addr)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Detokenizer Worker                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ tokenize_worker() - detokenizer 模式                     │    │
│  │   for msg in recv_zmq():                                │    │
│  │       text = tokenizer.decode(next_token)               │    │
│  │       send_to_frontend(UserReply(uid, text, finished))  │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │ ZMQ Pull (zmq_frontend_addr)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ API Server - 流式响应                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ stream_chat_completions(uid)                            │    │
│  │   async for ack in wait_for_ack(uid):                   │    │
│  │       chunk = {"delta": {"content": ack.incremental}}   │    │
│  │       yield f"data: {json.dumps(chunk)}\n\n"            │    │
│  │       if ack.finished: break                            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────┬──────────────────────────────────────────────────────────┘
       │ HTTP StreamingResponse
       ▼
┌──────────────┐
│  User Response│
└──────────────┘
```

### 2.5 Scheduler 主循环详细流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ normal_loop() / overlap_loop()                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               │
┌────────────────┐      │               │
│ receive_msg()  │      │               │
│ (blocking=T/F) │      │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               │               │
┌────────────────┐      │               │
│ _process_one_  │      │               │
│ msg(msg)       │      │               │
│                │      │               │
│ case UserMsg:  │      │               │
│  • 检查 input_len     │               │
│  • 调整 max_tokens    │               │
│  • metrics_collector. │               │
│    request_arrived()  │               │
│  • prefill_manager.   │               │
│    add_one_req()      │               │
│                │      │               │
│ case AbortMsg: │      │               │
│  • prefill_manager.   │               │
│    abort_req()        │               │
│  • decode_manager.    │               │
│    abort_req()        │               │
│  • metrics_collector. │               │
│    request_aborted()  │               │
└───────┬────────┘      │               │
        │               │               │
        ▼               │               │
┌─────────────────────────────────────────────────────────────────┐
│ _schedule_next_batch()                                          │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ prefill_manager.schedule_next_batch(prefill_budget)       │  │
│  │   │                                                        │  │
│  │   ├─ PrefillAdder(token_budget, reserved_size, ...)       │  │
│  │   │                                                        │  │
│  │   └─ for pending_req in pending_list:                     │  │
│  │       │                                                    │  │
│  │       ├─ if pending_req.chunked_req:                      │  │
│  │       │   └─ 继续处理 chunked 请求                          │  │
│  │       │                                                    │  │
│  │       ├─ _try_allocate_one(pending_req)                   │  │
│  │       │   ├─ table_manager.available_size > 0?            │  │
│  │       │   ├─ cache_manager.match_req(req)                 │  │
│  │       │   │   └─ prefix_cache.match_prefix(input_ids)     │  │
│  │       │   │       └─ _tree_walk() → (node, prefix_len)    │  │
│  │       │   ├─ metrics_collector.record_cache_tokens()      │  │
│  │       │   ├─ estimated_len + reserved <= available_size?  │  │
│  │       │   ├─ cache_manager.lock(handle)                   │  │
│  │       │   ├─ table_manager.allocate() → table_idx         │  │
│  │       │   └─ 设置 cached part 的 token IDs                 │  │
│  │       │                                                    │  │
│  │       └─ _add_one_req(pending_req, handle, table_idx)     │  │
│  │           ├─ chunk_size = min(token_budget, remain_len)   │  │
│  │           ├─ is_chunked = chunk_size < remain_len         │  │
│  │           ├─ CLS = ChunkedReq if is_chunked else Req      │  │
│  │           ├─ 设置 token_pool[table_idx, cached:]          │  │
│  │           └─ return CLS(...)                              │  │
│  │                                                            │  │
│  │ if batch is None:                                         │  │
│  │   decode_manager.schedule_next_batch()                    │  │
│  │     └─ Batch(running_reqs, phase="decode")                │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────┬─────────────────────────────────────────────────────────┘
        │
        │ (batch is not None)
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ _prepare_batch(batch)                                           │
│                                                                  │
│  ├─ graph_runner.pad_batch(batch)                              │
│  │   └─ padded_reqs = reqs + [dummy_req] * (pad_size - size)   │
│  │                                                              │
│  ├─ cache_manager.allocate_paged(reqs)                         │
│  │   ├─ 计算 needed_pages                                      │
│  │   ├─ _allocate(needed_pages)                                │
│  │   │   ├─ if needed_pages > free_pages:                      │
│  │   │   │   └─ evict = prefix_cache.evict(size)               │
│  │   │   └─ allocated = free_slots[:needed_pages]              │
│  │   └─ _write_page_table(page_table, allocated, ...)          │
│  │                                                              │
│  ├─ batch.positions = _make_positions(batch, device)           │
│  │   └─ for req in padded_reqs:                                │
│  │       arange(req.cached_len, req.device_len)                │
│  │                                                              │
│  ├─ input_mapping = _make_input_tuple(batch, device)           │
│  │   └─ for req in padded_reqs:                                │
│  │       mapping_host.fill_(req.table_idx)                     │
│  │                                                              │
│  ├─ batch.out_loc = page_table[input_mapping]                  │
│  │                                                              │
│  ├─ attn_backend.prepare_metadata(batch)                       │
│  │   └─ 根据 batch.phase 创建 PrefillMetadata / DecodeMetadata │
│  │                                                              │
│  └─ sample_args = sampler.prepare(batch)                       │
│      └─ 提取 temperature, top_p, top_k 等参数                   │
└───────┬─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ _forward(forward_input)                                         │
│                                                                  │
│  batch.input_ids = token_pool[input_mapping]                   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ engine.forward_batch(batch, sample_args)                  │  │
│  │   │                                                        │  │
│  │   with ctx.forward_batch(batch):                          │  │
│  │     if graph_runner.can_use_cuda_graph(batch):            │  │
│  │       logits = graph_runner.replay(batch)                 │  │
│  │         ├─ buffer.copy_from(batch)                        │  │
│  │         ├─ attn_backend.prepare_for_replay(batch)         │  │
│  │         ├─ graph.replay()                                 │  │
│  │         └─ return buffer.logits[:batch.size]              │  │
│  │     else:                                                 │  │
│  │       logits = model.forward()                            │  │
│  │         ├─ for layer in layers:                           │  │
│  │         │   ├─ hidden_states = attention.forward()        │  │
│  │         │   │   ├─ q = qkv_proj(hidden_states)            │  │
│  │         │   │   ├─ kv_cache.store_kv(k, v, out_loc)       │  │
│  │         │   │   └─ attn_backend.forward(q, k, v, batch)   │  │
│  │         │   └─ hidden_states = mlp.forward()              │  │
│  │         │       ├─ gate = gate_proj(hidden_states)        │  │
│  │         │       ├─ up = up_proj(hidden_states)            │  │
│  │         │       └─ down = down_proj(act(gate) * up)       │  │
│  │         └─ return logits(lm_head(hidden_states))          │  │
│  │                                                            │  │
│  │ next_tokens_gpu = sampler.sample(logits, args)            │  │
│  │   ├─ if greedy:                                           │  │
│  │   │   └─ argmax(logits)                                   │  │
│  │   └─ else:                                                │  │
│  │       ├─ temperature scaling                              │  │
│  │       ├─ top_p / top_k filtering                          │  │
│  │       └─ multinomial sample                               │  │
│  │                                                            │  │
│  │ token_pool[output_mapping] = next_tokens_gpu              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  return ForwardOutput(next_tokens_gpu, next_tokens_cpu, event) │
└───────┬─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ _process_last_data(last_data)                                   │
│                                                                  │
│  batch, (_, next_tokens_cpu, copy_done) = last_data            │
│  copy_done.synchronize()                                       │
│                                                                  │
│  with cache_manager.lazy_free_region():                        │
│    for i, req in enumerate(batch.reqs):                        │
│      if isinstance(req, ChunkedReq): continue                  │
│      │                                                          │
│      next_token = next_tokens_cpu[i]                           │
│      req.append_host(next_token)                               │
│      finished = not req.can_decode                             │
│      finished |= next_token == eos (if not ignore_eos)         │
│      │                                                          │
│      # Metrics recording                                         │
│      if req.device_len == req.cached_len + 1:                  │
│        metrics_collector.request_first_token(uid)              │
│      metrics_collector.request_output_token(uid)               │
│      │                                                          │
│      if finished and req not in finished_reqs:                 │
│        decode_manager.remove_req(req)                          │
│        _free_req_resources(req)                                │
│        ├─ table_manager.free(table_idx)                        │
│        └─ cache_manager.cache_req(req, finished=True)          │
│        metrics_collector.request_completed(uid)                │
│      elif batch.is_prefill:                                    │
│        decode_manager.running_reqs.add(req)                    │
│        cache_manager.cache_req(req, finished=False)            │
│        ├─ cached_len, new_handle =                             │
│        │   prefix_cache.insert_prefix(insert_ids, indices)     │
│        ├─ unlock(old_handle)                                   │
│        ├─ free 已缓存部分的 page_indices                        │
│        └─ lock(new_handle)                                     │
│                                                                  │
│  send_result(reply)  # ZMQ Push to Detokenizer                 │
│                                                                  │
│  # Update metrics counters                                       │
│  metrics_collector.set_queued_count(len(prefill_manager.       │
│    pending_list))                                              │
│  metrics_collector.set_active_count(len(decode_manager.        │
│    running_reqs))                                              │
│  _metrics_counter += 1                                         │
│  if _metrics_counter >= 10:                                    │
│    _send_metrics()  # MetricsReportMsg to API Server           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.6 多进程通信架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         主机 (Host)                                      │
│                                                                          │
│  ┌─────────────┐                                                         │
│  │ API Server  │  FastAPI + Uvicorn                                      │
│  │ (主进程)    │  - HTTP 端点：/v1/chat/completions, /metrics            │
│  └──────┬──────┘  - ZMQ: frontend_addr, tokenizer_addr, detokenizer_addr │
│         │ ZMQ Push/Pull                                                  │
│         ├─────────────────────────────────────────────────────────┐      │
│         │                                                         │      │
│         ▼                                                         │      │
│  ┌─────────────┐    ┌─────────────┐                              │      │
│  │ Tokenizer 0 │ ...│ Tokenizer N │  (num_tokenizers 个进程)      │      │
│  │  Worker     │    │  Worker     │  - ZMQ: frontend_addr        │      │
│  └──────┬──────┘    └──────┬──────┘  - ZMQ: backend_addr         │      │
│         │ ZMQ Push         │ ZMQ Push                             │      │
│         └────────┬─────────┘                                      │      │
│                  │                                                │      │
│         ┌────────┴────────┐                                       │      │
│         ▼                 ▼                                       │      │
│  ┌─────────────┐    ┌─────────────┐                              │      │
│  │ Scheduler 0 │    │ Scheduler N │  (world_size = TP 并行度)     │      │
│  │  (Rank 0)   │◄──►│  (Rank N)   │  - ZMQ: backend_addr (Rank 0)│      │
│  └──────┬──────┘    └──────┬──────┘  - NCCL: all-reduce, broadcast│      │
│         │                  │                                      │      │
│         └────────┬─────────┘                                      │      │
│                  │                                                │      │
│         ┌────────┴────────┐                                       │      │
│         ▼                 ▼                                       │      │
│  ┌─────────────┐                                                  │      │
│  │Detokenizer  │  (1 个进程)                                       │      │
│  │  Worker     │  - ZMQ: backend_addr, frontend_addr             │      │
│  └──────┬──────┘  - ZMQ: detokenizer_addr                        │      │
│         │ ZMQ Push                                               │      │
│         └─────────────────────────────────────────────────────────┘      │
│                  │                                                       │
│                  ▼                                                       │
│         ┌─────────────┐                                                  │
│         │ API Server  │◄─────────────────────────────────────────────────┘
│         │ (接收响应)  │
│         └─────────────┘
│
└─────────────────────────────────────────────────────────────────────────┘

通信方式说明:
┌───────────────┬────────────────────────────────────────────────────────┐
│ 通信通道       │ 用途                                                    │
├───────────────┼────────────────────────────────────────────────────────┤
│ ZMQ Push/Pull │ 控制消息传输 (tokenization 请求/响应，用户请求/响应)       │
├───────────────┼────────────────────────────────────────────────────────┤
│ NCCL          │ GPU 间张量数据交换 (all-reduce, all-gather, broadcast)   │
├───────────────┼────────────────────────────────────────────────────────┤
│ Gloo          │ CPU 进程组通信 (用于单节点多卡同步)                        │
└───────────────┴────────────────────────────────────────────────────────┘
```

---

## 3. Chunked Prefill 技术

### 3.1 技术背景

Chunked Prefill 是 Sarathi-Serve 提出的技术，用于解决长上下文服务时的峰值内存占用问题。通过将长 prompt 分割成较小的 chunk 进行 prefill，有效防止 OOM 错误。

### 3.2 核心实现 (`minisgl/scheduler/prefill.py`)

#### 3.2.1 ChunkedReq 类

```python
class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # 避免被添加到 decode manager
```

`ChunkedReq` 继承自 `Req`，但禁用了采样和解码能力，仅用于 prefill 阶段的部分处理。

#### 3.2.2 PrefillAdder 决策流程

```
┌─────────────────────────────────────────────────────────────────┐
│ PrefillAdder.try_add_one(pending_req)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ token_budget <= 0?  │
              └──────────┬──────────┘
                    Yes  │
                    │    │ No
                    │    ▼
                    │    ┌─────────────────────────┐
                    │    │ pending_req.chunked_req │
                    │    └───────────┬─────────────┘
                    │                │
                    │           Yes  │  No
                    │                │
                    │                ▼
                    │      ┌───────────────────────┐
                    │      │ _try_allocate_one()   │
                    │      │                       │
                    │      │ 1. table_manager.     │
                    │      │    available_size > 0 │
                    │      │                       │
                    │      │ 2. cache_manager.     │
                    │      │    match_req()        │
                    │      │    └─ prefix_cache.   │
                    │      │       match_prefix()  │
                    │      │                       │
                    │      │ 3. record_cache_      │
                    │      │    tokens()           │
                    │      │                       │
                    │      │ 4. estimated_len +    │
                    │      │    reserved <= avail? │
                    │      │                       │
                    │      │ 5. cache_manager.     │
                    │      │    lock(handle)       │
                    │      │                       │
                    │      │ 6. table_manager.     │
                    │      │    allocate()         │
                    │      │                       │
                    │      │ 7. 设置 cached part   │
                    │      │    token IDs          │
                    │      └───────────┬───────────┘
                    │                  │
                    │             None │  (handle, table_idx)
                    │                  │
                    │                  ▼
                    │        ┌─────────────────────┐
                    │        │ _add_one_req()      │
                    │        │                     │
                    │        │ remain_len =        │
                    │        │   input_len -       │
                    │        │   cached_len        │
                    │        │                     │
                    │        │ chunk_size =        │
                    │        │   min(token_budget, │
                    │        │   remain_len)       │
                    │        │                     │
                    │        │ is_chunked =        │
                    │        │   chunk_size <      │
                    │        │   remain_len        │
                    │        │                     │
                    │        │ CLS = ChunkedReq if │
                    │        │   is_chunked else   │
                    │        │   Req               │
                    │        │                     │
                    │        │ 更新 token_budget   │
                    │        │ 更新 reserved_size  │
                    │        │                     │
                    │        │ 设置 token_pool     │
                    │        │                     │
                    │        │ return CLS(...)     │
                    │        └─────────────────────┘
                    │                  │
                    └──────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ 返回:                    │
                    │ - Req (完整 prefill)      │
                    │ - ChunkedReq (部分)      │
                    │ - None (无法添加)        │
                    └──────────────────────────┘
```

#### 3.2.3 PrefillManager 调度流程

```python
@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq]

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )

        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []

        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # 无法继续添加

        self.pending_list = chunked_list + self.pending_list[len(reqs):]
        return Batch(reqs=reqs, phase="prefill")
```

调度流程：
1. 创建 `PrefillAdder`，传入 prefill budget 和 decode 预留空间
2. 遍历 pending list，尝试添加请求到 batch
3. 未完成的 chunked 请求保留在 pending list 头部
4. 返回 prefill batch 供 Engine 执行

### 3.3 配置参数

通过 `--max-prefill-length` 参数控制 chunk 大小，默认启用。注意设置过小（如 128）会显著降低性能。

---

## 4. Radix Attention 技术

### 4.1 技术背景

Radix Attention 是 SGLang 原创的设计，通过树状结构管理 KV Cache，实现跨请求的前缀共享。对于多轮对话、few-shot learning 等场景，可显著减少重复计算。

### 4.2 核心实现 (`minisgl/kvcache/radix_cache.py`)

#### 4.2.1 RadixTreeNode

```python
class RadixTreeNode:
    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid: int
        self.timestamp: int  # 用于 LRU 驱逐
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int
```

每个节点表示一段连续的 token 序列，关键属性：
- `key_fn`：从 token 序列提取键的函数
- `children`：子节点字典
- `ref_count`：引用计数，保护正在使用的缓存
- `timestamp`：时间戳，用于 LRU 驱逐策略
- `_value`：存储该段 token 对应的 KV Cache 索引

#### 4.2.2 match_prefix() 时序图

```
┌─────────────────────────────────────────────────────────────────┐
│ match_prefix(input_ids: torch.Tensor) -> MatchResult            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ _tree_walk(input_ids)                                           │
│                                                                  │
│  prefix_len = 0                                                 │
│  node = root_node                                               │
│  tic = time.monotonic_ns()                                      │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ while prefix_len < len(input_ids):                        │  │
│  │   │                                                        │  │
│  │   ├─ key = key_fn(input_ids[prefix_len:])                 │  │
│  │   │                                                        │  │
│  │   ├─ child_node = node.children.get(key)                  │  │
│  │   │   │                                                    │  │
│  │   │   │ None       ┌─────────────────────────────┐        │  │
│  │   │   └───────────►│ return (node, prefix_len)   │        │  │
│  │   │                └─────────────────────────────┘        │  │
│  │   │                                                        │  │
│  │   │ Found                                                  │  │
│  │   ▼                                                        │  │
│  │   node = child_node  # 向下遍历                             │  │
│  │                                                            │  │
│  │   match_len = node.get_match_len(input_ids[prefix_len:])   │  │
│  │     └─ fast_compare_key(node._key, input_ids)              │  │
│  │        └─ 逐元素比较，返回第一个不同位置的索引              │  │
│  │                                                            │  │
│  │   match_len = align_down(match_len, page_size)             │  │
│  │                                                            │  │
│  │   prefix_len += match_len                                  │  │
│  │                                                            │  │
│  │   ┌───────────────────────────────────────────────────┐    │  │
│  │   │ match_len != node.length? (部分匹配)               │    │  │
│  │   │                                                    │    │  │
│  │   │ Yes ──► node = node.split_at(match_len)            │    │  │
│  │   │            return (node, prefix_len)               │    │  │
│  │   └───────────────────────────────────────────────────┘    │  │
│  │                                                            │  │
│  │   node.timestamp = tic  # 更新时间戳                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  return (node, prefix_len)                                      │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              MatchResult(RadixCacheHandle(prefix_len, node))
```

#### 4.2.3 节点分裂 (split_at)

```
分裂前:                         分裂后:

     parent                         parent
       │                              │
       ▼                              ▼
   ┌───────┐                      ┌───────┐
   │ node  │ [0:pos] + [pos:len]  │new_node│ [0:pos]
   │ key   │                      │ key   │
   │ value │                      │ value │
   └───────┘                      └───┬───┘
                                      │
                            ┌─────────┴─────────┐
                            │                   │
                            ▼                   ▼
                       ┌─────────┐         ┌─────────┐
                       │ new_node│         │  node   │
                       │ [0:pos] │         │ [pos:]  │
                       │ key     │         │ key     │
                       │ value   │         │ value   │
                       │ ref=old │         │ ref=0   │
                       └─────────┘         └─────────┘
                                                │
                                      (children 转移给 node)

def split_at(self, pos: int) -> RadixTreeNode:
    parent = self.parent

    # 创建新节点（前缀部分）
    new_node = RadixTreeNode(self.key_fn, self.timestamp)
    new_node.set_key_value(self._key[:pos], self._value[:pos])
    new_node.set_parent(parent)  # 插入到 parent 和 self 之间
    new_node.ref_count = self.ref_count

    # 当前节点保留后缀部分
    self.set_key_value(self._key[pos:], self._value[pos:])
    self.set_parent(new_node)  # self 成为 new_node 的子节点

    return new_node
```

#### 4.2.4 insert_prefix() 时序图

```
┌─────────────────────────────────────────────────────────────────┐
│ insert_prefix(input_ids, indices) -> InsertResult               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    insert_len = align_down(len(input_ids), page_size)
    input_ids, indices = input_ids[:insert_len], indices[:insert_len]
                         │
                         ▼
    (node, prefix_len) = _tree_walk(input_ids)
                         │
                         ▼
              ┌─────────────────────┐
              │ prefix_len !=       │
              │ insert_len?         │
              └──────────┬──────────┘
                    Yes  │  No
                    │    │
                    │    └──────────────────────────┐
                    │                               │
                    ▼                               │
    ┌───────────────────────────────────┐           │
    │ 创建新节点插入到树中                │           │
    │                                   │           │
    │ new_node = RadixTreeNode()        │           │
    │ new_node.set_key_value(           │           │
    │   input_ids[prefix_len:],         │           │
    │   indices[prefix_len:]            │           │
    │ )                                 │           │
    │ new_node.set_parent(node)         │           │
    │ evictable_size += new_node.length │           │
    │ node = new_node                   │           │
    └───────────────┬───────────────────┘           │
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
              InsertResult(prefix_len, RadixCacheHandle(insert_len, node))
```

#### 4.2.5 evict() LRU 驱逐流程

```
┌─────────────────────────────────────────────────────────────────┐
│ evict(size: int) -> torch.Tensor                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    leave_nodes = _collect_leave_nodes_for_evict()
      └─ DFS 遍历树，收集 ref_count=0 的叶节点
                         │
                         ▼
    heapq.heapify(leave_nodes)  # 按 timestamp 排序 (LRU)
                         │
                         ▼
    ┌───────────────────────────────────────────────────────────┐
    │ while evicted_size < size:                                │
    │                                                            │
    │   node = heapq.heappop(leave_nodes)                       │
    │     └─ 弹出最早访问的节点                                  │
    │                                                            │
    │   evicted_size += node.length                             │
    │   evicted_indices.append(node.value)                      │
    │   evictable_size -= node.length                           │
    │                                                            │
    │   parent = node.parent                                    │
    │   del parent.children[key_fn(node._key)]                  │
    │     └─ 从父节点删除引用                                    │
    │                                                            │
    │   ┌───────────────────────────────────────────────────┐    │
    │   │ parent.is_leaf() and parent.ref_count == 0?       │    │
    │   │                                                    │    │
    │   │ Yes ──► heapq.heappush(leave_nodes, parent)       │    │
    │   └───────────────────────────────────────────────────┘    │
    └───────────────────────────────────────────────────────────┘
                         │
                         ▼
    return torch.cat(evicted_indices)
```

#### 4.2.6 缓存状态转换图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Radix Cache 状态转换                          │
└─────────────────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────────────────┐
          │                  Free State                      │
          │              (free_slots 列表)                   │
          └─────────────────────┬───────────────────────────┘
                                │ allocate_paged()
                                ▼
          ┌─────────────────────────────────────────────────┐
          │              Allocated State                     │
          │         (page_table 指向物理位置)                │
          └─────────────────────┬───────────────────────────┘
                                │ cache_req()
                                ▼
          ┌─────────────────────────────────────────────────┐
          │               Cached State                       │
          │    (insert_prefix 插入到 radix tree)             │
          │    ref_count > 0: protected                     │
          │    ref_count = 0: evictable                     │
          └─────────────────────┬───────────────────────────┘
                      ┌─────────┴─────────┐
                      │                   │
            lock()    │                   │   evict()
            (ref++)   │                   │   (ref==0)
                      ▼                   ▼
          ┌─────────────────┐   ┌─────────────────┐
          │  Protected      │   │    Evicted      │
          │  (不可驱逐)     │   │   (返回 free_   │
          │                 │   │    slots)       │
          └─────────────────┘   └─────────────────┘
```

### 4.3 缓存锁定机制

```python
def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
    node = handle.node
    if unlock:
        while not node.is_root():
            node.ref_count -= 1
            if node.ref_count == 0:
                self.evictable_size += node.length
                self.protected_size -= node.length
            node = node.parent
    else:
        while not node.is_root():
            if node.ref_count == 0:
                self.evictable_size -= node.length
                self.protected_size += node.length
            node.ref_count += 1
            node = node.parent
```

通过引用计数保护正在使用的缓存路径，确保不会被驱逐。

---

## 5. INT8 动态量化技术

### 5.1 技术背景

Mini-SGLang 支持对 BF16/FP16 模型权重进行动态 INT8 量化，在几乎不损失精度的情况下，将模型权重内存占用减少约 50%。

### 5.2 核心实现 (`minisgl/layers/quant.py`)

#### 5.2.1 量化算法流程

```
┌─────────────────────────────────────────────────────────────────┐
│ dynamic_quantize_int8_per_channel(weight: torch.Tensor)         │
│                                                                 │
│  输入：weight [out_features, in_features] (fp16/bf16/fp32)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 计算每行（输出通道）的 min/max                           │
│                                                                 │
│   min_val = weight.min(dim=1, keepdim=True)[0]  # [out, 1]      │
│   max_val = weight.max(dim=1, keepdim=True)[0]  # [out, 1]      │
│                                                                 │
│   对于每个输出通道 i:                                           │
│   min_val[i] = min(weight[i, :])                                │
│   max_val[i] = max(weight[i, :])                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 计算逐通道 scale                                        │
│                                                                 │
│   max_abs = torch.maximum(-min_val, max_val)  # [out, 1]        │
│   scale = max_abs / 127.0                                       │
│   scale = torch.clamp(scale, min=eps)                           │
│                                                                 │
│   对于每个输出通道 i:                                           │
│   scale[i] = max(|min_val[i]|, |max_val[i]|) / 127              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 量化并 clamp                                            │
│                                                                 │
│   qweight = torch.round(weight / scale)                         │
│   qweight = torch.clamp(qweight, -127, 127)                     │
│   qweight = qweight.to(torch.int8)                              │
│                                                                 │
│   对于每个元素 (i, j):                                          │
│   qweight[i,j] = clamp(round(weight[i,j] / scale[i]), -127, 127)│
│                                                                 │
│   scale = scale.squeeze(dim=1)  # [out_features]                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              返回：(qweight, scale)
              - qweight: [out, in] (int8)
              - scale: [out] (fp32)
```

#### 5.2.2 反量化流程

```
┌─────────────────────────────────────────────────────────────────┐
│ dequantize_int8_per_channel(qweight, scale, dtype)              │
│                                                                 │
│  输入：qweight [out_features, in_features] (int8)               │
│       scale [out_features] (fp32)                               │
│       dtype: torch.bfloat16 / torch.float16                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 检查 scale 形状                                          │
│                                                                 │
│   if scale.numel() == 1:  # 逐张量量化                            │
│       return qweight.to(dtype) * scale.to(dtype)                │
│                                                                 │
│   if scale.dim() == 1:  # 逐通道量化                             │
│       scale = scale.view(-1, 1)  # [out, 1]                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 反量化                                                  │
│                                                                 │
│   return (qweight.to(dtype) * scale.to(dtype))                  │
│                                                                 │
│   对于每个元素 (i, j):                                          │
│   weight_fp[i,j] = qweight[i,j] * scale[i]                      │
│                                                                 │
│   利用广播机制，scale 沿 in_features 维度复制                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              返回：weight_fp [out, in] (dtype)
```

#### 5.2.3 权重加载流程

```
┌─────────────────────────────────────────────────────────────────┐
│ Engine._load_weight_state_dict(config)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    state_dict = {}
                         │
                         ▼
    for k, v in load_weight(config.model_path, self.device):
        │
        ▼
    ┌───────────────────────────────────────────────────────────┐
    │ v = v.to(self.dtype)  # 转换为目标精度                       │
    │                                                            │
    │ ┌─────────────────────────────────────────────────────┐    │
    │ │ config.quantization == "int8" and                   │    │
    │ │ _should_quantize(k)?                                │    │
    │ └───────────────┬─────────────────────────────────────┘    │
    │            Yes   │   No                                    │
    │                 │                                          │
    │                 ▼                                          │
    │   ┌───────────────────────────┐                            │
    │   │ 量化权重                   │                            │
    │   │                           │                            │
    │   │ qweight, scale =          │                            │
    │   │   dynamic_quantize_int8_  │                            │
    │   │   per_channel(v)          │                            │
    │   │                           │                            │
    │   │ state_dict[k.replace(     │                            │
    │   │   ".weight", ".qweight")] │                            │
    │   │   = qweight               │                            │
    │   │                           │                            │
    │   │ state_dict[k.replace(     │                            │
    │   │   ".weight", ".scale")]   │                            │
    │   │   = scale                 │                            │
    │   └───────────┬───────────────┘                            │
    │                 │                                          │
    │                 ▼                                          │
    │   ┌───────────────────────────┐                            │
    │   │ 保留原始权重               │                            │
    │   │                           │                            │
    │   │ state_dict[k] = v         │                            │
    │   └───────────┬───────────────┘                            │
    │                 │                                          │
    │                 │                                          │
    └─────────────────┼──────────────────────────────────────────┘
                      │
                      ▼
    return state_dict
                      │
                      ▼
    model.load_state_dict(state_dict)
```

#### 5.2.4 LinearLayer 前向传播

```
┌─────────────────────────────────────────────────────────────────┐
│ _LinearTPImpl.forward(x: torch.Tensor) -> torch.Tensor          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ is_quantized()?     │
              │ qweight is not None │
              └──────────┬──────────┘
                    Yes  │  No
                    │    │
                    │    └───────────────────────────┐
                    │                                │
                    ▼                                │
    ┌───────────────────────────────────┐            │
    │ 反量化权重                         │            │
    │                                   │            │
    │ weight = dequantize_int8_per_     │            │
    │   channel(                        │            │
    │     qweight=self.qweight,         │            │
    │     scale=self.scale,             │            │
    │     dtype=x.dtype                 │            │
    │   )                               │            │
    │                                   │            │
    │ # 反量化过程：                     │            │
    │ # scale.view(-1, 1)               │            │
    │ # weight = qweight.to(dtype) *    │            │
    │ #          scale.to(dtype)        │            │
    └───────────────┬───────────────────┘            │
                    │                                │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────┐
    │ Step 3: 执行矩阵乘法                                       │
    │                                                            │
    │   output = F.linear(x, weight, self.bias)                 │
    │                                                            │
    │   # 对于每个样本和输出通道：                                │
    │   # output[b, i] = sum_j(x[b, j] * weight[i, j]) + bias[i]│
    └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          返回：output [batch, out_features]
```

#### 5.2.5 _should_quantize 判断逻辑

```python
def _should_quantize(name: str) -> bool:
    """判断权重是否应该量化"""
    quantize_suffixes = [
        ".qkv_proj.weight",    # QKV 投影
        ".q_proj.weight",
        ".k_proj.weight",
        ".v_proj.weight",
        ".o_proj.weight",      # Output 投影
        ".gate_proj.weight",   # Gate 投影 (MLP)
        ".up_proj.weight",     # Up 投影 (MLP)
        ".down_proj.weight",   # Down 投影 (MLP)
        ".gate_up_proj.weight",# 融合的 Gate+Up 投影
        ".lm_head.weight",     # LM Head
    ]
    return any(name.endswith(suffix) for suffix in quantize_suffixes)
```

不量化的层：
- LayerNorm / RMSNorm 参数（对精度敏感）
- Embedding 层（可选，当前未量化）
- RoPE 相关参数（频率计算需要高精度）

### 5.3 内存分析

以 Qwen2.5-7B 为例：

| 组件 | BF16 大小 | INT8 大小 | 节省 |
|------|----------|----------|------|
| Attention 权重 | ~2.5 GB | ~1.25 GB | 50% |
| MLP 权重 | ~5.0 GB | ~2.5 GB | 50% |
| LM Head | ~0.5 GB | ~0.25 GB | 50% |
| **总计** | **~8 GB** | **~4 GB** | **~50%** |

Scale 因子开销很小（约 10-20 MB），相对于权重节省可忽略不计。

### 5.4 使用方式

```bash
# 命令行
python -m minisgl.server --model Qwen2.5-7B --quant int8

# Python API
from minisgl import LLM
llm = LLM(model_path="...", quantization="int8")
```

---

## 6. Metrics Endpoint 实现

### 6.1 技术背景

Mini-SGLang 提供 `/metrics` 端点，以 Prometheus 格式暴露系统监控指标，便于运维和性能分析。

### 6.2 核心实现 (`minisgl/metrics/collector.py`)

#### 6.2.1 请求生命周期指标收集

```
┌─────────────────────────────────────────────────────────────────┐
│                    请求生命周期指标收集                           │
└─────────────────────────────────────────────────────────────────┘

时间线 ───────────────────────────────────────────────────────────►

  请求到达              首 token              输出 token              请求完成
     │                    │                     │                       │
     │                    │                     │                       │
     ▼                    ▼                     ▼                       ▼
┌─────────┐         ┌──────────┐         ┌───────────┐          ┌──────────┐
│request_ │         │request_  │         │request_   │          │request_  │
│arrived  │         │first_tok │         │output_tok │          │completed │
│()       │         │en()      │         │()         │          │()        │
└────┬────┘         └────┬─────┘         └─────┬─────┘          └────┬─────┘
     │                   │                     │                     │
     │                   │                     │                     │
     ▼                   ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MetricsCollector 内部状态更新                                            │
│                                                                          │
│  _active_requests[uid] = RequestMetrics(                                 │
│    uid=uid,                                                              │
│    arrival_time=now,                                                     │
│    num_input_tokens=input_len                                            │
│  )                                                                       │
│  _total_input_tokens += input_len                                        │
│                                                                          │
│                       req.first_token_time = now                         │
│                       req.queue_time = first_token_time - arrival_time   │
│                                                                          │
│                                              req.num_output_tokens += 1  │
│                                              req.last_token_time = now   │
│                                              _throughput_window.append() │
│                                                                          │
│                                                                    req = │
│                                                              _active_    │
│                                                              requests.   │
│                                                              pop(uid)    │
│                                                                    │     │
│                                                                    ▼     │
│                                                      _completed_metrics. │
│                                                      append(req)         │
│                                                      _total_completed++  │
│                                                      _total_output_      │
│                                                      tokens += req.      │
│                                                      num_output_tokens   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 MetricsCollector 核心数据结构

```python
class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        # 活跃请求字典：uid -> RequestMetrics
        self._active_requests: Dict[int, RequestMetrics] = {}

        # 已完成请求历史记录（环形队列）
        self._completed_metrics: deque[RequestMetrics] = deque(maxlen=max_history)

        # 计数器
        self._total_completed: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        # 吞吐量计算窗口：(timestamp, token_count)
        self._throughput_window: deque[tuple[float, int]] = deque(maxlen=1000)

        # 队列状态
        self._queued_count: int = 0
        self._active_count: int = 0

        # KV Cache 指标
        self._num_used_tokens: int = 0
        self._max_total_num_tokens: int = 0
        self._get_used_tokens_fn: Optional[Callable[[], int]] = None
        self._get_max_tokens_fn: Optional[Callable[[], int]] = None

        # 缓存命中率指标
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_hit_tokens: int = 0
        self._cache_prefill_tokens: int = 0
```

#### 6.2.3 get_snapshot() 指标计算流程

```
┌─────────────────────────────────────────────────────────────────┐
│ MetricsCollector.get_snapshot() -> MetricsSnapshot              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
    now = time.monotonic()
                         │
                         ▼
    throughput = _calculate_throughput()
      ┌───────────────────────────────────────────────────────────┐
      │ if not _throughput_window: return 0.0                     │
      │                                                            │
      │ now = time.monotonic()                                    │
      │ window_start = _throughput_window[0][0]                   │
      │ duration = now - window_start                             │
      │                                                            │
      │ if duration <= 0: return 0.0                              │
      │                                                            │
      │ total_tokens = sum(count for _, count in _throughput_window)│
      │ return total_tokens / duration                            │
      └───────────────────────────────────────────────────────────┘
                         │
                         ▼
    # 计算 TTFT 统计
    ttft_values = [
        req.ttft for req in _completed_metrics
        if req.ttft is not None
    ]
    ttft_values.sort()

    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0
    p50_ttft = _percentile(ttft_values, 50)
    p90_ttft = _percentile(ttft_values, 90)
    p99_ttft = _percentile(ttft_values, 99)
                         │
                         ▼
    # 计算队列时间统计
    queue_times = [
        req.queue_time for req in _completed_metrics
        if req.queue_time > 0
    ]
    queue_times.sort()

    avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0.0
    p50_queue_time = _percentile(queue_times, 50)
    p99_queue_time = _percentile(queue_times, 99)
                         │
                         ▼
    # 获取 KV Cache 信息
    num_used_tokens = 0
    max_total_num_tokens = 0
    token_usage = 0.0

    if _get_used_tokens_fn and _get_max_tokens_fn:
        try:
            num_used_tokens = _get_used_tokens_fn()
            max_total_num_tokens = _get_max_tokens_fn()
            if max_total_num_tokens > 0:
                token_usage = num_used_tokens / max_total_num_tokens
        except Exception:
            pass
                         │
                         ▼
    return MetricsSnapshot(
        timestamp=now,
        running_requests=_active_count,
        queued_requests=_queued_count,
        completed_requests=_total_completed,
        total_input_tokens=_total_input_tokens,
        total_output_tokens=_total_output_tokens,
        throughput_tokens_per_sec=throughput,
        avg_ttft=avg_ttft,
        p50_ttft=p50_ttft,
        p90_ttft=p90_ttft,
        p99_ttft=p99_ttft,
        num_used_tokens=num_used_tokens,
        max_total_num_tokens=max_total_num_tokens,
        token_usage=token_usage,
        cache_hit_rate=_cache_hit_tokens / _cache_prefill_tokens,
        cache_hit_tokens=_cache_hit_tokens,
        cache_prefill_tokens=_cache_prefill_tokens,
        cache_hits=_cache_hits,
        cache_misses=_cache_misses,
        avg_queue_time=avg_queue_time,
        p50_queue_time=p50_queue_time,
        p99_queue_time=p99_queue_time,
    )
```

#### 6.2.4 Percentile 计算方法

```python
def _percentile(self, sorted_values: list, percentile: int) -> float:
    """Calculate percentile from sorted values."""
    if not sorted_values:
        return 0.0
    index = int(len(sorted_values) * percentile / 100)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]
```

示例：
- 100 个请求，p50 索引 = int(100 * 50 / 100) = 50
- 100 个请求，p90 索引 = int(100 * 90 / 100) = 90
- 100 个请求，p99 索引 = int(100 * 99 / 100) = 99

### 6.3 指标上报流程

```
┌─────────────────────────────────────────────────────────────────┐
│              Scheduler 定期上报指标（每 10 个 batch）                  │
└─────────────────────────────────────────────────────────────────┘

Scheduler._process_last_data():
    │
    ├─ 处理每个请求的输出 token
    │   ├─ metrics_collector.request_first_token(uid)  # 首个 token
    │   └─ metrics_collector.request_output_token(uid)  # 每个 token
    │
    ├─ 请求完成时
    │   └─ metrics_collector.request_completed(uid)
    │
    └─ 更新计数
        ├─ metrics_collector.set_queued_count(len(prefill_manager.pending_list))
        ├─ metrics_collector.set_active_count(len(decode_manager.running_reqs))
        └─ _metrics_counter += 1

        if _metrics_counter >= 10:  # 每 10 个 batch
            _send_metrics()
            _metrics_counter = 0

┌─────────────────────────────────────────────────────────────────┐
│ _send_metrics()                                                 │
│                                                                  │
│  snapshot = metrics_collector.get_snapshot()                     │
│                                                                  │
│  msg = MetricsReportMsg(                                         │
│      running_requests=snapshot.running_requests,                 │
│      queued_requests=snapshot.queued_requests,                   │
│      completed_requests=snapshot.completed_requests,             │
│      total_input_tokens=snapshot.total_input_tokens,             │
│      total_output_tokens=snapshot.total_output_tokens,           │
│      throughput_tokens_per_sec=snapshot.throughput_tokens_per_sec,│
│      avg_ttft=snapshot.avg_ttft,                                 │
│      p50_ttft=snapshot.p50_ttft,                                 │
│      p90_ttft=snapshot.p90_ttft,                                 │
│      p99_ttft=snapshot.p99_ttft,                                 │
│      num_used_tokens=snapshot.num_used_tokens,                   │
│      max_total_num_tokens=snapshot.max_total_num_tokens,         │
│      token_usage=snapshot.token_usage,                           │
│      cache_hit_rate=snapshot.cache_hit_rate,                     │
│      cache_hit_tokens=snapshot.cache_hit_tokens,                 │
│      cache_prefill_tokens=snapshot.cache_prefill_tokens,         │
│      cache_hits=snapshot.cache_hits,                             │
│      cache_misses=snapshot.cache_misses,                         │
│      avg_queue_time=snapshot.avg_queue_time,                     │
│      p50_queue_time=snapshot.p50_queue_time,                     │
│      p99_queue_time=snapshot.p99_queue_time,                     │
│  )                                                               │
│                                                                  │
│  send_metrics_msg(msg)  # ZMQ Push to API Server                 │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ZMQ: zmq_frontend_addr
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ API Server - FrontendManager                                    │
│                                                                  │
│  async def listen():                                            │
│      while True:                                                │
│          msg = await recv_tokenizer.get()                       │
│          if isinstance(msg, MetricsReportMsg):                  │
│              update_metrics(msg)                                │
│                └─ self.metrics.update_from_msg(msg)               │
│                   └─ MetricsState 字段更新                        │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 /metrics 端点响应流程

```
┌─────────────────────────────────────────────────────────────────┐
│ GET /metrics                                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ api_server.metrics()                                            │
│                                                                  │
│  state = get_global_state()  # FrontendManager                  │
│  return PlainTextResponse(                                      │
│      state.get_metrics_prometheus(),                            │
│      media_type="text/plain"                                    │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ FrontendManager.get_metrics_prometheus()                        │
│   └─ self.metrics.to_prometheus()                               │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ MetricsState.to_prometheus() -> str                             │
│                                                                  │
│  lines = [                                                      │
│      # Request counts                                           │
│      "# HELP minisgl_running_requests Number of running requests",
│      "# TYPE minisgl_running_requests gauge",                   │
│      f"minisgl_running_requests {self.running_requests}",       │
│      ...                                                        │
│                                                                  │
│      # Token counts                                             │
│      "# HELP minisgl_total_input_tokens Total number of input...",
│      "# TYPE minisgl_total_input_tokens counter",               │
│      f"minisgl_total_input_tokens {self.total_input_tokens}",   │
│      ...                                                        │
│                                                                  │
│      # Throughput                                               │
│      "# HELP minisgl_output_throughput Output throughput...",   │
│      "# TYPE minisgl_output_throughput gauge",                  │
│      f"minisgl_output_throughput {self.throughput_tokens_per_sec:.2f}",
│      ...                                                        │
│                                                                  │
│      # TTFT metrics                                             │
│      "# HELP minisgl_ttft_avg Average time to first token...",  │
│      "# TYPE minisgl_ttft_avg gauge",                           │
│      f"minisgl_ttft_avg {self.avg_ttft:.6f}",                   │
│      ...                                                        │
│  ]                                                              │
│  return "\n".join(lines) + "\n"                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 API 端点 (`minisgl/server/api_server.py`)

```python
@app.get("/metrics")
async def metrics():
    """暴露 Prometheus 格式指标"""
    state = get_global_state()
    return PlainTextResponse(state.get_metrics_prometheus(), media_type="text/plain")
```

### 6.4 Prometheus 格式输出

```prometheus
# HELP minisgl_running_requests Number of running requests
# TYPE minisgl_running_requests gauge
minisgl_running_requests 5
# HELP minisgl_queued_requests Number of queued requests
# TYPE minisgl_queued_requests gauge
minisgl_queued_requests 10
# HELP minisgl_completed_requests Total number of completed requests
# TYPE minisgl_completed_requests counter
minisgl_completed_requests 100
# HELP minisgl_total_input_tokens Total number of input tokens processed
# TYPE minisgl_total_input_tokens counter
minisgl_total_input_tokens 50000
# HELP minisgl_total_output_tokens Total number of output tokens generated
# TYPE minisgl_total_output_tokens counter
minisgl_total_output_tokens 25000
# HELP minisgl_output_throughput Output throughput in tokens per second
# TYPE minisgl_output_throughput gauge
minisgl_output_throughput 1234.56
# HELP minisgl_ttft_avg Average time to first token in seconds
# TYPE minisgl_ttft_avg gauge
minisgl_ttft_avg 0.050000
# HELP minisgl_ttft_p50 P50 time to first token in seconds
# TYPE minisgl_ttft_p50 gauge
minisgl_ttft_p50 0.045000
# HELP minisgl_ttft_p90 P90 time to first token in seconds
# TYPE minisgl_ttft_p90 gauge
minisgl_ttft_p90 0.080000
# HELP minisgl_ttft_p99 P99 time to first token in seconds
# TYPE minisgl_ttft_p99 gauge
minisgl_ttft_p99 0.120000
# HELP minisgl_cache_hit_rate The prefix cache hit rate (tokens)
# TYPE minisgl_cache_hit_rate gauge
minisgl_cache_hit_rate 0.3500
```

### 6.5 指标报告流程

```
1. 请求到达 → metrics_collector.request_arrived()
2. 首 token 生成 → metrics_collector.request_first_token()
3. 每个输出 token → metrics_collector.request_output_token()
4. 请求完成 → metrics_collector.request_completed()
5. 每 10 个 batch → 通过 ZMQ 发送 MetricsReportMsg 到 API Server
6. 用户访问 /metrics → 返回 Prometheus 格式指标
```

### 6.6 缓存命中率统计

在 `PrefillAdder` 中记录缓存命中：

```python
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len

    if self.metrics_collector is not None:
        input_len = req.input_len
        if input_len > 0:
            self.metrics_collector.record_cache_tokens(
                hit_tokens=cached_len,
                total_tokens=input_len
            )
```

---

## 7. 总结

Mini-SGLang 作为一个轻量级推理框架，通过以下关键技术实现了高性能和易用性的平衡：

### 7.1 架构特点

- **多进程分布式设计**：API Server、Tokenizer、Scheduler、Detokenizer 各司其职，通过 ZMQ 和 NCCL 高效通信
- **张量并行支持**：原生支持多 GPU 部署，通过 `--tp` 参数灵活配置
- **清晰的代码组织**：约 5,000 行 Python 代码，模块化设计便于理解

### 7.2 核心技术贡献

| 技术 | 解决的问题 | 实现亮点 |
|------|-----------|---------|
| Chunked Prefill | 长上下文 OOM | 动态分块、预算控制 |
| Radix Attention | 前缀重复计算 | 树状缓存、LRU 驱逐 |
| INT8 动态量化 | 显存占用 | 逐通道量化、运行时反量化 |
| Metrics Endpoint | 运维监控 | Prometheus 兼容、细粒度指标 |

### 7.3 适用场景

- **教学与研究**：清晰的代码结构适合学习 LLM 推理系统原理
- **原型开发**：轻量级设计便于快速验证新想法
- **生产参考**：可作为构建生产级推理系统的参考实现

---

## 附录

### A. 配置参数总览

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径 | 必填 |
| `--tp` | 张量并行度 | 1 |
| `--max-prefill-length` | Prefill chunk 大小 | 自动 |
| `--page-size` | KVCache 页大小 | 64 |
| `--quantization` | 量化方案 (int8/none) | none |
| `--cache` | 缓存策略 (radix/naive) | radix |
| `--port` | 服务端口 | 1919 |

### B. 参考资料

- [SGLang 项目](https://github.com/sgl-project/sglang)
- [Sarathi-Serve: Chunked Prefill](https://arxiv.org/abs/2403.02310)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
