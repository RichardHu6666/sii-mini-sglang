# MiniSGL 服务启动调用链路分析

## 一、启动流程总览

```sequence
participant Main as 主进程
participant API as API Server
participant SubProc as 后端启动回调
participant Sched as Scheduler 进程
participant Detok as Detokenizer 进程
participant Tok as Tokenizer 进程

Main->>Main: 1. 解析命令行参数
Main->>API: 2. 启动 API Server
API->>API: 3. 初始化 FrontendManager
API->>API: 4. 创建 ZMQ 消息队列
API->>SubProc: 5. 调用后端启动回调
SubProc->>SubProc: 6. 设置多进程 spawn 模式
SubProc->>Sched: 7. 启动 Scheduler 进程
SubProc->>Detok: 8. 启动 Detokenizer 进程
SubProc->>Tok: 9. 启动 Tokenizer 进程

note over Sched: 初始化 Engine\n初始化各管理器\n进入主循环
note over Detok: 加载分词器\n初始化处理模块
note over Tok: 加载分词器\n初始化处理模块

Sched-->>SubProc: 10. 发送就绪信号
Detok-->>SubProc: 11. 发送就绪信号
Tok-->>SubProc: 12. 发送就绪信号
SubProc-->>API: 13. 所有子进程就绪
API->>API: 14. 启动 Uvicorn HTTP 服务

note over API,Sched: 服务启动完成，等待请求
```

### 1.3 启动流程说明

服务启动的完整流程如下：

1. **解析命令行参数** — 主进程解析用户传入的命令行参数（模型路径、TP 大小、内存比例等）。
2. **启动 API Server** — 主进程调用 `run_api_server()` 启动 FastAPI 服务。
3. **初始化 FrontendManager** — API Server 初始化前端管理器，用于管理用户连接和消息路由。
4. **创建 ZMQ 消息队列** — 创建用于进程间通信的 ZMQ 队列（接收/发送 Tokenize 和 Detokenize 消息）。
5. **调用后端启动回调** — API Server 调用 `start_subprocess()` 启动后端子进程。
6. **设置多进程 spawn 模式** — 配置 multiprocessing 使用 spawn 方式创建子进程。
7. **启动 Scheduler 进程** — 创建 Scheduler 进程，负责推理调度和模型执行。
8. **启动 Detokenizer 进程** — 创建 Detokenizer 进程，负责将 Token 转换为文本。
9. **启动 Tokenizer 进程** — 创建 Tokenizer 进程，负责将文本转换为 Token。
10. **发送就绪信号** — 各子进程完成初始化后，通过 ack_queue 发送就绪信号。
11. **等待所有子进程就绪** — 主进程等待接收所有子进程的就绪信号。
12. **启动 Uvicorn HTTP 服务** — 所有子进程就绪后，启动 HTTP 服务器监听用户请求。

---

## 二、各进程启动时的模块调用关系

### 2.1 Scheduler 进程启动链路

```
_run_scheduler()
    │
    ▼
Scheduler.__init__(config)
    │
    ├──► Engine.__init__(config)
    │       │
    │       ├──► set_tp_info()                    # 设置 TP 分布式信息
    │       ├──► _init_communication()            # 初始化进程间通信
    │       │       ├──► torch.distributed.init_process_group()
    │       │       └──► enable_pynccl_distributed()
    │       │
    │       ├──► create_model()                   # 在 meta 设备创建模型
    │       ├──► load_weight()                    # 加载模型权重
    │       │       └──► _dynamic_quantize_int8_per_channel() [可选]
    │       │
    │       ├──► create_kvcache_pool()            # 创建 KV Cache 池
    │       ├──► create_attention_backend()       # 创建 Attention 后端
    │       ├──► create_moe_backend()             # 创建 MoE 后端 [如果是 MoE 模型]
    │       ├──► Sampler.__init__()               # 初始化采样器
    │       │
    │       └──► GraphRunner.__init__()           # 捕获 CUDA Graph
    │               └──► _capture_cuda_graph()
    │
    ├──► TableManager.__init__()                  # 初始化 Token 表管理器
    ├──► CacheManager.__init__()                  # 初始化 KV Cache 管理器
    │       └──► create_prefix_cache()            # 创建前缀缓存
    │
    ├──► DecodeManager.__init__()                 # 初始化 Decode 管理器
    ├──► PrefillManager.__init__()                # 初始化 Prefill 管理器
    ├──► MetricsCollector.__init__()              # 初始化指标收集器
    │
    └──► SchedulerIOMixin.__init__()              # 初始化 ZMQ 通信
            ├──► ZmqPullQueue()                   # 接收 Backend 消息
            └──► ZmqPushQueue()                   # 发送 Metrics 消息
```

---

### 2.2 Tokenizer 进程启动链路

```
tokenize_worker()
    │
    ├──► ZmqPushQueue() x2                        # 创建发送队列
    │       ├──► backend_addr (发往 Scheduler)
    │       └──► frontend_addr (发往 API Server)
    │
    ├──► ZmqPullQueue()                           # 创建接收队列
    │       └──► tokenizer_addr (接收前端请求)
    │
    ├──► load_tokenizer()                         # 加载分词器模型
    │
    ├──► TokenizeManager.__init__()               # 初始化分词管理器
    │
    └──► DetokenizeManager.__init__()             # 初始化逆分词管理器
```

---

### 2.3 API Server 启动链路

```
run_api_server()
    │
    ├──► FrontendManager.__init__()
    │       ├──► ZmqAsyncPullQueue()              # 异步接收 Detokenizer 消息
    │       └──► ZmqAsyncPushQueue()              # 异步发送 Tokenize 请求
    │
    ├──► start_subprocess()                       # 启动后端子进程
    │
    └──► uvicorn.run()                            # 启动 HTTP 服务器
            │
            └──► FastAPI 路由注册
                    ├──► POST /generate
                    ├──► POST /v1/chat/completions
                    ├──► GET /v1/models
                    └──► GET /metrics
```

---

## 三、各模块执行的核心操作

### 3.1 Engine 模块

| 操作 | 说明 |
|------|------|
| 设置 TP 分布式 | 初始化 NCCL/GLOO 通信组，启用 PyNCCL |
| 加载模型权重 | 从 HuggingFace 加载，可选 int8 量化 |
| 分配 KV Cache | 根据 GPU 内存计算可分配页数，创建 K/V 缓存池 |
| 初始化 Page Table | 创建 token 到物理地址的映射表 |
| 创建 Attention 后端 | 根据硬件选择 FlashAttention/FlashInfer/TRTLLM |
| 创建 MoE 后端 | 如果是 MoE 模型，初始化 fused_moe 后端 |
| 捕获 CUDA Graph | 预捕获静态计算图加速推理 |

### 3.2 CacheManager 模块

| 操作 | 说明 |
|------|------|
| 初始化前缀缓存 | 创建 Radix Tree 或 Naive 缓存 |
| 分配 free_slots | 初始化所有页面为空闲状态 |
| 锁机制 | 为缓存句柄提供锁保护 |

### 3.3 TableManager 模块

| 操作 | 说明 |
|------|------|
| 初始化 token_pool | 预分配 token ID 池 (max_running_req × max_seq_len) |
| 初始化 page_table | 创建请求到 KV Cache 页面的映射 |

### 3.4 Scheduler 主循环

```
Scheduler.run_forever()
    │
    └──► overlap_loop() / normal_loop()  [每轮迭代]
            │
            ├──► receive_msg()            # 接收 ZMQ 消息
            │       ├──► UserMsg          # 处理新请求
            │       ├──► AbortMsg         # 处理中止请求
            │       └──► ExitMsg          # 处理退出信号
            │
            ├──► _schedule_next_batch()   # 调度下一批
            │       ├──► PrefillManager.schedule_next_batch()
            │       │       ├──► CacheManager.match_req()   # 缓存匹配
            │       │       ├──► TableManager.allocate()    # 分配资源
            │       │       └──► CacheManager.lock()        # 锁定缓存
            │       │
            │       └──► DecodeManager.schedule_next_batch()
            │
            ├──► _forward()               # 执行 Forward
            │       ├──► GraphRunner.replay() 或 model.forward()
            │       ├──► Sampler.sample()
            │       └──► 异步拷贝到 CPU
            │
            └──► _process_last_data()     # 处理上一批结果
                    ├──► 更新 request 状态
                    ├──► CacheManager.cache_req()   # 缓存 KV
                    └──► 发送 DetokenizeMsg
```

### 3.5 Tokenizer Worker 主循环

```
tokenize_worker()
    │
    └──► while True  [每轮迭代]
            │
            ├──► recv_listener.get()        # 接收 ZMQ 消息
            │
            ├──► DetokenizeMsg → DetokenizeManager.detokenize()
            │       └──► 发送 UserReply 到前端
            │
            ├──► TokenizeMsg → TokenizeManager.tokenize()
            │       └──► 发送 UserMsg 到 Scheduler
            │
            └──► AbortMsg → 发送 AbortBackendMsg 到 Scheduler
```

---

## 四、进程间通信拓扑

### 4.1 ZMQ 通信架构

```sequence
participant API as API Server
participant TZ as Tokenizer Worker
participant S0 as Scheduler-0
participant S1 as Scheduler-1
participant SN as Scheduler-N

API->>TZ: 1. zmq_tokenizer_addr\n(TokenizeMsg)
TZ->>S0: 2. zmq_backend_addr\n(UserMsg)
TZ->>S1: 2. zmq_backend_addr\n(UserMsg)
TZ->>SN: 2. zmq_backend_addr\n(UserMsg)

note over S0,SN: NCCL 通信\n(Tensor Parallel)
S0<->S1: 3. AllReduce
S1<->SN: 3. AllReduce

TZ->>API: 4. zmq_frontend_addr\n(UserReply / MetricsReportMsg)
```

### 4.2 消息类型流转

```sequence
participant FE as 前端
participant API as API Server
participant TZ as Tokenizer
participant SCHED as Scheduler

FE->>API: 1. HTTP 请求
API->>TZ: 2. TokenizeMsg
TZ->>TZ: 3. tokenize()
TZ->>SCHED: 4. UserMsg(input_ids)

note over SCHED: 5. 处理请求\n调度 → Forward → 采样

SCHED->>TZ: 6. DetokenizeMsg(next_token)
TZ->>TZ: 7. detokenize()
TZ->>API: 8. UserReply(incremental_output)
API-->>FE: 9. 流式响应
```

---

## 五、启动时序图

### 5.1 服务启动时序 (TP=1)

```sequence
participant Main as 主进程
participant API as API Server
participant Sched as Scheduler
participant Detok as Detokenizer
participant TZ as Tokenizer

Main->>API: 1. parse_args()
Main->>TZ: 2. start_subprocess()
Main->>Sched: 3. Process(_run_scheduler)

note over Sched: Engine 初始化
Sched->>Sched: 4. 创建模型
Sched->>Sched: 5. 加载权重
Sched->>Sched: 6. 分配 KV Cache
Sched->>Sched: 7. 捕获 CUDA Graph

note over Detok,TZ: 初始化
Detok->>Detok: 8. 加载分词器
TZ->>TZ: 9. 加载分词器

Sched-->>API: 10. ack_queue.put("ready")
API-->>Main: 11. ack_queue.get()
note over API,Main: 所有子进程就绪

Main->>API: 12. uvicorn.run()
note over API: HTTP 服务启动

note over Sched: run_forever()
note over Detok: tokenize_worker()
note over TZ: tokenize_worker()
```

---

### 5.2 Scheduler 内部初始化时序

```sequence
participant Scheduler
participant Engine
participant TableMgr as TableManager
participant CacheMgr as CacheManager
participant PrefillMgr
participant DecodeMgr
participant Metrics as MetricsCollector
participant IO as SchedulerIOMixin

Scheduler->>Engine: 1. Engine.__init__()
Engine->>Engine: 2. set_tp_info()
Engine->>Engine: 3. _init_communication()
Engine->>Engine: 4. create_model()
Engine->>Engine: 5. load_weight()
Engine->>Engine: 6. create_kvcache_pool()
Engine->>Engine: 7. create_attention_backend()
Engine->>Engine: 8. Sampler.__init__()
Engine->>Engine: 9. GraphRunner.__init__()

Scheduler->>TableMgr: 10. TableManager.__init__()
Scheduler->>CacheMgr: 11. CacheManager.__init__()
CacheMgr->>CacheMgr: 12. create_prefix_cache()

Scheduler->>DecodeMgr: 13. DecodeManager.__init__()
Scheduler->>PrefillMgr: 14. PrefillManager.__init__()
Scheduler->>Metrics: 15. MetricsCollector.__init__()
Scheduler->>IO: 16. SchedulerIOMixin.__init__()
IO->>IO: 17. ZmqPullQueue()
IO->>IO: 18. ZmqPushQueue()
```

---

### 5.3 请求处理时序

```sequence
participant User as 用户
participant API as API Server
participant TZ as Tokenizer
participant SCHED as Scheduler
participant ENGINE as Engine

User->>API: 1. POST /v1/chat/completions
API->>TZ: 2. TokenizeMsg (ZMQ)
TZ->>TZ: 3. tokenize()
TZ->>SCHED: 4. UserMsg (ZMQ)

SCHED->>SCHED: 5. PrefillManager.add_one_req()
SCHED->>SCHED: 6. schedule_next_batch()
SCHED->>SCHED: 7. CacheManager.match_req()
SCHED->>SCHED: 8. CacheManager.allocate_paged()

SCHED->>ENGINE: 9. forward_batch()
ENGINE->>ENGINE: 10. GraphRunner.replay() / model.forward()
ENGINE->>ENGINE: 11. Sampler.sample()
ENGINE-->>SCHED: 12. ForwardOutput

SCHED->>SCHED: 13. 更新 request 状态
SCHED->>TZ: 14. DetokenizeMsg (ZMQ)
TZ->>TZ: 15. detokenize()
TZ->>API: 16. UserReply (ZMQ)
API-->>User: 17. 流式返回

loop 生成每个 token
    SCHED->>SCHED: 18. schedule_decode()
    SCHED->>ENGINE: 19. forward_batch()
    ENGINE->>ENGINE: 20. Sampler.sample()
    ENGINE-->>SCHED: 21. next_token
    SCHED->>TZ: 22. DetokenizeMsg
    TZ->>TZ: 23. detokenize()
    TZ->>API: 24. UserReply
    API-->>User: 25. 流式返回
end

note over User: 生成完成
```

---

### 5.4 请求生命周期

从用户发起请求到接收响应，整个生命周期包含以下步骤：

1. **用户发送请求** — 用户向 API 服务器发送 HTTP 请求（如 `/v1/chat/completions`）。
2. **API 服务器转发** — API 服务器将请求封装为 `TokenizeMsg`，通过 ZMQ 转发给 Tokenizer 进程。
3. **分词处理** — Tokenizer 将输入文本转换为 Token ID 序列，封装为 `UserMsg` 发送给 Scheduler。
4. **请求调度** — Scheduler 将请求加入待处理队列，进行批次调度（Prefill 或 Decode）。
5. **引擎计算** — Scheduler 触发 Engine 执行 Forward 计算，采样生成下一个 Token。
6. **去分词处理** — Scheduler 将生成的 Token 封装为 `DetokenizeMsg` 发送给 Detokenizer。
7. **文本转换** — Detokenizer 将 Token ID 转换回文本片段，封装为 `UserReply` 返回给 API 服务器。
8. **流式返回** — API 服务器将文本片段流式返回给用户，直到生成完成。

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  用户   │     │API Server│     │Tokenizer│     │Scheduler│     │Detokeniz│     │  用户   │
│ (请求)  │────►│ (HTTP)  │────►│ (分词)  │────►│ (调度)  │────►│er (去词)│────►│ (响应)  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
                     │               │               │               │
                     │           ZMQ 消息          │           ZMQ 消息
                     │           TokenizeMsg       │           UserReply
                     │               │               │               │
                     │               ▼               │               │
                     │           UserMsg             │               │
                     │               │               │               │
                     │               ▼               │               │
                     │           DetokenizeMsg ◄─────┘               │
                     │               │                               │
                     └───────────────────────────────────────────────┘
```

---

## 六、关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| --model-path | Qwen2.5-14B-Instruct | 模型路径 |
| --tensor-parallel-size | 1 | TP 并行度 |
| --memory-ratio | 0.9 | GPU 内存使用比例 |
| --max-running-requests | 100 | 最大并发请求数 |
| --max-prefill-length | 4096 | 最大 Prefill 长度 |
| --page-size | 64 | KV Cache 页大小 |
| --attention-backend | auto | 自动选择后端 |
| --num-tokenizer | 0 | Tokenizer 进程数 |
