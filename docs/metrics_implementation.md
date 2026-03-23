# Mini-SGLang /metrics Endpoint 实现文档

## 1. 整体架构

Mini-SGLang 的 `/metrics` endpoint 采用分层设计，从指标采集到暴露的完整数据流如下：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Scheduler Process                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MetricsCollector                              │    │
│  │  - request_arrived()      - request_first_token()                │    │
│  │  - request_output_token() - request_completed()                  │    │
│  │  - record_cache_hit()     - set_kv_cache_info()                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                            │                                              │
│                            ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MetricsReportMsg (ZMQ)                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Tokenizer Process                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Forward metrics messages (pass-through)             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API Server (FastAPI)                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MetricsState                                  │    │
│  │              - update_from_msg()                                 │    │
│  │              - to_prometheus()                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                            │                                              │
│                            ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              GET /metrics → Prometheus Format                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 核心组件

| 组件 | 文件位置 | 职责 |
|------|----------|------|
| `MetricsCollector` | `python/minisgl/metrics/collector.py` | 在 Scheduler 中采集请求生命周期指标 |
| `MetricsReportMsg` | `python/minisgl/message/backend.py` | 通过 ZMQ 传输指标数据 |
| `MetricsState` | `python/minisgl/server/api_server.py` | 在 API Server 中维护最新指标状态 |
| `/metrics` endpoint | `python/minisgl/server/api_server.py` | 暴露 Prometheus 格式指标 |

### 1.2 数据流

1. **指标采集**：`Scheduler` 在处理请求时调用 `MetricsCollector` 的方法记录事件
2. **初始化上报**：Scheduler 初始化完成后立即调用 `_send_metrics()` 发送初始快照（确保 KV cache 总容量等非零）
3. **周期性上报**：之后每 10 个 batch，Scheduler 调用 `_send_metrics()` 将快照通过 ZMQ 发送
4. **消息转发**：Tokenizer 进程过滤并转发 `MetricsReportMsg` 到 Frontend
5. **状态更新**：API Server 监听 ZMQ 消息，更新 `MetricsState`
6. **指标暴露**：HTTP GET `/metrics` 返回 Prometheus exposition format 文本

---

## 2. 指标详细实现

### 2.1 请求计数指标

#### minisgl_running_requests (gauge)

**含义**：当前正在 **decode 阶段** 处理的请求数（不包括 prefill 阶段的请求）

**采集点**：`MetricsCollector._active_count`，由 Scheduler 定期更新

```python
# scheduler.py: _process_last_data()
# Prefill 完成后，将请求加入 decode_manager.running_reqs
elif batch.is_prefill:  # for prefill, non-chunk req
    self.decode_manager.running_reqs.add(req)
    self.cache_manager.cache_req(req, finished=False)

# scheduler.py: _process_last_data()
# 定期上报 running_requests 和 queued_requests
self.metrics_collector.set_queued_count(len(self.prefill_manager.pending_list))
self.metrics_collector.set_active_count(len(self.decode_manager.running_reqs))

# scheduler.py: _forward()
# 只在 decode batch 时调用 filter_reqs，保留仍在 decode 的请求
if batch.is_decode:
    self.decode_manager.filter_reqs(forward_input.batch.reqs)

# collector.py: get_snapshot()
running_requests = self._active_count
```

**计算逻辑**：
1. 请求到达 Scheduler 时，进入 `prefill_manager.pending_list` 等待 prefill
2. Prefill 调度时，请求从 pending_list 移除，进入 prefill batch
3. Prefill forward 完成后，请求被加入 `decode_manager.running_reqs`
4. Decode forward 时，`filter_reqs` 保留仍可 decode 的请求（过滤已完成的）
5. 请求完成时，从 `running_reqs` 移除
6. Scheduler 每 batch 调用 `set_active_count()` 上报当前 decode 请求数

**重要说明**：
- `minisgl_running_requests` 仅统计 decode 阶段的请求，不包括正在 prefill 的请求
- 正在 prefill 中的请求（已调度但未完成 prefill）不会被计入任何指标
- 因此 `running_requests + queued_requests` 可能略小于系统实际处理的总请求数

---

#### minisgl_queued_requests (gauge)

**含义**：当前排队等待 prefill 处理的请求数

**采集点**：`PrefillManager.pending_list` 的长度

```python
# scheduler.py: _process_last_data()
self.metrics_collector.set_queued_count(len(self.prefill_manager.pending_list))

# collector.py
def set_queued_count(self, count: int) -> None:
    self._queued_count = count
```

**计算逻辑**：
- 请求到达时加入 `pending_list`
- Prefill 调度时从 `pending_list` 移除
- Scheduler 每 batch 调用 `set_queued_count()` 上报当前排队数

---


### 2.2 Token 指标

#### minisgl_total_input_tokens (counter)

**含义**：自服务启动以来处理的输入 token 总数

**采集点**：`request_arrived()` 时累加

```python
# collector.py: request_arrived()
def request_arrived(self, uid: int, num_input_tokens: int, ...) -> None:
    self._active_requests[uid] = RequestMetrics(
        uid=uid,
        arrival_time=now,
        num_input_tokens=num_input_tokens,
    )
    self._total_input_tokens += num_input_tokens  # 累加
```

---


---

#### minisgl_output_throughput (gauge)

**含义**：输出 token 吞吐率（tokens/秒）

**采集点**：基于滑动窗口计算

```python
# collector.py: request_output_token()
def request_output_token(self, uid: int) -> None:
    ...
    self._throughput_window.append((req.last_token_time, 1))

# collector.py: _calculate_throughput()
def _calculate_throughput(self) -> float:
    now = time.monotonic()
    window_start = self._throughput_window[0][0]
    duration = now - window_start
    total_tokens = sum(count for _, count in self._throughput_window)
    return total_tokens / duration
```

**计算逻辑**：
- 每次生成 output token 时记录时间戳和数量
- 使用 `deque(maxlen=1000)` 作为滑动窗口
- 吞吐量 = 窗口内总 token 数 / 窗口时间跨度

---

### 2.4 KV Cache 指标

#### minisgl_num_used_tokens (gauge)

**含义**：KV cache 中已使用的 token 数量

**采集点**：通过回调函数实时获取

```python
# scheduler.py: __init__()
self.metrics_collector.set_kv_cache_info(
    get_used_tokens_fn=lambda: (self.cache_manager.num_pages - len(self.cache_manager.free_slots)) * self.cache_manager.page_size,
    get_max_tokens_fn=lambda: self.cache_manager.num_pages * self.cache_manager.page_size,
)

# collector.py: get_snapshot()
if self._get_used_tokens_fn and self._get_max_tokens_fn:
    num_used_tokens = self._get_used_tokens_fn()
    max_total_num_tokens = self._get_max_tokens_fn()
    token_usage = num_used_tokens / max_total_num_tokens
```

**计算逻辑**：
- KV cache 以 page 为单位管理
- 已使用 tokens = (总页数 - 空闲页数) × 每页大小

---

#### minisgl_max_total_num_tokens (gauge)

**含义**：KV cache 池总容量（tokens）

**计算逻辑**：`num_pages * page_size`

---


### 2.5 Cache 命中指标

#### minisgl_cache_hit_rate (gauge)

**含义**：前缀缓存命中率

**采集点**：`PrefillAdder._try_allocate_one()`

```python
# prefill.py
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len  # 命中的前缀长度

    # 记录 cache hit/miss
    if self.metrics_collector is not None:
        if cached_len > 0:
            self.metrics_collector.record_cache_hit(hit=True)
        else:
            self.metrics_collector.record_cache_hit(hit=False)

# collector.py
def record_cache_hit(self, hit: bool) -> None:
    if hit:
        self._cache_hits += 1
    else:
        self._cache_misses += 1

@property
def cache_hit_rate(self) -> float:
    total = self._cache_hits + self._cache_misses
    if total == 0:
        return 0.0
    return self._cache_hits / total
```

---


## 4. 关键设计决策

### 4.1 为什么使用 ZMQ 传输指标？

- Scheduler 运行在独立进程，需要跨进程通信
- ZMQ 是 Mini-SGLang 已有的 IPC 机制
- **初始化时发送一次**：确保 KV cache 总容量等指标在 server 启动后即可正确获取
- **周期性推送**（每 10 batch）：避免频繁通信开销，同时保持指标更新

### 4.2 什么是"batch"？

在 Mini-SGLang 的调度器中，**batch 指的是每轮 forward 计算的处理单元**：

- **Prefill batch**：调度器从 pending queue 中调度请求进行 prefill（提示词处理）
- **Decode batch**：调度器调度正在 decode 阶段的请求生成下一个 token
- 调度器采用"Prefill 优先"策略，优先处理 prefill 请求，没有 prefill 请求时才处理 decode

因此"每 10 个 batch"指的是**每 10 轮 forward 计算**（可能是 prefill 或 decode），而不是特指 prefill 或 decode。

```python
# scheduler.py: _process_last_data()
# 每处理完一个 batch 后调用
self._metrics_counter += 1
if self._metrics_counter >= 10:  # 每 10 个 batch 发送一次 metrics
    self._send_metrics()
    self._metrics_counter = 0
```


## 5. 相关文件

| 文件 | 说明 |
|------|------|
| `python/minisgl/metrics/collector.py` | 核心指标采集逻辑 |
| `python/minisgl/message/backend.py` | `MetricsReportMsg` 定义 |
| `python/minisgl/message/frontend.py` | 前端消息反序列化 |
| `python/minisgl/message/tokenizer.py` | Tokenizer 消息透传 |
| `python/minisgl/scheduler/scheduler.py` | Scheduler 指标采集点 |
| `python/minisgl/scheduler/prefill.py` | Cache hit 记录 |
| `python/minisgl/scheduler/decode.py` | Decode manager 运行请求管理 |
| `python/minisgl/server/api_server.py` | `/metrics` endpoint |
| `benchmark/online/bench_simple.py` | 基准测试脚本 |
| `benchmark/online/plot_metrics.py` | 指标可视化 |

---

#### minisgl_queued_requests (gauge)

**含义**：当前排队等待处理的请求数

**采集点**：`PrefillManager.pending_list` 的长度

```python
# scheduler.py: _process_last_data()
self.metrics_collector.set_queued_count(len(self.prefill_manager.pending_list))

# collector.py
def set_queued_count(self, count: int) -> None:
    self._queued_count = count
```

**计算逻辑**：
- Scheduler 每 batch 更新 queued count
- 调用 `set_queued_count()` 设置当前pending列表长度

---

#### minisgl_completed_requests (counter)

**含义**：自服务启动以来完成的请求总数

**采集点**：`MetricsCollector._total_completed` 计数器

```python
# collector.py: request_completed()
def request_completed(self, uid: int) -> None:
    if uid in self._active_requests:
        metrics = self._active_requests.pop(uid)
        self._completed_metrics.append(metrics)
        self._total_completed += 1  # 累加计数器
```

---

### 2.2 Token 指标

#### minisgl_total_input_tokens (counter)

**含义**：自服务启动以来处理的输入 token 总数

**采集点**：`request_arrived()` 时累加

```python
# collector.py: request_arrived()
def request_arrived(self, uid: int, num_input_tokens: int, ...) -> None:
    self._active_requests[uid] = RequestMetrics(
        uid=uid,
        arrival_time=now,
        num_input_tokens=num_input_tokens,
    )
    self._total_input_tokens += num_input_tokens  # 累加
```

---

#### minisgl_total_output_tokens (counter)

**含义**：自服务启动以来生成的输出 token 总数

**采集点**：`request_completed()` 时累加

```python
# collector.py: request_output_token()
def request_output_token(self, uid: int) -> None:
    if uid in self._active_requests:
        req = self._active_requests[uid]
        req.num_output_tokens += 1  # 记录单个请求的output token

# collector.py: request_completed()
def request_completed(self, uid: int) -> None:
    ...
    self._total_output_tokens += metrics.num_output_tokens  # 累加到全局
```

---

#### minisgl_output_throughput (gauge)

**含义**：输出 token 吞吐率（tokens/秒）

**采集点**：基于滑动窗口计算

```python
# collector.py: request_output_token()
def request_output_token(self, uid: int) -> None:
    ...
    self._throughput_window.append((req.last_token_time, 1))

# collector.py: _calculate_throughput()
def _calculate_throughput(self) -> float:
    now = time.monotonic()
    window_start = self._throughput_window[0][0]
    duration = now - window_start
    total_tokens = sum(count for _, count in self._throughput_window)
    return total_tokens / duration
```

**计算逻辑**：
- 每次生成 output token 时记录时间戳和数量
- 使用 `deque(maxlen=1000)` 作为滑动窗口
- 吞吐量 = 窗口内总token数 / 窗口时间跨度

---

### 2.3 延迟指标 (TTFT)

#### minisgl_ttft_avg / p50 / p90 / p99 (gauge)

**含义**：Time To First Token 的统计值

**采集流程**：

```python
# 1. 请求到达时记录 arrival_time
# collector.py: request_arrived()
self._active_requests[uid] = RequestMetrics(
    uid=uid,
    arrival_time=now,  # 记录到达时间
)

# 2. 生成第一个 token 时记录 first_token_time
# collector.py: request_first_token()
def request_first_token(self, uid: int) -> None:
    if uid in self._active_requests:
        req = self._active_requests[uid]
        req.first_token_time = time.monotonic()

# 3. 请求完成时计算 TTFT
# RequestMetrics 属性
@property
def ttft(self) -> Optional[float]:
    if self.first_token_time is None:
        return None
    return self.first_token_time - self.arrival_time

# 4. 计算统计值
# collector.py: get_snapshot()
ttft_values = [req.ttft for req in self._completed_metrics if req.ttft is not None]
ttft_values.sort()
avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0
p50_ttft = self._percentile(ttft_values, 50)
p90_ttft = self._percentile(ttft_values, 90)
p99_ttft = self._percentile(ttft_values, 99)
```

**TTFT 定义**：从请求提交到第一个输出 token 生成的时间间隔

---

### 2.4 KV Cache 指标

#### minisgl_num_used_tokens (gauge)

**含义**：KV cache 中已使用的 token 数量

**采集点**：通过回调函数实时获取

```python
# scheduler.py: __init__()
self.metrics_collector.set_kv_cache_info(
    get_used_tokens_fn=lambda: (self.cache_manager.num_pages - len(self.cache_manager.free_slots)) * self.cache_manager.page_size,
    get_max_tokens_fn=lambda: self.cache_manager.num_pages * self.cache_manager.page_size,
)

# collector.py: get_snapshot()
if self._get_used_tokens_fn and self._get_max_tokens_fn:
    num_used_tokens = self._get_used_tokens_fn()
    max_total_num_tokens = self._get_max_tokens_fn()
    token_usage = num_used_tokens / max_total_num_tokens
```

**计算逻辑**：
- KV cache 以 page 为单位管理
- 已使用 tokens = (总页数 - 空闲页数) × 每页大小

---

#### minisgl_max_total_num_tokens (gauge)

**含义**：KV cache 池总容量（tokens）

**计算逻辑**：`num_pages * page_size`

---

#### minisgl_token_usage (gauge)

**含义**：KV cache 使用率

**计算逻辑**：`num_used_tokens / max_total_num_tokens`

---

### 2.5 Cache 命中指标

#### minisgl_cache_hit_rate (gauge)

**含义**：前缀缓存命中率

**采集点**：`PrefillAdder._try_allocate_one()`

```python
# prefill.py
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len  # 命中的前缀长度

    # 记录 cache hit/miss
    if self.metrics_collector is not None:
        if cached_len > 0:
            self.metrics_collector.record_cache_hit(hit=True)
        else:
            self.metrics_collector.record_cache_hit(hit=False)

# collector.py
def record_cache_hit(self, hit: bool) -> None:
    if hit:
        self._cache_hits += 1
    else:
        self._cache_misses += 1

@property
def cache_hit_rate(self) -> float:
    total = self._cache_hits + self._cache_misses
    if total == 0:
        return 0.0
    return self._cache_hits / total
```

---

#### minisgl_cache_hits / minisgl_cache_misses (counter)

**含义**：前缀缓存命中/未命中次数

**采集点**：同上

---
---

## 4. 关键设计决策

### 4.1 为什么使用 ZMQ 传输指标？

- Scheduler 运行在独立进程，需要跨进程通信
- ZMQ 是 Mini-SGLang 已有的 IPC 机制
- **初始化时发送一次**：确保 KV cache 总容量等指标在 server 启动后即可正确获取
- **周期性推送**（每 10 batch）：避免频繁通信开销，同时保持指标更新

### 4.2 什么是"batch"？

在 Mini-SGLang 的调度器中，**batch 指的是每轮 forward 计算的处理单元**：

- **Prefill batch**：调度器从 pending queue 中调度请求进行 prefill（提示词处理）
- **Decode batch**：调度器调度正在 decode 阶段的请求生成下一个 token
- 调度器采用"Prefill 优先"策略，优先处理 prefill 请求，没有 prefill 请求时才处理 decode

因此"每 10 个 batch"指的是**每 10 轮 forward 计算**（可能是 prefill 或 decode），而不是特指 prefill 或 decode。

```python
# scheduler.py: _process_last_data()
# 每处理完一个 batch 后调用
self._metrics_counter += 1
if self._metrics_counter >= 10:  # 每 10 个 batch 发送一次 metrics
    self._send_metrics()
    self._metrics_counter = 0
```

### 4.3 为什么指标计算在 Collector 端完成？

- 减少数据传输量（传输统计值而非原始数据）
- 保持 Scheduler 和 API Server 解耦
- API Server 只负责暴露，不负责计算

### 4.4 滑动窗口 vs 固定时间窗口

- 使用固定大小的 `deque(maxlen=1000)` 作为滑动窗口
- 避免维护定时器，简化实现
- 自适应吞吐量计算

---

## 5. 相关文件

| 文件 | 说明 |
|------|------|
| `python/minisgl/metrics/collector.py` | 核心指标采集逻辑 |
| `python/minisgl/message/backend.py` | `MetricsReportMsg` 定义 |
| `python/minisgl/message/frontend.py` | 前端消息反序列化 |
| `python/minisgl/message/tokenizer.py` | Tokenizer 消息透传 |
| `python/minisgl/scheduler/scheduler.py` | Scheduler 指标采集点 |
| `python/minisgl/scheduler/prefill.py` | Cache hit 记录 |
| `python/minisgl/server/api_server.py` | `/metrics` endpoint |
| `benchmark/online/bench_simple.py` | 基准测试脚本 |
| `benchmark/online/plot_metrics.py` | 指标可视化 |
