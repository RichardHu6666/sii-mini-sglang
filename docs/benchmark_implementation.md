# Mini-SGLang Benchmark 实现逻辑文档

## 1. 整体架构

Mini-SGLang 的 benchmark 系统位于 `benchmark/online/` 目录，主要包含两个核心组件：

| 文件 | 职责 |
|------|------|
| `bench_simple.py` | 在线 benchmark 主脚本，执行并发请求测试并收集指标 |
| `plot_metrics.py` | 可视化脚本，将采样数据绘制成时间序列图表 |

### 1.1 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                    bench_simple.py                               │
│                                                                  │
│  ┌────────────────┐     ┌──────────────────┐                   │
│  │  生成测试数据   │────▶│  发送并发请求     │                   │
│  │  (tokenizer)   │     │  (OpenAI Client) │                   │
│  └────────────────┘     └──────────────────┘                   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              收集响应时间戳 (tics)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────┐     ┌──────────────────┐                   │
│  │  计算统计指标   │◀────│  MetricsSampler  │                   │
│  │  (TPOT/TTFT)   │     │  (采样/metrics)   │                   │
│  └────────────────┘     └──────────────────┘                   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              保存结果到 JSON                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    plot_metrics.py                               │
│                                                                  │
│  ┌────────────────┐     ┌──────────────────┐                   │
│  │  读取 JSON 数据   │────▶│  绘制时间序列图   │                   │
│  └────────────────┘     └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Benchmark 流程详解

### 2.1 启动流程

```python
# 1. 解析命令行参数
args = parse_args()
# --concurrency: 并发请求数 (默认 64)
# --input-len: 输入序列长度 (默认 1024)
# --output-len: 输出序列长度 (默认 256)
# --metrics-interval: /metrics 采样间隔 (默认 0.2 秒)

# 2. 创建输出目录
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# 3. 初始化 OpenAI 异步客户端
async with OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="") as client:
    # 4. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # 5. 测试连接
    test_result = await benchmark_one(client, test_msg, 2, MODEL)

    # 6. 生成测试消息
    messages = [generate_prompt(tokenizer, input_len) for _ in range(concurrency)]
    output_lengths = [random.randint(...) for _ in range(concurrency)]

    # 7. 启动 metrics 采样
    metrics_sampler = MetricsSampler(...)
    await metrics_sampler.start()

    # 8. 执行 benchmark
    results = await benchmark_one_batch(client, messages, output_lengths, MODEL)

    # 9. 处理结果并保存
    metrics_data = calculate_metrics(results)
    save_to_json(metrics_data)

    # 10. 停止 metrics 采样并保存
    await metrics_sampler.stop()
    metrics_sampler.save(...)
```

---

### 2.2 请求执行与时间戳收集

每个请求的执行过程由 `benchmark_one()` 函数处理：

```python
async def benchmark_one(client, prompt, output_length, model, ...) -> RawResult:
    # 1. 创建 OpenAI stream 请求
    response = await client.chat.completions.create(
        model=model,
        stream=True,  # 流式接收 token
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_length,
        temperature=0.0,
        extra_body={"ignore_eos": True, "top_k": 1},  # 强制生成指定数量 token
    )

    # 2. 记录时间戳
    tics = [time.perf_counter()]  # tics[0]: 请求开始时间
    async for _ in response:
        tics.append(time.perf_counter())  # 每收到一个 token 记录一次
        # tics[1]: 第一个 token 时间 (用于 TTFT)
        # tics[2]: 第二个 token 时间
        # ...

    # 3. 返回原始结果
    return RawResult(
        input_len=input_length,
        output_len=output_length,
        message=prompt,
        tics=tics,  # 时间戳列表
    )
```

**时间戳含义**：
- `tics[0]`：请求发送前的时刻
- `tics[1]`：收到第一个 token 的时刻
- `tics[2]`：收到第二个 token 的时刻
- ...
- `tics[n]`：收到第 n 个 token 的时刻

---

### 2.3 并发执行

`benchmark_one_batch()` 函数并发执行所有请求：

```python
async def benchmark_one_batch(client, prompts, output_lengths, model, ...) -> List[RawResult]:
    # 1. 为每个请求创建任务
    tasks = [
        benchmark_one(client, prompt, output_length, model, pbar, ...)
        for prompt, output_length in zip(prompts, output_lengths)
    ]

    # 2. 并发执行所有任务
    return await asyncio.gather(*tasks)
```

所有请求几乎同时启动，模拟真实的高并发场景。

---

### 2.4 指标计算

`calculate_metrics()` 函数从原始时间戳计算各项指标：

```python
def calculate_metrics(results: List[RawResult]) -> Dict[str, Any]:
    ttft_values = []
    tpot_values = []
    e2e_values = []
    total_tokens = 0

    for r in results:
        tics = r.tics

        # TTFT: Time To First Token
        ttft = tics[1] - tics[0]
        ttft_values.append(ttft)

        # TPOT: Time Per Output Token (不包括第一个 token)
        if len(tics) > 2:
            for i in range(1, len(tics) - 1):
                tpot_values.append(tics[i + 1] - tics[i])

        # E2E: End-to-End 延迟
        e2e = tics[-1] - tics[0]
        e2e_values.append(e2e)

        total_tokens += len(tics) - 1  # output token 数量

    # 计算统计数据
    def stats(values):
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p99": percentile(values, 99),
        }

    # 计算整体吞吐率
    min_start = min(r.tics[0] for r in results)
    max_end = max(r.tics[-1] for r in results)
    duration = max_end - min_start

    return {
        "num_requests": len(results),
        "total_tokens": total_tokens,
        "duration_seconds": duration,
        "request_throughput": len(results) / duration,
        "token_throughput": total_tokens / duration,
        "ttft_seconds": stats(ttft_values),
        "tpot_seconds": stats(tpot_values),
        "e2e_latency_seconds": stats(e2e_values),
    }
```

**指标定义**：

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| TTFT | 首 token 延迟 | `tics[1] - tics[0]` |
| TPOT | 每个 output token 的间隔 | `tics[i+1] - tics[i]` |
| E2E | 端到端延迟 | `tics[-1] - tics[0]` |
| Request Throughput | 请求吞吐率 | `请求数 / 总时间` |
| Token Throughput | token 吞吐率 | `总 token 数 / 总时间` |

---

## 3. Metrics 采样

### 3.1 采样原理

`MetricsSampler` 类定期从服务器的 `/metrics` endpoint 采集 Prometheus 格式指标：

```python
class MetricsSampler:
    def __init__(self, host, port, interval, output_dir):
        self.url = f"http://{host}:{port}/metrics"
        self.interval = interval  # 采样间隔 (秒)
        self.samples: List[Dict[str, Any]] = []

    async def _sample_loop(self):
        async with aiohttp.ClientSession() as session:
            while self.running:
                sample = await self._sample_once(session)
                if sample:
                    sample["timestamp"] = time.time() - self.start_time
                    self.samples.append(sample)
                await asyncio.sleep(self.interval)
```

### 3.2 采样的指标

每次采样从 `/metrics` endpoint 获取以下指标：

| 指标类别 | 具体指标 |
|----------|----------|
| 请求计数 | `minisgl_running_requests`, `minisgl_queued_requests`, `minisgl_completed_requests` |
| Token 计数 | `minisgl_total_input_tokens`, `minisgl_total_output_tokens` |
| 吞吐率 | `minisgl_output_throughput` |
| TTFT | `minisgl_ttft_avg`, `minisgl_ttft_p50`, `minisgl_ttft_p90`, `minisgl_ttft_p99` |
| KV Cache | `minisgl_num_used_tokens`, `minisgl_max_total_num_tokens`, `minisgl_token_usage` |
| Cache 命中 | `minisgl_cache_hit_rate`, `minisgl_cache_hits`, `minisgl_cache_misses` |
| 排队时间 | `minisgl_queue_time_avg`, `minisgl_queue_time_p50`, `minisgl_queue_time_p99` |

### 3.3 可视化

采样数据保存为 JSON 格式，并通过 `plot_metrics.py` 绘制时间序列图：

```python
# 读取 JSON 数据
with open(input_file, "r") as f:
    samples = json.load(f)

# 按类别分组绘制
METRIC_GROUPS = {
    "request_counts": [...],      # 请求数量
    "token_usage": [...],         # Token 使用量
    "kv_cache_tokens": [...],     # KV Cache 容量
    "throughput": [...],          # 吞吐率
}

# 每个类别一个子图
fig, axes = plt.subplots(num_groups, 1, figsize=(14, 5 * num_groups))
```

---

## 4. 关键设计决策

### 4.1 为什么使用 `time.perf_counter()`？

- `perf_counter()` 提供最高可用分辨率
- 不受系统时钟调整影响
- 适合测量短时间间隔

### 4.2 为什么 TPOT 不包括第一个 token？

- 第一个 token 的延迟主要受 TTFT 影响（包括排队、调度等）
- TPOT 应该反映 decode 阶段的纯生成延迟
- 因此 TPOT 从第二个 token 开始计算

### 4.3 为什么使用 `asyncio.gather()`？

- 模拟真实并发场景
- 所有请求几乎同时启动
- 可以观察系统在高负载下的行为

### 4.4 为什么强制 `ignore_eos=True`？

- 确保每个请求生成指定数量的 token
- 避免某些请求提前结束影响统计
- 使 benchmark 结果更可重复

---

## 5. 使用示例

### 5.1 运行 Benchmark

```bash
python benchmark/online/bench_simple.py \
    --host 127.0.0.1 \
    --port 1919 \
    --concurrency 64 \
    --input-len 1024 \
    --output-len 256 \
    --output-dir ./benchmark_results
```

### 5.2 生成可视化图表

```bash
python benchmark/online/plot_metrics.py \
    benchmark_results/20260322_123456_metrics_samples.json
```

### 5.3 输出文件

| 文件 | 内容 |
|------|------|
| `{timestamp}_metrics.json` | 汇总指标（TTFT/TPOT/E2E统计） |
| `{timestamp}_metrics_samples.json` | 时间序列采样数据 |
| `{timestamp}_plot.png` | 可视化图表 |

---

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `benchmark/online/bench_simple.py` | Benchmark 主脚本 |
| `benchmark/online/plot_metrics.py` | 可视化脚本 |
| `python/minisgl/benchmark/client.py` | Benchmark 客户端工具 |
| `python/minisgl/server/api_server.py` | API Server (`/metrics` endpoint) |
| `python/minisgl/metrics/collector.py` | 指标采集逻辑 |
