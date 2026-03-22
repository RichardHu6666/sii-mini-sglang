# Online Benchmark Script with Metrics

## Overview

The enhanced `bench_simple.py` script provides comprehensive benchmarking capabilities with:
- Command-line configuration for concurrency and sequence lengths
- JSON export of TTFT, throughput, and latency metrics
- Time-series visualization of server `/metrics` endpoint

## Quick Start

### Basic Usage

```bash
cd /inspire/hdd/project/mianxiangdayuyanmoxing/261130310/sii-mini-sglang
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

# Run with default settings (64 concurrency, 1024 input, 256 output)
python benchmark/online/bench_simple.py

# Custom concurrency and lengths
python benchmark/online/bench_simple.py \
    --concurrency 32 \
    --input-len 512 \
    --output-len 128
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | 127.0.0.1 | Server host |
| `--port` | 1919 | Server port |
| `--concurrency` | 64 | Number of concurrent requests |
| `--input-len` | 1024 | Input sequence length (tokens) |
| `--output-len` | 256 | Output sequence length (tokens) |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | ./benchmark_results | Output directory for results |
| `--metrics-interval` | 1.0 | Interval (seconds) to sample /metrics |
| `--skip-metrics` | - | Skip sampling /metrics endpoint |
| `--model-path` | auto | Model path for tokenizer |

## Output Files

The script generates the following files in `--output-dir`:

### 1. Metrics JSON (`{timestamp}_metrics.json`)

Contains aggregated benchmark results:

```json
{
  "config": {
    "concurrency": 64,
    "input_length": 1024,
    "output_length": 256,
    "seed": 42,
    "model": "Qwen/Qwen3-0.6B",
    "timestamp": "20260322_143022"
  },
  "results": {
    "num_requests": 64,
    "total_tokens": 18432,
    "duration_seconds": 15.234,
    "request_throughput": 4.201,
    "token_throughput": 1209.87,
    "ttft_seconds": {
      "avg": 0.0523,
      "min": 0.0312,
      "max": 0.1245,
      "p50": 0.0498,
      "p90": 0.0876,
      "p99": 0.1198
    },
    "tpot_seconds": {...},
    "e2e_latency_seconds": {...}
  }
}
```

### 2. Metrics Samples JSON (`{timestamp}_metrics_samples.json`)

Time-series samples from `/metrics` endpoint:

```json
[
  {
    "timestamp": 0.0,
    "minisgl_running_requests": 0,
    "minisgl_queued_requests": 0,
    "minisgl_output_throughput": 0.0,
    "minisgl_ttft_avg": 0.0
  },
  {
    "timestamp": 1.0,
    "minisgl_running_requests": 32,
    "minisgl_queued_requests": 32,
    "minisgl_output_throughput": 856.23,
    "minisgl_ttft_avg": 0.0534
  },
  ...
]
```

### 3. Time-Series Plot (`{timestamp}_metrics_samples_timeseries.png`)

Visualization of metrics over time:
- Running requests
- Queued requests
- Output throughput
- TTFT (avg, p50, p90, p99)

## Examples

### Example 1: Light Load Test

```bash
python benchmark/online/bench_simple.py \
    --concurrency 16 \
    --input-len 256 \
    --output-len 64 \
    --output-dir ./results/light_load
```

### Example 2: Heavy Load Test

```bash
python benchmark/online/bench_simple.py \
    --concurrency 128 \
    --input-len 2048 \
    --output-len 512 \
    --output-dir ./results/heavy_load
```

### Example 3: Without Metrics Sampling

```bash
python benchmark/online/bench_simple.py \
    --concurrency 64 \
    --skip-metrics
```

### Example 4: Custom Metrics Sampling Interval

```bash
python benchmark/online/bench_simple.py \
    --concurrency 64 \
    --metrics-interval 0.5  # Sample twice per second
```

## Plotting Script

Use `plot_metrics.py` to regenerate plots from saved JSON data:

```bash
# Generate plots from saved metrics samples
python benchmark/online/plot_metrics.py ./benchmark_results/20260322_143022_metrics_samples.json

# Custom output directory
python benchmark/online/plot_metrics.py \
    ./results/metrics_samples.json \
    --output-dir ./plots \
    --title "Qwen3-0.6B Benchmark 2026-03-22"
```

The plotting script generates:
- Combined time-series plot with all metrics
- Individual plots for each metric with statistics

## Metrics Explained

### Summary Table

| Metric Name | Type | Description |
|-------------|------|-------------|
| `minisgl_running_requests` | gauge | Number of running requests |
| `minisgl_queued_requests` | gauge | Number of queued requests |
| `minisgl_completed_requests` | counter | Total completed requests |
| `minisgl_total_input_tokens` | counter | Total input tokens processed |
| `minisgl_total_output_tokens` | counter | Total output tokens generated |
| `minisgl_output_throughput` | gauge | Output throughput (tokens/sec) |
| `minisgl_ttft_avg` | gauge | Average TTFT (seconds) |
| `minisgl_ttft_p50` | gauge | P50 TTFT (seconds) |
| `minisgl_ttft_p90` | gauge | P90 TTFT (seconds) |
| `minisgl_ttft_p99` | gauge | P99 TTFT (seconds) |
| `minisgl_num_used_tokens` | gauge | KV cache used tokens |
| `minisgl_max_total_num_tokens` | gauge | KV cache max capacity |
| `minisgl_token_usage` | gauge | KV cache usage ratio |
| `minisgl_cache_hit_rate` | gauge | Prefix cache hit rate |
| `minisgl_cache_hits` | counter | Total cache hits |
| `minisgl_cache_misses` | counter | Total cache misses |
| `minisgl_queue_time_avg` | gauge | Average queue time (seconds) |
| `minisgl_queue_time_p50` | gauge | P50 queue time (seconds) |
| `minisgl_queue_time_p99` | gauge | P99 queue time (seconds) |

### Request Count Metrics

#### minisgl_running_requests (gauge)
- **Definition**: Number of requests currently being processed
- **Type**: Gauge (can go up or down)
- **Good value**: Depends on concurrency and model capacity

#### minisgl_queued_requests (gauge)
- **Definition**: Number of requests waiting in the queue
- **Type**: Gauge
- **Good value**: Close to 0 (indicates no backlog)

#### minisgl_completed_requests (counter)
- **Definition**: Total number of completed requests since server start
- **Type**: Counter (only increases)

### Token Metrics

#### minisgl_total_input_tokens (counter)
- **Definition**: Total number of input tokens processed
- **Type**: Counter

#### minisgl_total_output_tokens (counter)
- **Definition**: Total number of output tokens generated
- **Type**: Counter

#### minisgl_output_throughput (gauge)
- **Definition**: Output tokens generated per second
- **Unit**: tokens/second
- **Good value**: Higher is better, depends on model and hardware

### Latency Metrics

#### TTFT (Time To First Token)
- **Definition**: Time from request submission to first output token
- **Unit**: Seconds
- **Good value**: < 100ms for interactive applications

Metrics available:
- `minisgl_ttft_avg`: Average TTFT
- `minisgl_ttft_p50`: Median TTFT (50th percentile)
- `minisgl_ttft_p90`: 90th percentile TTFT
- `minisgl_ttft_p99`: 99th percentile TTFT

#### TPOT (Time Per Output Token)
- **Definition**: Time between consecutive output tokens
- **Unit**: Seconds
- **Good value**: < 50ms for smooth streaming

#### E2E Latency (End-to-End)
- **Definition**: Total time from request to completion
- **Unit**: Seconds
- **Depends on**: Input length, output length, model size

#### Queue Time
- **Definition**: Time a request spends waiting in queue before processing
- **Unit**: Seconds
- **Good value**: Close to 0

Metrics available:
- `minisgl_queue_time_avg`: Average queue time
- `minisgl_queue_time_p50`: Median queue time
- `minisgl_queue_time_p99`: 99th percentile queue time

### KV Cache Metrics (SGLang Compatible)

#### minisgl_num_used_tokens (gauge)
- **Definition**: Number of tokens currently stored in KV cache
- **Type**: Gauge

#### minisgl_max_total_num_tokens (gauge)
- **Definition**: Maximum total tokens the KV cache pool can hold
- **Type**: Gauge

#### minisgl_token_usage (gauge)
- **Definition**: Ratio of used tokens to total capacity (used/total)
- **Range**: 0.0 to 1.0
- **Good value**: Lower is better (more available cache)

### Cache Metrics

#### minisgl_cache_hit_rate (gauge)
- **Definition**: Ratio of cache hits to total cache lookups
- **Range**: 0.0 to 1.0
- **Good value**: Higher is better (indicates good prefix reuse)

#### minisgl_cache_hits (counter)
- **Definition**: Total number of successful prefix cache matches
- **Type**: Counter

#### minisgl_cache_misses (counter)
- **Definition**: Total number of cache lookups that did not match
- **Type**: Counter

## Server Requirements

The benchmark script requires:
1. A running Mini-SGLang server with `/metrics` endpoint enabled
2. Server accessible at the specified `--host` and `--port`
3. OpenAI-compatible API at `/v1/chat/completions`

## Dependencies

- `aiohttp`: For async HTTP requests to /metrics
- `matplotlib`: For generating time-series plots (optional)
- `transformers`: For tokenizer
- `openai`: Python client library

Install optional dependencies:
```bash
pip install aiohttp matplotlib
```

## Troubleshooting

### Connection Refused
```
Make sure the server is running on http://127.0.0.1:1919
```
- Check server is running: `curl http://127.0.0.1:1919/v1/models`
- Verify port with `--port` argument

### No Metrics Data
```
Warning: No metrics samples to save
```
- Ensure server has `/metrics` endpoint enabled
- Check network connectivity between benchmark and server
- Increase `--metrics-interval` if server is slow

### Plot Generation Failed
```
matplotlib not installed, skipping plot generation
```
- Install matplotlib: `pip install matplotlib`
- Or use `plot_metrics.py` later to generate plots
