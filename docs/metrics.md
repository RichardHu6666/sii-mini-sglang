# Metrics Endpoint Documentation

## Overview

Mini-SGLang now exposes a `/metrics` endpoint that provides Prometheus-format metrics for monitoring the inference server.

## Endpoint

**GET /metrics**

Returns metrics in Prometheus exposition format.

## Metrics Exposed

### Request Counts

| Metric | Type | Description |
|--------|------|-------------|
| `minisgl_running_requests` | gauge | Number of requests currently being processed |
| `minisgl_queued_requests` | gauge | Number of requests waiting in the queue |
| `minisgl_completed_requests` | counter | Total number of completed requests |

### Token Usage

| Metric | Type | Description |
|--------|------|-------------|
| `minisgl_total_input_tokens` | counter | Total number of input tokens processed |
| `minisgl_total_output_tokens` | counter | Total number of output tokens generated |
| `minisgl_output_throughput` | gauge | Current output throughput (tokens/second) |

### Latency Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `minisgl_ttft_avg` | gauge | Average Time To First Token (seconds) |
| `minisgl_ttft_p50` | gauge | P50 Time To First Token (seconds) |
| `minisgl_ttft_p90` | gauge | P90 Time To First Token (seconds) |
| `minisgl_ttft_p99` | gauge | P99 Time To First Token (seconds) |

## Example Usage

### Curl

```bash
curl http://localhost:8000/metrics
```

### Example Output

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
```

### Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'minisgl'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

You can create a Grafana dashboard using the following queries:

- **Running Requests**: `minisgl_running_requests`
- **Queued Requests**: `minisgl_queued_requests`
- **Throughput**: `minisgl_output_throughput`
- **TTFT P99**: `minisgl_ttft_p99`
- **Token Usage Rate**: `rate(minisgl_total_output_tokens[1m])`

## Implementation Details

### Metrics Collection Flow

1. **Request Arrival**: When a `UserMsg` is received, `request_arrived()` is called
2. **First Token**: When the first output token is generated, `request_first_token()` is called
3. **Output Tokens**: For each output token, `request_output_token()` is called
4. **Request Completion**: When a request finishes, `request_completed()` is called
5. **Metrics Reporting**: Every 10 batches, metrics are sent to the API server via ZMQ

### Architecture

```
User Request → API Server → Tokenizer → Scheduler (Rank 0)
                                          ↓
                                    MetricsCollector
                                          ↓
                              ZMQ (MetricsReportMsg)
                                          ↓
API Server ←────────────────────────────────┘
    ↓
/metrics endpoint
```

## Testing

Run the test suite:

```bash
PYTHONPATH=python:$PYTHONPATH python tests/metrics/test_collector.py
```
