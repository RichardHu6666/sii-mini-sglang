"""Test script for bench_simple.py metrics functionality."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/inspire/hdd/project/mianxiangdayuyanmoxing/261130310/sii-mini-sglang/python')

from benchmark.online.bench_simple import MetricsSampler, calculate_metrics
from minisgl.benchmark.client import RawResult


def test_calculate_metrics():
    """Test the calculate_metrics function."""
    # Create mock results
    base_time = time.time()
    results = [
        RawResult(
            input_len=100,
            output_len=10,
            message="test",
            tics=[base_time + i * 0.1 for i in range(11)]
        ),
        RawResult(
            input_len=150,
            output_len=8,
            message="test2",
            tics=[base_time + 0.05 + i * 0.1 for i in range(9)]
        ),
    ]

    metrics = calculate_metrics(results)

    print("Calculated metrics:")
    print(json.dumps(metrics, indent=2))

    # Verify expected fields
    assert "num_requests" in metrics
    assert "total_tokens" in metrics
    assert "duration_seconds" in metrics
    assert "request_throughput" in metrics
    assert "token_throughput" in metrics
    assert "ttft_seconds" in metrics
    assert "tpot_seconds" in metrics
    assert "e2e_latency_seconds" in metrics

    assert metrics["num_requests"] == 2
    assert metrics["ttft_seconds"]["avg"] > 0

    print("\nAll metrics calculations passed!")
    return True


def test_metrics_sampler_parse():
    """Test the MetricsSampler Prometheus parsing."""
    sample_prometheus = """
# HELP minisgl_running_requests Number of running requests
# TYPE minisgl_running_requests gauge
minisgl_running_requests 5
# HELP minisgl_queued_requests Number of queued requests
# TYPE minisgl_queued_requests gauge
minisgl_queued_requests 10
# HELP minisgl_output_throughput Output throughput in tokens per second
# TYPE minisgl_output_throughput gauge
minisgl_output_throughput 1234.56
# HELP minisgl_ttft_avg Average time to first token in seconds
# TYPE minisgl_ttft_avg gauge
minisgl_ttft_avg 0.050000
"""

    sampler = MetricsSampler("localhost", 8000, 1.0, Path("./test_output"))
    parsed = sampler._parse_prometheus(sample_prometheus.strip())

    print("\nParsed Prometheus metrics:")
    print(json.dumps(parsed, indent=2))

    assert parsed["minisgl_running_requests"] == 5.0
    assert parsed["minisgl_queued_requests"] == 10.0
    assert parsed["minisgl_output_throughput"] == 1234.56
    assert parsed["minisgl_ttft_avg"] == 0.05

    print("\nPrometheus parsing passed!")
    return True


def main():
    print("=" * 60)
    print("Testing bench_simple.py metrics functionality")
    print("=" * 60)

    test_calculate_metrics()
    print()
    test_metrics_sampler_parse()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
