"""Test script for metrics endpoint."""

import sys
import time

sys.path.insert(0, '/inspire/hdd/project/mianxiangdayuyanmoxing/261130310/sii-mini-sglang/python')

from minisgl.metrics import MetricsCollector, get_collector


def test_metrics_collector():
    """Test the MetricsCollector class."""
    collector = MetricsCollector()

    # Simulate request arrivals
    collector.request_arrived(uid=1, num_input_tokens=100)
    collector.request_arrived(uid=2, num_input_tokens=150)
    collector.request_arrived(uid=3, num_input_tokens=200)

    time.sleep(0.01)

    # Simulate first token
    collector.request_first_token(uid=1)
    collector.request_first_token(uid=2)

    time.sleep(0.01)

    # Simulate output tokens
    for _ in range(10):
        collector.request_output_token(uid=1)
        collector.request_output_token(uid=2)
        time.sleep(0.001)

    # Complete requests
    collector.request_completed(uid=1)
    collector.request_completed(uid=2)

    # Update queued count
    collector.set_queued_count(1)

    # Get snapshot
    snapshot = collector.get_snapshot()

    print("Metrics Snapshot:")
    print(f"  Running requests: {snapshot.running_requests}")
    print(f"  Queued requests: {snapshot.queued_requests}")
    print(f"  Completed requests: {snapshot.completed_requests}")
    print(f"  Total input tokens: {snapshot.total_input_tokens}")
    print(f"  Total output tokens: {snapshot.total_output_tokens}")
    print(f"  Throughput: {snapshot.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"  Avg TTFT: {snapshot.avg_ttft:.6f} seconds")
    print(f"  P50 TTFT: {snapshot.p50_ttft:.6f} seconds")
    print(f"  P90 TTFT: {snapshot.p90_ttft:.6f} seconds")
    print(f"  P99 TTFT: {snapshot.p99_ttft:.6f} seconds")

    print("\nPrometheus format:")
    print(snapshot.to_prometheus())

    # Verify expected values
    assert snapshot.running_requests == 1, f"Expected 1 running request, got {snapshot.running_requests}"
    assert snapshot.queued_requests == 1, f"Expected 1 queued request, got {snapshot.queued_requests}"
    assert snapshot.completed_requests == 2, f"Expected 2 completed requests, got {snapshot.completed_requests}"
    assert snapshot.total_input_tokens == 450, f"Expected 450 input tokens, got {snapshot.total_input_tokens}"
    assert snapshot.total_output_tokens == 20, f"Expected 20 output tokens, got {snapshot.total_output_tokens}"

    print("\nAll tests passed!")


def test_prometheus_format():
    """Test the Prometheus format output."""
    from minisgl.message import MetricsReportMsg

    msg = MetricsReportMsg(
        running_requests=5,
        queued_requests=10,
        completed_requests=100,
        total_input_tokens=50000,
        total_output_tokens=25000,
        throughput_tokens_per_sec=1234.56,
        avg_ttft=0.05,
        p50_ttft=0.045,
        p90_ttft=0.08,
        p99_ttft=0.12,
    )

    print("MetricsReportMsg Prometheus format:")
    from minisgl.server.api_server import MetricsState

    state = MetricsState()
    state.update_from_msg(msg)
    print(state.to_prometheus())


if __name__ == "__main__":
    test_metrics_collector()
    print("\n" + "=" * 50 + "\n")
    test_prometheus_format()
