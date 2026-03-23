"""
Metrics collection for Mini-SGLang inference server.

This module provides utilities to collect and expose metrics such as:
- Running requests count
- Queued requests count
- Completed requests count
- Token usage (input/output)
- Output throughput
- TTFT (Time To First Token)
- KV cache usage
- Cache hit rate
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from collections import deque

import torch


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    uid: int
    arrival_time: float  # When the request arrived
    first_token_time: Optional[float] = None  # When the first token was generated
    last_token_time: Optional[float] = None  # When the last token was generated
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    queue_time: float = 0.0  # Time spent waiting in queue

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token in seconds."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def total_latency(self) -> Optional[float]:
        """Total end-to-end latency in seconds."""
        if self.last_token_time is None:
            return None
        return self.last_token_time - self.arrival_time

    @property
    def is_complete(self) -> bool:
        return self.last_token_time is not None


@dataclass
class MetricsSnapshot:
    """A snapshot of current metrics."""
    timestamp: float
    running_requests: int
    queued_requests: int
    completed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    throughput_tokens_per_sec: float
    avg_ttft: float
    p50_ttft: float
    p90_ttft: float
    p99_ttft: float
    # KV cache metrics (SGLang compatible)
    num_used_tokens: int = 0
    max_total_num_tokens: int = 0
    token_usage: float = 0.0
    # Cache metrics (token-level)
    cache_hit_rate: float = 0.0
    cache_hit_tokens: int = 0
    cache_prefill_tokens: int = 0
    # Cache metrics (legacy, kept for compatibility)
    cache_hits: int = 0
    cache_misses: int = 0
    # Queue time metrics
    avg_queue_time: float = 0.0
    p50_queue_time: float = 0.0
    p99_queue_time: float = 0.0

    def to_prometheus(self) -> str:
        """Convert metrics to Prometheus exposition format."""
        lines = [
            # Request counts
            f"# HELP minisgl_running_requests Number of running requests",
            f"# TYPE minisgl_running_requests gauge",
            f"minisgl_running_requests {self.running_requests}",
            f"# HELP minisgl_queued_requests Number of queued requests",
            f"# TYPE minisgl_queued_requests gauge",
            f"minisgl_queued_requests {self.queued_requests}",
            f"# HELP minisgl_completed_requests Total number of completed requests",
            f"# TYPE minisgl_completed_requests counter",
            f"minisgl_completed_requests {self.completed_requests}",
            # Token counts
            f"# HELP minisgl_total_input_tokens Total number of input tokens processed",
            f"# TYPE minisgl_total_input_tokens counter",
            f"minisgl_total_input_tokens {self.total_input_tokens}",
            f"# HELP minisgl_total_output_tokens Total number of output tokens generated",
            f"# TYPE minisgl_total_output_tokens counter",
            f"minisgl_total_output_tokens {self.total_output_tokens}",
            # Throughput
            f"# HELP minisgl_output_throughput Output throughput in tokens per second",
            f"# TYPE minisgl_output_throughput gauge",
            f"minisgl_output_throughput {self.throughput_tokens_per_sec:.2f}",
            # TTFT metrics
            f"# HELP minisgl_ttft_avg Average time to first token in seconds",
            f"# TYPE minisgl_ttft_avg gauge",
            f"minisgl_ttft_avg {self.avg_ttft:.6f}",
            f"# HELP minisgl_ttft_p50 P50 time to first token in seconds",
            f"# TYPE minisgl_ttft_p50 gauge",
            f"minisgl_ttft_p50 {self.p50_ttft:.6f}",
            f"# HELP minisgl_ttft_p90 P90 time to first token in seconds",
            f"# TYPE minisgl_ttft_p90 gauge",
            f"minisgl_ttft_p90 {self.p90_ttft:.6f}",
            f"# HELP minisgl_ttft_p99 P99 time to first token in seconds",
            f"# TYPE minisgl_ttft_p99 gauge",
            f"minisgl_ttft_p99 {self.p99_ttft:.6f}",
            # KV cache metrics (SGLang compatible)
            f"# HELP minisgl_num_used_tokens Number of used tokens in KV cache",
            f"# TYPE minisgl_num_used_tokens gauge",
            f"minisgl_num_used_tokens {self.num_used_tokens}",
            f"# HELP minisgl_max_total_num_tokens Maximum total number of tokens in KV cache pool",
            f"# TYPE minisgl_max_total_num_tokens gauge",
            f"minisgl_max_total_num_tokens {self.max_total_num_tokens}",
            f"# HELP minisgl_token_usage The token usage ratio (used/total)",
            f"# TYPE minisgl_token_usage gauge",
            f"minisgl_token_usage {self.token_usage:.6f}",
            # Cache hit rate
            f"# HELP minisgl_cache_hit_rate The prefix cache hit rate",
            f"# TYPE minisgl_cache_hit_rate gauge",
            f"minisgl_cache_hit_rate {self.cache_hit_rate:.4f}",
            f"# HELP minisgl_cache_hits Total number of cache hits",
            f"# TYPE minisgl_cache_hits counter",
            f"minisgl_cache_hits {self.cache_hits}",
            f"# HELP minisgl_cache_misses Total number of cache misses",
            f"# TYPE minisgl_cache_misses counter",
            f"minisgl_cache_misses {self.cache_misses}",
            # Queue time metrics
            f"# HELP minisgl_queue_time_avg Average queue time in seconds",
            f"# TYPE minisgl_queue_time_avg gauge",
            f"minisgl_queue_time_avg {self.avg_queue_time:.6f}",
            f"# HELP minisgl_queue_time_p50 P50 queue time in seconds",
            f"# TYPE minisgl_queue_time_p50 gauge",
            f"minisgl_queue_time_p50 {self.p50_queue_time:.6f}",
            f"# HELP minisgl_queue_time_p99 P99 queue time in seconds",
            f"# TYPE minisgl_queue_time_p99 gauge",
            f"minisgl_queue_time_p99 {self.p99_queue_time:.6f}",
        ]
        return "\n".join(lines) + "\n"


class MetricsCollector:
    """
    Collects metrics for the inference server.

    This collector tracks:
    - Request counts (running, queued, completed)
    - Token usage (input/output)
    - Latency metrics (TTFT, total latency, queue time)
    - Throughput
    - KV cache usage
    - Prefix cache hit rate
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the metrics collector.

        Args:
            max_history: Maximum number of completed request metrics to keep in history.
        """
        self._active_requests: Dict[int, RequestMetrics] = {}
        self._completed_metrics: deque[RequestMetrics] = deque(maxlen=max_history)

        # Counters
        self._total_completed: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        # Throughput calculation
        self._throughput_window: deque[tuple[float, int]] = deque(maxlen=1000)
        self._throughput_start_time: float = time.monotonic()

        # Queued requests (tracked externally)
        self._queued_count: int = 0
        self._active_count: int = 0  # Actively processing requests (decode phase)

        # KV cache metrics
        self._num_used_tokens: int = 0
        self._max_total_num_tokens: int = 0
        self._get_used_tokens_fn: Optional[Callable[[], int]] = None
        self._get_max_tokens_fn: Optional[Callable[[], int]] = None

        # Cache hit rate metrics (token-level)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_hit_tokens: int = 0
        self._cache_prefill_tokens: int = 0

    def set_kv_cache_info(self, get_used_tokens_fn: Callable[[], int],
                          get_max_tokens_fn: Callable[[], int]) -> None:
        """Set functions to get KV cache info."""
        self._get_used_tokens_fn = get_used_tokens_fn
        self._get_max_tokens_fn = get_max_tokens_fn

    def record_cache_hit(self, hit: bool) -> None:
        """Record a cache hit or miss (legacy, kept for compatibility)."""
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def record_cache_tokens(self, hit_tokens: int, total_tokens: int) -> None:
        """Record cache hit at token level."""
        self._cache_hit_tokens += hit_tokens
        self._cache_prefill_tokens += total_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate based on tokens."""
        if self._cache_prefill_tokens == 0:
            return 0.0
        return self._cache_hit_tokens / self._cache_prefill_tokens

    def request_arrived(self, uid: int, num_input_tokens: int, arrival_time: Optional[float] = None) -> None:
        """Record that a new request has arrived."""
        now = arrival_time or time.monotonic()
        self._active_requests[uid] = RequestMetrics(
            uid=uid,
            arrival_time=now,
            num_input_tokens=num_input_tokens,
        )
        self._total_input_tokens += num_input_tokens

    def request_first_token(self, uid: int) -> None:
        """Record that the first token was generated for a request."""
        if uid in self._active_requests:
            req = self._active_requests[uid]
            req.first_token_time = time.monotonic()
            # Calculate queue time (time from arrival to first token)
            req.queue_time = req.first_token_time - req.arrival_time

    def request_output_token(self, uid: int) -> None:
        """Record that an output token was generated for a request."""
        if uid in self._active_requests:
            req = self._active_requests[uid]
            req.num_output_tokens += 1
            req.last_token_time = time.monotonic()

            # Track for throughput calculation
            self._throughput_window.append((req.last_token_time, 1))

    def request_completed(self, uid: int) -> None:
        """Record that a request has completed."""
        if uid in self._active_requests:
            metrics = self._active_requests.pop(uid)
            self._completed_metrics.append(metrics)
            self._total_completed += 1
            self._total_output_tokens += metrics.num_output_tokens

    def request_aborted(self, uid: int) -> None:
        """Record that a request was aborted."""
        if uid in self._active_requests:
            del self._active_requests[uid]

    def set_queued_count(self, count: int) -> None:
        """Set the current number of queued requests."""
        self._queued_count = count

    def set_active_count(self, count: int) -> None:
        """Set the current number of actively processing requests (decode phase)."""
        self._active_count = count

    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of current metrics."""
        now = time.monotonic()

        # Calculate throughput
        throughput = self._calculate_throughput()

        # Calculate TTFT statistics
        ttft_values = [
            req.ttft for req in self._completed_metrics
            if req.ttft is not None
        ]
        ttft_values.sort()

        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0
        p50_ttft = self._percentile(ttft_values, 50)
        p90_ttft = self._percentile(ttft_values, 90)
        p99_ttft = self._percentile(ttft_values, 99)

        # Calculate queue time statistics
        queue_times = [
            req.queue_time for req in self._completed_metrics
            if req.queue_time > 0
        ]
        queue_times.sort()
        avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0.0
        p50_queue_time = self._percentile(queue_times, 50)
        p99_queue_time = self._percentile(queue_times, 99)

        # Get KV cache info
        num_used_tokens = 0
        max_total_num_tokens = 0
        token_usage = 0.0
        if self._get_used_tokens_fn and self._get_max_tokens_fn:
            try:
                num_used_tokens = self._get_used_tokens_fn()
                max_total_num_tokens = self._get_max_tokens_fn()
                if max_total_num_tokens > 0:
                    token_usage = num_used_tokens / max_total_num_tokens
            except Exception:
                pass

        # running_requests: requests actually being processed (decode phase)
        # queued_requests: requests waiting in pending list
        return MetricsSnapshot(
            timestamp=now,
            running_requests=self._active_count,
            queued_requests=self._queued_count,
            completed_requests=self._total_completed,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            throughput_tokens_per_sec=throughput,
            avg_ttft=avg_ttft,
            p50_ttft=p50_ttft,
            p90_ttft=p90_ttft,
            p99_ttft=p99_ttft,
            num_used_tokens=num_used_tokens,
            max_total_num_tokens=max_total_num_tokens,
            token_usage=token_usage,
            cache_hit_rate=self.cache_hit_rate,
            cache_hit_tokens=self._cache_hit_tokens,
            cache_prefill_tokens=self._cache_prefill_tokens,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            avg_queue_time=avg_queue_time,
            p50_queue_time=p50_queue_time,
            p99_queue_time=p99_queue_time,
        )

    def _calculate_throughput(self) -> float:
        """Calculate output throughput over the tracking window."""
        if not self._throughput_window:
            return 0.0

        now = time.monotonic()
        window_start = self._throughput_window[0][0]
        duration = now - window_start

        if duration <= 0:
            return 0.0

        total_tokens = sum(count for _, count in self._throughput_window)
        return total_tokens / duration

    def _percentile(self, sorted_values: list, percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_collector() -> None:
    """Reset the global metrics collector."""
    global _global_collector
    _global_collector = None
