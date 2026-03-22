from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class MetricsReportMsg(BaseBackendMsg):
    """Message to report metrics from scheduler to API server."""
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
