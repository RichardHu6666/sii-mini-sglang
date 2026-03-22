from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .backend import MetricsReportMsg
from .utils import deserialize_type, serialize_type

# Register MetricsReportMsg for deserialization
_MSG_EXTRA_TYPES = {"MetricsReportMsg": MetricsReportMsg}


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        # Merge extra types for deserialization
        cls_map = {**globals(), **_MSG_EXTRA_TYPES}
        return deserialize_type(cls_map, json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
