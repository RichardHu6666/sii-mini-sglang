from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .backend import MetricsReportMsg
from .utils import deserialize_type, serialize_type

# Register MetricsReportMsg for deserialization
_MSG_EXTRA_TYPES = {"MetricsReportMsg": MetricsReportMsg}


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        # Merge extra types for deserialization
        cls_map = {**globals(), **_MSG_EXTRA_TYPES}
        return deserialize_type(cls_map, json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
