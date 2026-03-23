from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO

# 加入EPinfo
_EP__INFO : DistributedInfo | None = None

# 初始化设置ep信息
def set_ep_info(rank: int, size: int) -> None:
    global _EP__INFO
    if _EP__INFO is not None:
        raise RuntimeError("EP info has been set")
    _EP__INFO = DistributedInfo(rank, size)
    
# 获取ep信息（已设置情况下）
def get_ep_info() -> DistributedInfo:
    if _EP__INFO is None:
        raise RuntimeError("EP info has not been set")
    return _EP__INFO

# 尝试获取ep信息
def try_get_ep_info() -> DistributedInfo | None:
    return _EP__INFO

# 导出列表

__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "get_tp_info",
    "try_get_tp_info",
    "set_ep_info",
    "get_ep_info",
    "try_get_ep_info",
]

