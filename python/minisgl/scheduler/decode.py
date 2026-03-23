from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
    _decode_enqueued_at: dict[int, float] = field(default_factory=dict)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        now = time.perf_counter()
        merged = {req for req in self.running_reqs.union(reqs) if req.can_decode}
        for req in merged:
            self._decode_enqueued_at.setdefault(req.uid, now)
        for req_uid in list(self._decode_enqueued_at.keys()):
            if not any(req.uid == req_uid for req in merged):
                self._decode_enqueued_at.pop(req_uid, None)
        self.running_reqs = merged

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        self._decode_enqueued_at.pop(req.uid, None)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                self._decode_enqueued_at.pop(uid, None)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self, short_remain_threshold: int = 64, max_wait_ms: int = 30) -> Batch | None:
        if not self.runnable:
            return None

        now = time.perf_counter()
        wait_s = max(max_wait_ms, 0) / 1000.0

        def _key(req: Req) -> tuple[int, int, int]:
            enq_ts = self._decode_enqueued_at.get(req.uid, now)
            is_urgent = int((now - enq_ts) >= wait_s)
            is_short = int(req.remain_len <= short_remain_threshold)
            return (-is_urgent, -is_short, int(enq_ts * 1_000_000))

        ordered = sorted(self.running_reqs, key=_key)
        return Batch(reqs=ordered, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
