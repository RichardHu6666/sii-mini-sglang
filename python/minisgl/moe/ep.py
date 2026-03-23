import csv
import math
import os
import torch
import torch.distributed as dist
from minisgl.distributed import get_ep_info
from minisgl.distributed.impl import ep_all_to_all, get_ep_group
from minisgl.env import ENV
from minisgl.moe.base import BaseMoeBackend
from minisgl.moe.fused import fused_experts_impl, fused_topk
from minisgl.utils import init_logger


logger = init_logger(__name__)


class EPMoe(BaseMoeBackend):
    def __init__(self) -> None:
        self._profile_step = 0
        self._csv_initialized = False
        self._csv_path = os.getenv("MINISGL_EP_PROFILE_CSV", "")
        self._window = max(int(os.getenv("MINISGL_EP_PROFILE_WINDOW", "200")), 1)
        self._route_hist: list[float] = []
        self._dispatch_hist: list[float] = []
        self._compute_hist: list[float] = []
        self._combine_hist: list[float] = []
        self._reduce_hist: list[float] = []
        self._comm_hist: list[float] = []
        self._spike_dispatch_count = 0
        self._spike_combine_count = 0
        self._spike_comm_count = 0

    def _push_hist(self, hist: list[float], x: float) -> None:
        hist.append(float(x))
        if len(hist) > self._window:
            del hist[0]

    @staticmethod
    def _agg(hist: list[float]) -> tuple[float, float, float, float]:
        if not hist:
            return 0.0, 0.0, 0.0, 0.0
        sorted_hist = sorted(hist)
        n = len(sorted_hist)
        p95 = sorted_hist[min(int(n * 0.95), n - 1)]
        p99 = sorted_hist[min(int(n * 0.99), n - 1)]
        mean = sum(sorted_hist) / n
        return mean, p95, p99, sorted_hist[-1]

    def _init_csv(self) -> None:
        if self._csv_initialized or not self._csv_path:
            return
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "pairs",
                    "recv",
                    "route_ms",
                    "dispatch_ms",
                    "compute_ms",
                    "combine_ms",
                    "reduce_ms",
                    "comm_ms",
                    "route_mean",
                    "route_p95",
                    "route_p99",
                    "route_max",
                    "dispatch_mean",
                    "dispatch_p95",
                    "dispatch_p99",
                    "dispatch_max",
                    "compute_mean",
                    "compute_p95",
                    "compute_p99",
                    "compute_max",
                    "combine_mean",
                    "combine_p95",
                    "combine_p99",
                    "combine_max",
                    "comm_mean",
                    "comm_p95",
                    "comm_p99",
                    "comm_max",
                    "reduce_mean",
                    "reduce_p95",
                    "reduce_p99",
                    "reduce_max",
                    "expert_max",
                    "expert_mean",
                    "expert_cv",
                    "send_max",
                    "send_mean",
                    "recv_max",
                    "recv_mean",
                    "spike_comm_gt1ms",
                    "spike_dispatch_gt1ms",
                    "spike_combine_gt1ms",
                ]
            )
        self._csv_initialized = True

    def _append_csv(self, row: list[object]) -> None:
        if not self._csv_path:
            return
        self._init_csv()
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        do_profile = bool(ENV.EP_PROFILE) and not torch.cuda.is_current_stream_capturing()
        ev_route_end = ev_comm_dispatch_end = ev_compute_end = ev_comm_combine_end = ev_end = None
        if do_profile:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_route_end = torch.cuda.Event(enable_timing=True)
            ev_comm_dispatch_end = torch.cuda.Event(enable_timing=True)
            ev_compute_end = torch.cuda.Event(enable_timing=True)
            ev_comm_combine_end = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        ep_info = get_ep_info()
        ep_size = ep_info.size
        ep_rank = ep_info.rank
        num_tokens, hidden_size = hidden_states.shape
        num_local_experts = w1.shape[0]
        num_pairs = num_tokens * topk
        small_packet_threshold = int(ENV.EP_SMALL_PACKET_THRESHOLD.value)
        small_packet_enabled = bool(ENV.EP_SMALL_PACKET_ENABLE) or small_packet_threshold > 0
        use_small_packet_sync = (
            small_packet_enabled
            and small_packet_threshold > 0
            and num_pairs <= small_packet_threshold
            and bool(ENV.EP_SMALL_PACKET_SYNC)
        )
        # 选择topk expert
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        # 计算目标rank 和local expert id
        flat_ids = topk_ids.view(-1)
        dest_rank = flat_ids.to(torch.int64) // num_local_experts
        local_ids = (flat_ids % num_local_experts).to(torch.int32)
        # 创建token索引
        token_idx = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, topk)
            .reshape(-1)
        )
        # 排序token
        sort_idx = torch.argsort(dest_rank, stable=True)
        sorted_token_idx = token_idx[sort_idx]
        sorted_local_ids = local_ids[sort_idx]
        # 计算通信量
        send_hidden = hidden_states[sorted_token_idx].contiguous()
        send_counts = torch.bincount(dest_rank, minlength=ep_size)
        local_expert_load = torch.bincount(local_ids, minlength=num_local_experts).to(torch.float32)
        expert_mean = float(local_expert_load.mean().item()) if num_local_experts > 0 else 0.0
        expert_max = float(local_expert_load.max().item()) if num_local_experts > 0 else 0.0
        expert_std = (
            float(local_expert_load.std(unbiased=False).item()) if num_local_experts > 0 else 0.0
        )
        expert_cv = (expert_std / expert_mean) if expert_mean > 0 else 0.0
        if do_profile and ev_route_end is not None:
            ev_route_end.record()

        # Fast path: all routed experts are local, skip all communication.
        local_only = int(send_counts[ep_rank].item()) == num_pairs
        if local_only:
            unit_weights = torch.ones(
                num_pairs,
                1,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            local_out = fused_experts_impl(
                send_hidden,
                w1,
                w2,
                unit_weights,
                sorted_local_ids.unsqueeze(1),
                activation=activation,
                apply_router_weight_on_input=False,
            )
            if do_profile and ev_comm_dispatch_end is not None:
                ev_comm_dispatch_end.record()
            if do_profile and ev_compute_end is not None:
                ev_compute_end.record()
            combined = local_out
            if do_profile and ev_comm_combine_end is not None:
                ev_comm_combine_end.record()
            total_recv = num_pairs
        else:
            # Control plane: use all_gather for tiny count metadata to reduce jitter.
            # gathered[src_rank, dst_rank] = count sent from src_rank to dst_rank
            gathered_counts = torch.empty(
                (ep_size, ep_size),
                dtype=send_counts.dtype,
                device=send_counts.device,
            )
            work_counts = dist.all_gather_into_tensor(
                gathered_counts,
                send_counts,
                group=get_ep_group(),
                async_op=not use_small_packet_sync,
            )
            if work_counts is not None:
                work_counts.wait()
            recv_counts = gathered_counts[:, ep_rank].contiguous()

            # 交换hidden state
            send_splits = send_counts.tolist()
            recv_splits = recv_counts.tolist()
            total_recv = sum(recv_splits)

            recv_hidden = hidden_states.new_empty(total_recv, hidden_size)
            recv_ids = sorted_local_ids.new_empty(total_recv)
            work_hidden = ep_all_to_all(
                recv_hidden,
                send_hidden,
                recv_splits,
                send_splits,
                async_op=not use_small_packet_sync,
            )
            # 交换expert id
            work_ids = ep_all_to_all(
                recv_ids,
                sorted_local_ids,
                recv_splits,
                send_splits,
                async_op=not use_small_packet_sync,
            )
            if work_hidden is not None:
                work_hidden.wait()
            if work_ids is not None:
                work_ids.wait()
            if do_profile and ev_comm_dispatch_end is not None:
                ev_comm_dispatch_end.record()

            # 本地expert 计算
            if total_recv > 0:
                unit_weights = torch.ones(
                    total_recv,
                    1,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                local_out = fused_experts_impl(
                    recv_hidden,
                    w1,
                    w2,
                    unit_weights,
                    recv_ids.unsqueeze(1),
                    activation=activation,
                    apply_router_weight_on_input=False,
                )
            else:
                local_out = hidden_states.new_empty(0, hidden_size)
            if do_profile and ev_compute_end is not None:
                ev_compute_end.record()

            combined = hidden_states.new_empty(num_pairs, hidden_size)
            work_combined = ep_all_to_all(
                combined,
                local_out,
                send_splits,
                recv_splits,
                async_op=not use_small_packet_sync,
            )
            if work_combined is not None:
                work_combined.wait()
            if do_profile and ev_comm_combine_end is not None:
                ev_comm_combine_end.record()

        # 恢复token顺序 加权输出
        result = hidden_states.new_empty(num_pairs, hidden_size)
        result[sort_idx] = combined
        result = result.view(num_tokens, topk, hidden_size)
        weights = topk_weights.to(hidden_states.dtype).unsqueeze(-1)
        out = (result * weights).sum(dim=1)

        if do_profile and ev_end is not None and ev_route_end is not None and ev_comm_dispatch_end is not None and ev_compute_end is not None and ev_comm_combine_end is not None:
            ev_end.record()
            self._profile_step += 1
            interval = max(int(ENV.EP_PROFILE_INTERVAL.value), 1)
            if self._profile_step % interval == 0:
                torch.cuda.synchronize(hidden_states.device)
                route_ms = ev_start.elapsed_time(ev_route_end)
                comm_dispatch_ms = ev_route_end.elapsed_time(ev_comm_dispatch_end)
                compute_ms = ev_comm_dispatch_end.elapsed_time(ev_compute_end)
                comm_combine_ms = ev_compute_end.elapsed_time(ev_comm_combine_end)
                reduce_ms = ev_comm_combine_end.elapsed_time(ev_end)
                comm_total_ms = comm_dispatch_ms + comm_combine_ms

                self._push_hist(self._route_hist, route_ms)
                self._push_hist(self._dispatch_hist, comm_dispatch_ms)
                self._push_hist(self._compute_hist, compute_ms)
                self._push_hist(self._combine_hist, comm_combine_ms)
                self._push_hist(self._reduce_hist, reduce_ms)
                self._push_hist(self._comm_hist, comm_total_ms)

                if comm_total_ms > 1.0:
                    self._spike_comm_count += 1
                if comm_dispatch_ms > 1.0:
                    self._spike_dispatch_count += 1
                if comm_combine_ms > 1.0:
                    self._spike_combine_count += 1

                route_s = self._agg(self._route_hist)
                dispatch_s = self._agg(self._dispatch_hist)
                compute_s = self._agg(self._compute_hist)
                combine_s = self._agg(self._combine_hist)
                reduce_s = self._agg(self._reduce_hist)
                comm_s = self._agg(self._comm_hist)

                send_counts_f = send_counts.to(torch.float32)
                recv_counts_f = recv_counts.to(torch.float32) if not local_only else send_counts_f
                send_max = float(send_counts_f.max().item())
                send_mean = float(send_counts_f.mean().item())
                recv_max = float(recv_counts_f.max().item())
                recv_mean = float(recv_counts_f.mean().item())

                logger.info_rank0(
                    "EP profile step=%d pairs=%d recv=%d ms(route=%.3f comm=%.3f compute=%.3f reduce=%.3f, dispatch=%.3f, combine=%.3f)",
                    self._profile_step,
                    num_pairs,
                    total_recv,
                    route_ms,
                    comm_total_ms,
                    compute_ms,
                    reduce_ms,
                    comm_dispatch_ms,
                    comm_combine_ms,
                )
                logger.info_rank0(
                    "EP p0 window=%d step=%d stat(route m/p95/p99/max=%.3f/%.3f/%.3f/%.3f, dispatch=%.3f/%.3f/%.3f/%.3f, compute=%.3f/%.3f/%.3f/%.3f, combine=%.3f/%.3f/%.3f/%.3f, reduce=%.3f/%.3f/%.3f/%.3f, comm=%.3f/%.3f/%.3f/%.3f) load(expert max/mean/cv=%.1f/%.1f/%.3f) msg(send max/mean=%.1f/%.1f recv max/mean=%.1f/%.1f) spikes(>1ms comm=%d dispatch=%d combine=%d)",
                    self._window,
                    self._profile_step,
                    route_s[0],
                    route_s[1],
                    route_s[2],
                    route_s[3],
                    dispatch_s[0],
                    dispatch_s[1],
                    dispatch_s[2],
                    dispatch_s[3],
                    compute_s[0],
                    compute_s[1],
                    compute_s[2],
                    compute_s[3],
                    combine_s[0],
                    combine_s[1],
                    combine_s[2],
                    combine_s[3],
                    reduce_s[0],
                    reduce_s[1],
                    reduce_s[2],
                    reduce_s[3],
                    comm_s[0],
                    comm_s[1],
                    comm_s[2],
                    comm_s[3],
                    expert_max,
                    expert_mean,
                    expert_cv,
                    send_max,
                    send_mean,
                    recv_max,
                    recv_mean,
                    self._spike_comm_count,
                    self._spike_dispatch_count,
                    self._spike_combine_count,
                )
                self._append_csv(
                    [
                        self._profile_step,
                        num_pairs,
                        total_recv,
                        route_ms,
                        comm_dispatch_ms,
                        compute_ms,
                        comm_combine_ms,
                        reduce_ms,
                        comm_total_ms,
                        route_s[0],
                        route_s[1],
                        route_s[2],
                        route_s[3],
                        dispatch_s[0],
                        dispatch_s[1],
                        dispatch_s[2],
                        dispatch_s[3],
                        compute_s[0],
                        compute_s[1],
                        compute_s[2],
                        compute_s[3],
                        combine_s[0],
                        combine_s[1],
                        combine_s[2],
                        combine_s[3],
                        comm_s[0],
                        comm_s[1],
                        comm_s[2],
                        comm_s[3],
                        reduce_s[0],
                        reduce_s[1],
                        reduce_s[2],
                        reduce_s[3],
                        expert_max,
                        expert_mean,
                        expert_cv,
                        send_max,
                        send_mean,
                        recv_max,
                        recv_mean,
                        self._spike_comm_count,
                        self._spike_dispatch_count,
                        self._spike_combine_count,
                    ]
                )

        return out
