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

        ep_size = get_ep_info().size
        num_tokens, hidden_size = hidden_states.shape
        num_local_experts = w1.shape[0]
        num_pairs = num_tokens * topk
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
        if do_profile and ev_route_end is not None:
            ev_route_end.record()
        
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=get_ep_group())
        # 交换hidden state
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        total_recv = sum(recv_splits)

        recv_hidden = hidden_states.new_empty(total_recv, hidden_size)
        ep_all_to_all(recv_hidden, send_hidden, recv_splits, send_splits)
        # 交换expert id
        recv_ids = sorted_local_ids.new_empty(total_recv)
        ep_all_to_all(recv_ids, sorted_local_ids, recv_splits, send_splits)
        if do_profile and ev_comm_dispatch_end is not None:
            ev_comm_dispatch_end.record()

        # 本地expert 计算
        if total_recv > 0:
            unit_weights = torch.ones(
                total_recv, 1, dtype=torch.float32, device=hidden_states.device
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
        ep_all_to_all(combined, local_out, send_splits, recv_splits)
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

        return out
