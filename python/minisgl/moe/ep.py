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
        self._buffer_cache: dict[tuple[torch.device, torch.dtype, str], torch.Tensor] = {}
        self._profile_step = 0

    def _get_buffer(
        self,
        key: tuple[torch.device, torch.dtype, str],
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        need_numel = 1
        for dim in shape:
            need_numel *= dim
        buf = self._buffer_cache.get(key)
        if buf is None or buf.numel() < need_numel:
            buf = torch.empty(need_numel, device=key[0], dtype=key[1])
            self._buffer_cache[key] = buf
        return buf[:need_numel].view(shape)

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
        do_profile = bool(ENV.EP_PROFILE)
        t0 = t1 = t2 = t3 = t4 = t5 = None
        ep_size = get_ep_info().size
        device = hidden_states.device
        num_tokens, hidden_size = hidden_states.shape
        num_local_experts = w1.shape[0]
        num_pairs = num_tokens * topk
        if do_profile:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t5 = torch.cuda.Event(enable_timing=True)
            t0.record()
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

        token_idx = torch.arange(num_pairs, device=device, dtype=torch.int64) // topk
        sort_idx = torch.argsort(dest_rank)
        sorted_token_idx = token_idx[sort_idx]
        sorted_local_ids = local_ids[sort_idx]
        send_hidden = hidden_states[sorted_token_idx].contiguous()
        if do_profile and t1 is not None:
            t1.record()

        send_counts = torch.bincount(dest_rank, minlength=ep_size)
        recv_counts = self._get_buffer(
            (device, send_counts.dtype, "recv_counts"),
            (ep_size,),
        )
        work_counts = dist.all_to_all_single(
            recv_counts,
            send_counts,
            group=get_ep_group(),
            async_op=True,
        )
        work_counts.wait()

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
            async_op=True,
        )
        work_ids = ep_all_to_all(
            recv_ids,
            sorted_local_ids,
            recv_splits,
            send_splits,
            async_op=True,
        )
        if work_hidden is not None:
            work_hidden.wait()
        if work_ids is not None:
            work_ids.wait()
        if do_profile and t2 is not None:
            t2.record()

        # 本地expert 计算
        if total_recv > 0:
            unit_weights = self._get_buffer(
                (device, torch.float32, "unit_weights"),
                (total_recv, 1),
            )
            unit_weights.fill_(1.0)
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
        if do_profile and t3 is not None:
            t3.record()

        combined = hidden_states.new_empty(num_pairs, hidden_size)
        work_combined = ep_all_to_all(
            combined,
            local_out,
            send_splits,
            recv_splits,
            async_op=True,
        )
        if work_combined is not None:
            work_combined.wait()
        if do_profile and t4 is not None:
            t4.record()

        # 直接按token聚合，避免中间大tensor重排
        sorted_weights = topk_weights.reshape(-1)[sort_idx].to(hidden_states.dtype)
        out = hidden_states.new_zeros(num_tokens, hidden_size)
        out.index_add_(0, sorted_token_idx, combined * sorted_weights.unsqueeze(-1))
        if do_profile and t5 is not None:
            t5.record()
            self._profile_step += 1
            interval = max(int(ENV.EP_PROFILE_INTERVAL.value), 1)
            if self._profile_step % interval == 0:
                torch.cuda.synchronize(device)
                prep_ms = t0.elapsed_time(t1)
                dispatch_ms = t1.elapsed_time(t2)
                expert_ms = t2.elapsed_time(t3)
                combine_ms = t3.elapsed_time(t4)
                reduce_ms = t4.elapsed_time(t5)
                logger.info_rank0(
                    "EP profile step=%d pairs=%d recv=%d ms(prep=%.3f dispatch=%.3f expert=%.3f combine=%.3f reduce=%.3f)",
                    self._profile_step,
                    num_pairs,
                    total_recv,
                    prep_ms,
                    dispatch_ms,
                    expert_ms,
                    combine_ms,
                    reduce_ms,
                )
        return out
