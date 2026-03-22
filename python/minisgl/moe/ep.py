import torch
import torch.distributed as dist
from minisgl.distributed import get_ep_info
from minisgl.distributed.impl import ep_all_to_all, get_ep_group
from minisgl.moe.base import BaseMoeBackend
from minisgl.moe.fused import fused_experts_impl, fused_topk


class EPMoe(BaseMoeBackend):
    def __init__(self) -> None:
        self._buffer_cache: dict[tuple[torch.device, torch.dtype, str], torch.Tensor] = {}

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
        ep_size = get_ep_info().size
        device = hidden_states.device
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

        # 按(目标rank, 本地expert)分桶，便于后续在接收端重建expert id
        route_key = dest_rank * num_local_experts + local_ids.to(torch.int64)
        token_idx = torch.arange(num_pairs, device=device, dtype=torch.int64) // topk
        sort_idx = torch.argsort(route_key)
        sorted_token_idx = token_idx[sort_idx]
        send_hidden = hidden_states[sorted_token_idx].contiguous()

        # 交换每个目标rank上的expert计数（小消息），避免再发送逐token expert id
        send_expert_counts = torch.bincount(
            route_key,
            minlength=ep_size * num_local_experts,
        ).view(ep_size, num_local_experts)
        send_counts = send_expert_counts.sum(dim=1)
        recv_expert_counts = self._get_buffer(
            (device, send_expert_counts.dtype, "recv_expert_counts"),
            (ep_size, num_local_experts),
        )
        work_counts = dist.all_to_all_single(
            recv_expert_counts,
            send_expert_counts,
            group=get_ep_group(),
            async_op=True,
        )
        work_counts.wait()

        recv_counts = recv_expert_counts.sum(dim=1)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        total_recv = sum(recv_splits)

        recv_hidden = hidden_states.new_empty(total_recv, hidden_size)
        work_hidden = ep_all_to_all(
            recv_hidden,
            send_hidden,
            recv_splits,
            send_splits,
            async_op=True,
        )
        if work_hidden is not None:
            work_hidden.wait()

        # 接收顺序是source-rank主序，在每个source段内按expert id分组
        if total_recv > 0:
            expert_template = (
                torch.arange(num_local_experts, device=device, dtype=torch.int32)
                .unsqueeze(0)
                .expand(ep_size, -1)
                .reshape(-1)
            )
            recv_ids = torch.repeat_interleave(expert_template, recv_expert_counts.reshape(-1))
        else:
            recv_ids = local_ids.new_empty(0)

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

        # 直接按token聚合，避免中间大tensor重排
        sorted_weights = topk_weights.reshape(-1)[sort_idx].to(hidden_states.dtype)
        out = hidden_states.new_zeros(num_tokens, hidden_size)
        out.index_add_(0, sorted_token_idx, combined * sorted_weights.unsqueeze(-1))
        return out
