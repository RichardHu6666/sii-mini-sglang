from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers.quant import dequantize_int8_per_channel
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # For quantized weights
        self.qweight = None
        self.scale = None
        self._comm = DistributedCommunicator()

    @property
    def is_quantized(self) -> bool:
        return self.qweight is not None

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_int8_per_channel(self.qweight, self.scale, dtype)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        from .base import _concat_prefix

        # Check for quantized weight
        qweight_key = _concat_prefix(prefix, "qweight")
        if qweight_key in state_dict:
            qweight = state_dict.pop(qweight_key)
            scale_key = _concat_prefix(prefix, "scale")
            scale = state_dict.pop(scale_key, None)

            assert qweight.dtype == torch.int8, f"Expected int8 qweight, got {qweight.dtype}"
            assert qweight.shape == (self.num_embeddings_tp, self.weight.shape[1])

            self.qweight = qweight
            self.scale = scale
            self.weight = None
        else:
            # Standard FP weight loading
            weight_key = _concat_prefix(prefix, "weight")
            if weight_key in state_dict:
                item = state_dict.pop(weight_key)
                assert isinstance(item, torch.Tensor)
                assert item.shape == (self.num_embeddings_tp, self.weight.shape[1])
                self.weight = item

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing

        # Handle quantized weights - dequantize for embedding lookup
        if self.is_quantized:
            weight = self.dequantize_weight(x.dtype)
        else:
            weight = self.weight

        y = indexing(
            weights=weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            possible_qweight = f"{prefix}.qweight"
            possible_scale = f"{prefix}.scale"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)
            if possible_qweight in state_dict:
                state_dict.pop(possible_qweight)
            if possible_scale in state_dict:
                state_dict.pop(possible_scale)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self

        # Handle quantized weights
        if module.is_quantized:
            weight = module.dequantize_weight(x.dtype)
        else:
            weight = module.weight

        logits = F.linear(x, weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
