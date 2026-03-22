from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers.quant import dequantize_int8_per_channel
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None
        # For quantized weights
        self.qweight = None
        self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle quantized weights
        if self.is_quantized:
            weight = self.dequantize_weight(x.dtype)
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)

    @property
    def is_quantized(self) -> bool:
        return self.qweight is not None

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_int8_per_channel(self.qweight, self.scale, dtype)

    def load_state_dict(
        self,
        state_dict: dict,
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
            bias_key = _concat_prefix(prefix, "bias")
            bias = state_dict.pop(bias_key, None)

            assert qweight.dtype == torch.int8, f"Expected int8 qweight, got {qweight.dtype}"
            assert qweight.shape == (self.local_output_size, self.local_input_size)

            self.qweight = qweight
            self.scale = scale
            self.weight = None  # Clear fp weight
            if bias is not None:
                self.bias = bias
        else:
            # Standard FP weight loading
            weight_key = _concat_prefix(prefix, "weight")
            if weight_key in state_dict:
                item = state_dict.pop(weight_key)
                assert isinstance(item, torch.Tensor)
                assert item.shape == (self.local_output_size, self.local_input_size)
                self.weight = item

            bias_key = _concat_prefix(prefix, "bias")
            if bias_key in state_dict and self.bias is not None:
                item = state_dict.pop(bias_key)
                assert isinstance(item, torch.Tensor)
                assert item.shape == (self.local_output_size,)
                self.bias = bias

        if not _internal and state_dict:
            pass  # Don't raise error for unused keys at this level


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle quantized weights
        if self.is_quantized:
            weight = self.dequantize_weight(x.dtype)
        else:
            weight = self.weight
        y = F.linear(x, weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle quantized weights
        if self.is_quantized:
            weight = self.dequantize_weight(x.dtype)
        else:
            weight = self.weight
        y = F.linear(x, weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
