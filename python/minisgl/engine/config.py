from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype # 模型权重和计算精度
    max_running_req: int = 256 # 同时处理多少个用户请求
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1 # KV缓存分页
    memory_ratio: float = 0.9 # 显存使用比例
    distributed_timeout: float = 60.0 # 分布式通信超时时间
    use_dummy_weight: bool = False # 使用虚拟权重（验证速度、ep/tp）
    use_pynccl: bool = True # 使用pynccl
    max_seq_len_override: int | None = None # 强制覆盖模型配置
    num_page_override: int | None = None  # if not None, will override the number of pages
    quantization: str | None = None  # "int8" for dynamic int8 quantization
    ep_size: int = 1 # 新增
    

    #加载hf配置
    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    # 属性计算
    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    # 最大前向长度
    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    # 分布式通信的地址
    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
