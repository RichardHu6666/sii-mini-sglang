from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info, set_ep_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)

# forward返回值定义
class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        # 分布式信息设置
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        set_ep_info(
            rank=config.tp_info.rank % max(config.ep_size, 1),
            size=config.ep_size
        )
        _adjust_config(config)

        # gpu设备设置
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        # 创建cuda流
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        # 初始化
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)
        # 通信初始化
        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )
    # 通信初始化
    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        ## 启用ep
        if config.ep_size > 1:
            torch.distributed.init_process_group(
                backend="nccl" ,
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # cpu用gloo通信库，避免显存不均衡
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
            ## 设置ep进程组
            from minisgl.distributed.impl import set_ep_group
            
            ep_nccl_group = torch.distributed.group.WORLD
            assert ep_nccl_group is not None
            set_ep_group(ep_nccl_group)
            logger.info_rank0(f"Ep enabled: ep_size={config.ep_size}")
            
            ## 启用pynccl
            if config.use_pynccl:
                max_bytes = (
                    config.max_forward_len
                    * config.model_config.hidden_size
                    * self.dtype.itemsize
                )
                enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        ## 或者单gpu / 不用pynccl
        elif config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len
                * config.model_config.hidden_size
                *self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        ## TP但不用pynccl
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        
        return tp_cpu_group
            
    # 权重加载
    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        # 虚拟权重快速测试
        if config.use_dummy_weight:
            result: Dict[str, torch.Tensor] = {}
            for k, v in self.model.state_dict().items():
                if "experts" in k:
                    prefix, param = k.rsplit("experts.", 1)
                    for i in range(v.shape[0]):
                        result[f"{prefix}experts.{i}.{param}"] = torch.randn_like(
                            v[i], device=self.device
                        )
                else:
                    result[k] = torch.randn_like(v, device=self.device)
            return result
        else:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # Load weights and apply dynamic quantization if requested
            state_dict = {}
            for k, v in load_weight(config.model_path, self.device):
                v = v.to(self.dtype)

                # Apply dynamic int8 quantization if requested
                if config.quantization == "int8" and _should_quantize(k):
                    # Quantize weight to int8 and store scale separately
                    qweight, scale = _dynamic_quantize_int8_per_channel(v)
                    state_dict[k.replace(".weight", ".qweight")] = qweight
                    state_dict[k.replace(".weight", ".scale")] = scale
                else:
                    state_dict[k] = v
            return state_dict

                k: v.to(self.device)
                for k, v in load_weight(
                    config.model_path,
                    self.device,
                    ep_size=config.ep_size
                )
            }
    # kv缓存页数确定
    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    # 显存同步检查
    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory
    # 前向传播
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _dynamic_quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform per-channel dynamic int8 quantization on weight tensor.

    Args:
        weight: 2D weight tensor of shape (out_features, in_features) in fp16/bf16/fp32

    Returns:
        qweight: int8 quantized weight
        scale: per-channel scale factor (out_features,)
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"

    # Compute per-channel (row-wise) min/max
    min_val = weight.min(dim=1, keepdim=True)[0]
    max_val = weight.max(dim=1, keepdim=True)[0]

    # Compute scale: scale = max(|min|, |max|) / 127
    eps = torch.finfo(torch.float32).eps
    max_abs = torch.maximum(-min_val, max_val)
    scale = max_abs / 127.0
    scale = torch.clamp(scale, min=eps)

    # Quantize: q = round(w / scale) and clamp to [-127, 127]
    qweight = torch.round(weight / scale)
    qweight = torch.clamp(qweight, -127, 127).to(torch.int8)

    # Return scale as 1D tensor
    scale = scale.squeeze(dim=1)

    return qweight, scale


def _should_quantize(name: str) -> bool:
    """Check if a weight tensor should be quantized.

    We quantize linear layer weights but not norms, embeddings, etc.
    """
    # Quantize projection layers
    quantize_suffixes = [
        ".qkv_proj.weight", ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
        ".o_proj.weight", ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
        ".gate_up_proj.weight", ".lm_head.weight",
    ]
    return any(name.endswith(suffix) for suffix in quantize_suffixes)

# 配置调整
def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)
    ## 自动选择注意力后端
    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")
    ## tensorRT LLM约束 16/32/64
    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")
    ## 自动选择MoE后端
    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
    ## ep约束 关闭cuda graph
    if config.ep_size > 1:
        assert config.model_config.is_moe, "EP needs MoE models"
        assert config.model_config.num_experts % config.ep_size == 0, (
            f"num of experts ({config.model_config.num_experts})"
            f"must be divisible by ep size ({config.ep_size})"
        )
        if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs> 0:
            override("cuda_graph_max_bs", 0)
            logger.info_rank0("CUDA graphs disabled by ep mode")