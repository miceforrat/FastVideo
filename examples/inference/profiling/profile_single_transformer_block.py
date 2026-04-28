# input_args: 	dim: 1536 	ffn_dim: 8960 	num_heads: 12 	local_attn_size: -1 	sink_size: 0 	qk_norm: rms_norm_across_heads 	cross_attn_norm: True 	eps: 1e-06 	added_kv_proj_dim: None 	supported_attn_backends: (<AttentionBackendEnum.SAGE_ATTN: 3>, <AttentionBackendEnum.FLASH_ATTN: 1>, <AttentionBackendEnum.TORCH_SDPA: 2>, <AttentionBackendEnum.VIDEO_SPARSE_ATTN: 5>, <AttentionBackendEnum.VMOBA_ATTN: 6>, <AttentionBackendEnum.SAGE_ATTN_THREE: 4>, <AttentionBackendEnum.SLA_ATTN: 7>, <AttentionBackendEnum.SAGE_SLA_ATTN: 8>) 	prefix: Wan.blocks.0
# (Worker pid=219961) ===== DEBUG INPUTS =====
# (Worker pid=219961) hidden_states: shape=(1, 4680, 1536), dtype=torch.bfloat16, device=cuda:0
# (Worker pid=219961) encoder_hidden_states: shape=(1, 512, 1536), dtype=torch.bfloat16, device=cuda:0
# (Worker pid=219961) temb: shape=(1, 1, 6, 1536), dtype=torch.bfloat16, device=cuda:0
# (Worker pid=219961) freqs_cis: tuple(len=2)
# (Worker pid=219961)   freqs_cis[0]: shape=(4680, 128), dtype=torch.float64, device=cuda:0
# (Worker pid=219961)   freqs_cis[1]: shape=(4680, 128), dtype=torch.float64, device=cuda:0
# (Worker pid=219961) block_mask: None
# (Worker pid=219961) kv_cache: dict(keys=['k', 'v', 'global_end_index', 'local_end_index'])
# (Worker pid=219961)   kv_cache[k]: shape=(1, 32760, 12, 128), dtype=torch.bfloat16, device=cuda:0
# (Worker pid=219961)   kv_cache[v]: shape=(1, 32760, 12, 128), dtype=torch.bfloat16, device=cuda:0
# (Worker pid=219961)   kv_cache[global_end_index]: shape=(1,), dtype=torch.int64, device=cuda:0
# (Worker pid=219961)   kv_cache[local_end_index]: shape=(1,), dtype=torch.int64, device=cuda:0
# (Worker pid=219961) crossattn_cache: dict(keys=['k', 'v', 'is_init'])
# (Worker pid=219961)   crossattn_cache[k]: shape=(1, 512, 12, 128), dtype=torch.bfloat16, device=cuda:0
# (Worker pid=219961)   crossattn_cache[v]: shape=(1, 512, 12, 128), dtype=torch.bfloat16, device=cuda:0
# (Worker pid=219961)   crossattn_cache[is_init]: type=<class 'bool'>
# (Worker pid=219961) current_start: 9360
# (Worker pid=219961) cache_start: 0
# (Worker pid=219961) ========================

from fastvideo.profiling.hack_transformer_block import (
    HackCausalWanTransformerBlock,
)
from fastvideo.models.dits.causal_wanvideo import AttentionBackendEnum
from fastvideo.forward_context import set_forward_context
from fastvideo.utils import set_mixed_precision_policy
from fastvideo.profiling.small_node_profiler import get_current_simple_profiler

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


# =========================
# AR-like diffusion config
# =========================
FRAME_SEQ_LENGTH = 1560
BLOCK_SIZES = [3, 3, 3, 3, 3, 3, 3]   # total 21 frames
TOTAL_FRAMES = sum(BLOCK_SIZES)
MAX_SEQ_LEN = TOTAL_FRAMES * FRAME_SEQ_LENGTH

BS = 1
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
NUM_HEADS=12
HEAD_DIM=128
DIM = NUM_HEADS* HEAD_DIM
from fastvideo.attention.selector import  global_force_attn_backend

def build_block():
    block = HackCausalWanTransformerBlock(
        dim=DIM,
        ffn_dim=8960,
        num_heads=NUM_HEADS,
        local_attn_size=-1,
        sink_size=0,
        qk_norm="rms_norm_across_heads",
        cross_attn_norm=True,
        eps=1e-6,
        added_kv_proj_dim=None,
        supported_attention_backends=(
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.SAGE_ATTN_THREE,
            AttentionBackendEnum.SLA_ATTN,
            AttentionBackendEnum.SAGE_SLA_ATTN,
        ),
        prefix="Wan.blocks.0",
    ).cuda().eval().to(torch.bfloat16)

    return block


def build_static_inputs(device=DEVICE, bs=BS):
    """
    构造与 chunk 无关、可以复用的部分：
    - encoder_hidden_states
    - crossattn_cache
    - kv_cache（完整容量预分配）
    """
    encoder_hidden_states = torch.randn(
        bs, 512, DIM, device=device, dtype=DTYPE
    )

    crossattn_cache = {
        "k": torch.randn(
            bs, 512, NUM_HEADS, HEAD_DIM, device=device, dtype=DTYPE
        ),
        "v": torch.randn(
            bs, 512, NUM_HEADS, HEAD_DIM, device=device, dtype=DTYPE
        ),
        "is_init": True,
    }

    # 预分配完整 AR 序列的 KV cache
    kv_cache = {
        "k": torch.randn(
            bs, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=DTYPE
        ),
        "v": torch.randn(
            bs, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=DTYPE
        ),
        # 初始为空 cache
        "global_end_index": torch.tensor(0, device=device, dtype=torch.int64),
        "local_end_index": torch.tensor(0, device=device, dtype=torch.int64),
    }

    return dict(
        encoder_hidden_states=encoder_hidden_states,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
    )


def build_chunk_inputs(
    encoder_hidden_states,
    kv_cache,
    crossattn_cache,
    *,
    start_index: int,
    frames: int,
    frame_seq_length: int = FRAME_SEQ_LENGTH,
):
    """
    为单个 chunk 构造输入，模拟：
    [chunk] start_index=..., frames=..., seq_len=..., current_start=...
    [kv] global_end=..., local_end=...
    """
    seq_len = frames * frame_seq_length
    current_start = start_index * frame_seq_length
    cache_start = 0

    hidden_states = torch.randn(
        BS, seq_len, DIM, device=DEVICE, dtype=DTYPE
    )

    # temb 的 frame 维度与当前 chunk 的 frames 对齐
    temb = torch.randn(
        BS, 1, 6, DIM, device=DEVICE, dtype=DTYPE
    )

    freqs_cis = (
        torch.randn(seq_len, 128, device=DEVICE, dtype=torch.float64),
        torch.randn(seq_len, 128, device=DEVICE, dtype=torch.float64),
    )

    block_mask = None

    # 在进入本 chunk 前，cache 的有效长度应当等于 current_start
    kv_cache["global_end_index"].fill_(current_start)
    kv_cache["local_end_index"].fill_(current_start)

    return dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        freqs_cis=freqs_cis,
        block_mask=block_mask,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        current_start=current_start,
        cache_start=cache_start,
        original_seq_len=seq_len,
    )


def forward_once(block, **inputs):
    with torch.autocast(
        device_type="cuda",
        dtype=DTYPE,
        enabled=True,
    ), set_forward_context(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=None,
    ):
        res = block.forward(**inputs)
    return res


def profile_ar_diffusion_once(block, encoder_hidden_states, kv_cache, crossattn_cache):
    """
    模拟一次完整 forward 内部的所有 chunk 生成过程。
    """
    outputs = []
    start_index = 0

    # print(f"frame_seq_length = {FRAME_SEQ_LENGTH}")
    # print(f"block_sizes = {BLOCK_SIZES}")
    chunk_id=0
    for frames in BLOCK_SIZES:
        seq_len = frames * FRAME_SEQ_LENGTH
        current_start = start_index * FRAME_SEQ_LENGTH

        # print(
        #     f"[chunk] start_index={start_index}, "
        #     f"frames={frames}, seq_len={seq_len}, current_start={current_start}"
        # )
        # print(
        #     f"[kv] global_end={kv_cache['global_end_index'].item()}, "
        #     f"local_end={kv_cache['local_end_index'].item()}"
        # )

        inputs = build_chunk_inputs(
            encoder_hidden_states,
            kv_cache,
            crossattn_cache,
            start_index=start_index,
            frames=frames,
            frame_seq_length=FRAME_SEQ_LENGTH,
        )

        with torch.cuda.nvtx.range(
            f"chunk_{chunk_id}_start{start_index}_frames{frames}_cur{current_start}"
        ):
            out = forward_once(block, **inputs)
        outputs.append(out)

        # 模拟该 chunk forward 后，kv cache 有效长度推进到本 chunk 结束
        # new_end = current_start + seq_len
        # kv_cache["global_end_index"].fill_(new_end)
        # kv_cache["local_end_index"].fill_(new_end)
        # print()
        start_index += frames
        chunk_id+=1

    return outputs

from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
import os
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")

distributed_init_method = "tcp://127.0.0.1:29501"

@torch.no_grad()
def main():
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=distributed_init_method,
    )
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        sequence_model_parallel_size=1,
    )
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32
    output_dtype = None

    mp_policy = MixedPrecisionPolicy(
        param_dtype,
        reduce_dtype,
        output_dtype,
        cast_forward_inputs=False,
    )

    set_mixed_precision_policy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )

    torch.cuda.set_device(0)

    block = build_block()
    block.iter_fwds = 7 # only chunk_nums
    static_inputs = build_static_inputs(device=DEVICE, bs=BS)

    encoder_hidden_states = static_inputs["encoder_hidden_states"]
    kv_cache = static_inputs["kv_cache"]
    crossattn_cache = static_inputs["crossattn_cache"]
    warmup_iters = 5
    # warmup
    for _ in range(warmup_iters):
        kv_cache["global_end_index"].zero_()
        kv_cache["local_end_index"].zero_()
        _ = profile_ar_diffusion_once(
            block, encoder_hidden_states, kv_cache, crossattn_cache
        )
    print("warmup finished!!!")

    # torch.cuda.synchronize()

    # 正式 profile

    # s = torch.cuda.Event(enable_timing=True)
    # e = torch.cuda.Event(enable_timing=True)
    # s.record()
    get_current_simple_profiler().set_nvtx_profiling(True)
    for i in range(1):
        # print(f"\n===== profile iter {i} =====")
        kv_cache["global_end_index"].zero_()
        kv_cache["local_end_index"].zero_()
        outputs = profile_ar_diffusion_once(
            block, encoder_hidden_states, kv_cache, crossattn_cache
        )
    # e.record()
    torch.cuda.synchronize()
    
    last_out = outputs[-1]
    print("forward ok")
    print("num chunks:", len(outputs))
    print("last output shape:", tuple(last_out.shape))
    print("last output dtype:", last_out.dtype)
    print("last output device:", last_out.device)
    print("final kv global_end:", kv_cache["global_end_index"].item())
    print("final kv local_end:", kv_cache["local_end_index"].item())
    # print(f"fwd time: {s.elapsed_time(e)}")


if __name__ == "__main__":
    main()