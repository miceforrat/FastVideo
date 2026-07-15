import argparse
import os

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.profiling.hack_transformer_block import (
    HackCausalWanTransformerBlock,
)
from fastvideo.models.dits.causal_wanvideo import AttentionBackendEnum
from fastvideo.forward_context import set_forward_context
from fastvideo.utils import set_mixed_precision_policy
from fastvideo.profiling.small_node_profiler import get_current_simple_profiler
from fastvideo.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

from fastvideo.profiling.data_analysis import summarize_multi_rank_stats,explore_node_dict, expand_summary_with_stats

def _get_duration(node: dict, default: float = 0.0):
    if node is None:
        return default
    return node.get("stats", {}).get("duration_avg_ms", default)


def _get_child(node: dict, name: str):
    if node is None:
        return None
    return node.get("sub_nodes", {}).get(name, None)


def print_excelwise_block_data(all_stats: dict):
    root = all_stats["sub_nodes"]["ROOT"]

    fwd_block = _get_child(root, "fwd_block")

    labels = [
        "fwd_block",
        "prepare",
        "self_attn",
        "cross_attn_core",
        "ffn",
        "fc_in",
        "act",
        "fc_out",
        "hs_norm",
        "to_q",
        "to_k",
        "to_v",
        "q_rms_norm",
        "core_attn",
        "to_out",
    ]

    print("\t".join(labels))

    block_modules = fwd_block.get("sub_nodes", {})

    prepare = block_modules.get("prepare")
    self_attn = block_modules.get("self_attn")
    cross_attn_core = block_modules.get("cross_attn_core")
    ffn = block_modules.get("ffn")

    ffn_submodules = ffn.get("sub_nodes", {}) if ffn is not None else {}
    self_attn_submodules = self_attn.get("sub_nodes", {}) if self_attn is not None else {}

    vals = [
        _get_duration(fwd_block),
        _get_duration(prepare),
        _get_duration(self_attn),
        _get_duration(cross_attn_core),
        _get_duration(ffn),

        _get_duration(ffn_submodules.get("fc_in")),
        _get_duration(ffn_submodules.get("act")),
        _get_duration(ffn_submodules.get("fc_out")),

        _get_duration(self_attn_submodules.get("hs_norm")),
        _get_duration(self_attn_submodules.get("to_q")),
        _get_duration(self_attn_submodules.get("to_k")),
        _get_duration(self_attn_submodules.get("to_v")),
        _get_duration(self_attn_submodules.get("q_rms_norm")),
        _get_duration(self_attn_submodules.get("core_attn")),
        _get_duration(self_attn_submodules.get("to_out")),
    ]

    print("\t".join(str(v) for v in vals))

# =========================
# Fixed runtime config
# =========================
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29501"
WARMUP_ITERS = 5
PROFILE_ITERS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile one HackCausalWanTransformerBlock under AR-like chunked inputs."
    )

    parser.add_argument(
        "--seqlen",
        type=int,
        default=1560,
        help="Sequence length of one frame.",
    )
    parser.add_argument(
        "--kv-cache-frames",
        type=int,
        default=21,
        help="KV cache size in number of frames. Actual token capacity = kv_cache_frames * seqlen.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Number of frames per chunk.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=7,
        help="Number of chunks to simulate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Per-head attention dimension.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=8960,
        help="FFN hidden dimension.",
    )
    
    parser.add_argument(
        "--text-len",
        type=int,
        default=512,
        help="textlen.",
    )
    
    parser.add_argument(
        "--profile-chunk-idx",
        type=int,
        required=True,
        help="Required. Profile only this chunk index, e.g. 0, 1, 2...",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="logs/profile_results.csv",
        help="Path to CSV file for appending profile results.",
    )


    return parser.parse_args()


def get_dim(args):
    return args.num_heads * args.head_dim


def get_kv_cache_token_size(args):
    return args.kv_cache_frames * args.seqlen


def build_block(args):
    dim = get_dim(args)

    block = HackCausalWanTransformerBlock(
        dim=dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
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
    ).to(DEVICE).eval().to(DTYPE)

    return block


def build_static_inputs(args):
    """
    Build inputs that are reused across chunks:
    - encoder_hidden_states
    - crossattn_cache
    - kv_cache
    """
    dim = get_dim(args)
    bs = args.batch_size
    kv_cache_tokens = get_kv_cache_token_size(args)

    encoder_hidden_states = torch.randn(
        bs, args.text_len, dim, device=DEVICE, dtype=DTYPE
    )

    crossattn_cache = {
        "k": torch.randn(
            bs, args.text_len, args.num_heads, args.head_dim, device=DEVICE, dtype=DTYPE
        ),
        "v": torch.randn(
            bs, args.text_len, args.num_heads, args.head_dim, device=DEVICE, dtype=DTYPE
        ),
        "is_init": True,
    }

    kv_cache = {
        "k": torch.randn(
            bs,
            kv_cache_tokens,
            args.num_heads,
            args.head_dim,
            device=DEVICE,
            dtype=DTYPE,
        ),
        "v": torch.randn(
            bs,
            kv_cache_tokens,
            args.num_heads,
            args.head_dim,
            device=DEVICE,
            dtype=DTYPE,
        ),
        "global_end_index": torch.tensor(0, device=DEVICE, dtype=torch.int64),
        "local_end_index": torch.tensor(0, device=DEVICE, dtype=torch.int64),
    }

    return dict(
        encoder_hidden_states=encoder_hidden_states,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
    )


def build_chunk_inputs(
    args,
    encoder_hidden_states,
    kv_cache,
    crossattn_cache,
    *,
    start_frame_index: int,
    chunk_frames: int,
):
    """
    Build inputs for one AR chunk.

    start_frame_index: frame index where current chunk starts
    chunk_frames: number of frames in this chunk
    """
    dim = get_dim(args)

    seq_len = chunk_frames * args.seqlen
    current_start = start_frame_index * args.seqlen
    cache_start = 0

    hidden_states = torch.randn(
        args.batch_size,
        seq_len,
        dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    temb = torch.randn(
        args.batch_size,
        1,
        6,
        dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    freqs_cis = (
        torch.randn(seq_len, args.head_dim, device=DEVICE, dtype=torch.float64),
        torch.randn(seq_len, args.head_dim, device=DEVICE, dtype=torch.float64),
    )

    return dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        freqs_cis=freqs_cis,
        block_mask=None,
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


def profile_ar_diffusion_once(
    args,
    block,
    encoder_hidden_states,
    kv_cache,
    crossattn_cache,
):
    """
    Simulate one full AR-like forward composed of multiple chunks.
    """
    outputs = []
    start_frame_index = 0

    block_sizes = [args.chunk_size] * args.num_chunks

    for chunk_id, chunk_frames in enumerate(block_sizes):
        current_start = start_frame_index * args.seqlen

        inputs = build_chunk_inputs(
            args,
            encoder_hidden_states,
            kv_cache,
            crossattn_cache,
            start_frame_index=start_frame_index,
            chunk_frames=chunk_frames,
        )

        with torch.cuda.nvtx.range(
            f"chunk_{chunk_id}_startFrame{start_frame_index}"
            f"_frames{chunk_frames}_cur{current_start}"
        ):
            out = forward_once(block, **inputs)

        outputs.append(out)
        start_frame_index += chunk_frames

    return outputs


@torch.no_grad()
def main():
    args = parse_args()

    os.environ.setdefault("MASTER_ADDR", MASTER_ADDR)
    os.environ.setdefault("MASTER_PORT", MASTER_PORT)

    distributed_init_method = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"

    torch.cuda.set_device(0)

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=distributed_init_method,
    )

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

    print("===== Config =====")
    print(f"seqlen/frame       : {args.seqlen}")
    print(f"kv_cache_frames    : {args.kv_cache_frames}")
    print(f"kv_cache_tokens    : {get_kv_cache_token_size(args)}")
    print(f"chunk_size(frames) : {args.chunk_size}")
    print(f"num_chunks         : {args.num_chunks}")
    print(f"total_frames       : {args.chunk_size * args.num_chunks}")
    print(f"batch_size         : {args.batch_size}")
    print(f"num_heads          : {args.num_heads}")
    print(f"head_dim           : {args.head_dim}")
    print(f"dim                : {get_dim(args)}")
    print(f"ffn_dim            : {args.ffn_dim}")
    print("==================")

    block = build_block(args)
    block.iter_fwds = args.num_chunks

    static_inputs = build_static_inputs(args)

    encoder_hidden_states = static_inputs["encoder_hidden_states"]
    kv_cache = static_inputs["kv_cache"]
    crossattn_cache = static_inputs["crossattn_cache"]

    for _ in range(WARMUP_ITERS):
        _ = profile_ar_diffusion_once(
            args,
            block,
            encoder_hidden_states,
            kv_cache,
            crossattn_cache,
        )

    print("warmup finished!!!")

    get_current_simple_profiler().set_nvtx_profiling(True)
    get_current_simple_profiler().set_module_profiling(True)

    outputs = None
    for _ in range(PROFILE_ITERS):
        outputs = profile_ar_diffusion_once(
            args,
            block,
            encoder_hidden_states,
            kv_cache,
            crossattn_cache,
        )

    torch.cuda.synchronize()

    last_out = outputs[-1]

    print("forward ok")
    res = get_current_simple_profiler().collect_all_nodes_as_dict()
    summarized = summarize_multi_rank_stats([explore_node_dict(res)])
    # for route in summarized.keys():
    #     print(route)
    expanded = expand_summary_with_stats(summarized)
    print_excelwise_block_data(expanded)
    # print("num chunks:", len(outputs))
    # print("last output shape:", tuple(last_out.shape))
    # print("last output dtype:", last_out.dtype)
    # print("last output device:", last_out.device)
    # print("final kv global_end:", kv_cache["global_end_index"].item())
    # print("final kv local_end:", kv_cache["local_end_index"].item())


if __name__ == "__main__":
    main()