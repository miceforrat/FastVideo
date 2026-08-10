import argparse
import csv
import os

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.profiling.hack_transformer_block import (
    HackCausalWanTransformerBlock,
)

from fastvideo.models.dits.causal_wanvideo import AttentionBackendEnum
from fastvideo.forward_context import set_forward_context
from fastvideo.utils import set_mixed_precision_policy

from fastvideo.profiling.small_node_profiler import (
    get_current_simple_profiler,
)

from fastvideo.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

from fastvideo.profiling.data_analysis import (
    summarize_multi_rank_stats,
    explore_node_dict,
    expand_summary_with_stats,
)


# =========================================================
# Green Context
# =========================================================

from torch.cuda.green_contexts import GreenContext


# =========================================================
# Runtime config
# =========================================================

DEVICE = "cuda:0"
DTYPE = torch.bfloat16

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29501"

WARMUP_ITERS = 5
PROFILE_ITERS = 3

# limit SMs
GREEN_SMS = 102


# =========================================================
# Helpers
# =========================================================

def _get_duration(node: dict, default: float = 0.0):
    if node is None:
        return default
    return node.get("stats", {}).get("duration_avg_ms", default)


def _get_child(node: dict, name: str):
    if node is None:
        return None
    return node.get("sub_nodes", {}).get(name, None)


def append_profile_row_to_csv(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    file_exists = os.path.exists(csv_path)
    file_empty = (not file_exists) or os.path.getsize(csv_path) == 0

    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if file_empty:
            writer.writeheader()

        writer.writerow(row)


# =========================================================
# Args
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seqlen",
        type=int,
        default=1560
    )

    parser.add_argument(
        "--kv-cache-frames",
        type=int,
        default=21,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--chunk-idx",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=8960,
    )

    parser.add_argument(
        "--text-len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="profile_results.csv",
    )

    return parser.parse_args()


# =========================================================
# Shapes
# =========================================================

def get_dim(args):
    return args.num_heads * args.head_dim


def get_kv_cache_token_size(args):
    return args.kv_cache_frames * args.seqlen


# =========================================================
# Model
# =========================================================

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


# =========================================================
# Inputs
# =========================================================

def build_static_inputs(args):

    dim = get_dim(args)

    bs = args.batch_size

    kv_cache_tokens = get_kv_cache_token_size(args)

    history_frames = args.chunk_idx * args.chunk_size

    history_tokens = history_frames * args.seqlen

    history_tokens = min(
        history_tokens,
        kv_cache_tokens
    )


    encoder_hidden_states = torch.randn(
        bs,
        args.text_len,
        dim,
        device=DEVICE,
        dtype=DTYPE,
    )


    crossattn_cache = {

        "k": torch.randn(
            bs,
            args.text_len,
            args.num_heads,
            args.head_dim,
            device=DEVICE,
            dtype=DTYPE,
        ),

        "v": torch.randn(
            bs,
            args.text_len,
            args.num_heads,
            args.head_dim,
            device=DEVICE,
            dtype=DTYPE,
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

        "global_end_index": torch.tensor(
            history_tokens,
            device=DEVICE,
            dtype=torch.int64,
        ),

        "local_end_index": torch.tensor(
            history_tokens,
            device=DEVICE,
            dtype=torch.int64,
        ),
    }


    return {
        "encoder_hidden_states": encoder_hidden_states,
        "kv_cache": kv_cache,
        "crossattn_cache": crossattn_cache,
        "history_frames": history_frames,
        "history_tokens": history_tokens,
    }



def build_chunk_inputs(
    args,
    encoder_hidden_states,
    kv_cache,
    crossattn_cache,
):

    dim = get_dim(args)

    chunk_frames = args.chunk_size

    seq_len = chunk_frames * args.seqlen

    start_frame_index = args.chunk_idx * args.chunk_size

    current_start = start_frame_index * args.seqlen


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
        torch.randn(
            seq_len,
            args.head_dim,
            device=DEVICE,
            dtype=torch.float64,
        ),
        torch.randn(
            seq_len,
            args.head_dim,
            device=DEVICE,
            dtype=torch.float64,
        ),
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
        cache_start=0,
        original_seq_len=seq_len,
    )


# =========================================================
# Forward
# =========================================================

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

        out = block.forward(**inputs)

    return out


# =========================================================
# Main
# =========================================================

@torch.no_grad()
def main():

    args = parse_args()


    os.environ.setdefault(
        "MASTER_ADDR",
        MASTER_ADDR
    )

    os.environ.setdefault(
        "MASTER_PORT",
        MASTER_PORT
    )


    torch.cuda.set_device(0)


    # =====================================================
    # Create Green Context
    # =====================================================

    green_ctx = GreenContext.create(
        num_sms=GREEN_SMS
    )

    green_stream = green_ctx.Stream()


    print(
        f"Using Green Context: {GREEN_SMS} SMs"
    )


    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=
            f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
    )


    initialize_model_parallel(
        tensor_model_parallel_size=1,
        sequence_model_parallel_size=1,
    )


    set_mixed_precision_policy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        mp_policy=MixedPrecisionPolicy(
            torch.bfloat16,
            torch.float32,
            None,
            cast_forward_inputs=False,
        ),
    )


    block = build_block(args)


    static_inputs = build_static_inputs(args)

    inputs = build_chunk_inputs(
        args,
        static_inputs["encoder_hidden_states"],
        static_inputs["kv_cache"],
        static_inputs["crossattn_cache"],
    )


    # =====================================================
    # warmup on Green Stream
    # =====================================================

    for _ in range(WARMUP_ITERS):

        with torch.cuda.stream(green_stream):

            _ = forward_once(
                block,
                **inputs
            )


    green_stream.synchronize()


    print("warmup finished")


    # =====================================================
    # profile
    # =====================================================

    prof = get_current_simple_profiler()

    prof.set_nvtx_profiling(True)

    prof.set_module_profiling(True)


    outputs = []


    for _ in range(PROFILE_ITERS):

        with torch.cuda.stream(green_stream):

            with torch.cuda.nvtx.range(
                f"chunk_{args.chunk_idx}_green_{GREEN_SMS}SM"
            ):

                out = forward_once(
                    block,
                    **inputs
                )

            outputs.append(out)


    green_stream.synchronize()


    print("forward ok")


    res = prof.collect_all_nodes_as_dict()


    summarized = summarize_multi_rank_stats(
        [explore_node_dict(res)]
    )

    expanded = expand_summary_with_stats(
        summarized
    )


    root = expanded["sub_nodes"]["ROOT"]

    fwd_block = root["sub_nodes"]["fwd_block"]


    row = {
        "chunk_idx": args.chunk_idx,
        "green_sms": GREEN_SMS,
        "fwd_block_ms": _get_duration(fwd_block),
    }


    print(row)


    append_profile_row_to_csv(
        args.csv_path,
        row,
    )


if __name__ == "__main__":
    main()