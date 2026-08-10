import os
import argparse
import threading
import time

from contextlib import contextmanager

import torch
import greenctx


# ============================================================
# IMPORTANT:
# Patch FastVideo ForwardContext BEFORE importing DiT/attention
# ============================================================

import fastvideo.forward_context as fv_forward_context


_forward_context_tls = threading.local()


def patched_get_forward_context():
    ctx = getattr(
        _forward_context_tls,
        "value",
        None,
    )

    assert ctx is not None, (
        "Forward context is not set for this thread. "
        "Please use set_forward_context()."
    )

    return ctx


@contextmanager
def patched_set_forward_context(
    current_timestep,
    attn_metadata,
    forward_batch=None,
):
    prev_context = getattr(
        _forward_context_tls,
        "value",
        None,
    )

    ctx = fv_forward_context.ForwardContext(
        current_timestep=current_timestep,
        attn_metadata=attn_metadata,
        forward_batch=forward_batch,
    )

    _forward_context_tls.value = ctx

    try:
        yield ctx

    finally:
        _forward_context_tls.value = prev_context


# Replace canonical functions BEFORE importing attention/model modules
fv_forward_context.get_forward_context = (
    patched_get_forward_context
)

fv_forward_context.set_forward_context = (
    patched_set_forward_context
)


# This testbench always uses the patched setter
set_forward_context = (
    patched_set_forward_context
)


# ============================================================
# NOW import FastVideo modules
# ============================================================

from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

from fastvideo.utils import (
    set_mixed_precision_policy,
)

from fastvideo.models.dits.causal_wanvideo import (
    CausalWanTransformer3DModel,
)

from fastvideo.configs.models.dits import (
    WanVideoConfig,
)

from fastvideo.models.vaes.wanvae import (
    WanDecoder3d,
    WanCausalConv3d,
    forward_context,
    first_chunk,
    feat_idx,
)


# ============================================================
# Safety verification / explicit patch
# ============================================================

import fastvideo.attention.layer as fv_attention_layer


# Even if import ordering changes later, force attention.layer
# to use the thread-local getter.
fv_attention_layer.get_forward_context = (
    patched_get_forward_context
)


print(
    "[patch] FastVideo ForwardContext -> thread local"
)

print(
    "[patch] attention.layer.get_forward_context =",
    fv_attention_layer.get_forward_context.__name__,
)


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--workload",
        type=str,
        choices=[
            "dit",
            "vae",
        ],
        default="dit",
    )


    parser.add_argument(
        "--stream0-sms",
        type=int,
        default=88,
    )


    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--num-denoise-steps",
        type=int,
        default=4,
    )


    parser.add_argument(
        "--num-latents",
        type=int,
        default=3,
    )


    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
    )


    parser.add_argument(
        "--profile-iters",
        type=int,
        default=10,
    )


    return parser.parse_args()


args = parse_args()


# ============================================================
# Runtime
# ============================================================

DEVICE = "cuda:0"

DTYPE = torch.bfloat16


torch.cuda.set_device(0)

torch.manual_seed(0)

torch.cuda.manual_seed_all(0)


# ============================================================
# Green Context
# ============================================================

total_sms = (
    torch.cuda
    .get_device_properties(0)
    .multi_processor_count
)


stream0_sms = args.stream0_sms

stream1_sms = (
    total_sms
    -
    stream0_sms
)


if stream0_sms <= 0:
    raise ValueError(
        "--stream0-sms must be > 0"
    )


if stream1_sms <= 0:
    raise ValueError(
        "--stream0-sms must be smaller than total SM count"
    )


print()
print(
    "========================================"
)

print(
    "Green Context"
)

print(
    "========================================"
)

print(
    "total SMs:",
    total_sms,
)

print(
    "stream0 SMs:",
    stream0_sms,
)

print(
    "stream1 SMs:",
    stream1_sms,
)


gc = greenctx.GreenContext(
    stream0_sms,
    0,
)


stream0 = torch.cuda.ExternalStream(
    gc.dit_stream(),
    device=0,
)


stream1 = torch.cuda.ExternalStream(
    gc.vae_stream(),
    device=0,
)


print(
    "stream0:",
    hex(gc.dit_stream()),
)

print(
    "stream1:",
    hex(gc.vae_stream()),
)


# ============================================================
# Distributed
# ============================================================

os.environ[
    "MASTER_ADDR"
] = "127.0.0.1"

os.environ[
    "MASTER_PORT"
] = "29501"


init_distributed_environment(
    world_size=1,
    rank=0,
    local_rank=0,
    distributed_init_method=
        "tcp://127.0.0.1:29501",
)


initialize_model_parallel(
    tensor_model_parallel_size=1,
    sequence_model_parallel_size=1,
)


# ============================================================
# Mixed precision
# ============================================================

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=None,
    cast_forward_inputs=False,
)


set_mixed_precision_policy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=None,
    mp_policy=mp_policy,
)


# ============================================================
# Placeholders
# ============================================================

run0 = None

run1 = None

shared_weight_ptr = None


# ============================================================
# DiT SAME-MODEL workload
# ============================================================

if args.workload == "dit":

    print()
    print(
        "========================================"
    )

    print(
        "Build ONE shared DiT"
    )

    print(
        "========================================"
    )


    hf_config = {

        "_class_name":
            "CausalWanTransformer3DModel",

        "attention_head_dim":
            128,

        "cross_attn_norm":
            True,

        "eps":
            1e-6,

        "ffn_dim":
            8960,

        "freq_dim":
            256,

        "image_dim":
            None,

        "in_channels":
            16,

        "num_attention_heads":
            12,

        "num_layers":
            30,

        "out_channels":
            16,

        "patch_size":
            [1, 2, 2],

        "qk_norm":
            "rms_norm_across_heads",

        "text_dim":
            4096,
    }


    config = WanVideoConfig()


    config.hidden_size = 1536

    config.num_attention_heads = 12

    config.attention_head_dim = 128

    config.ffn_dim = 8960

    config.num_layers = 30

    config.in_channels = 16

    config.out_channels = 16

    config.num_channels_latents = 16

    config.patch_size = [
        1,
        2,
        2,
    ]

    config.text_dim = 4096

    config.freq_dim = 256

    config.cross_attn_norm = True

    config.qk_norm = (
        "rms_norm_across_heads"
    )

    config.eps = 1e-6

    config.num_frames_per_block = 3

    config.sliding_window_num_frames = 21


    # ========================================================
    # ONLY ONE MODEL INSTANCE
    # ========================================================

    model = CausalWanTransformer3DModel(
        config=config,
        hf_config=hf_config,
    )


    model = model.to(
        DEVICE,
        dtype=DTYPE,
    )


    model.eval()


    shared_weight_ptr = (
        next(
            model.parameters()
        )
        .data_ptr()
    )


    print(
        "model object:",
        hex(id(model)),
    )

    print(
        "shared first weight:",
        hex(shared_weight_ptr),
    )


    # ========================================================
    # Inputs
    #
    # Two separate activation tensors.
    #
    # Only weights are deliberately shared.
    # ========================================================

    B = 1


    hidden_states0 = torch.randn(
        B,
        16,
        3,
        60,
        104,
        device=DEVICE,
        dtype=DTYPE,
    )


    hidden_states1 = (
        hidden_states0
        .clone()
    )


    encoder_hidden_states0 = torch.randn(
        B,
        512,
        4096,
        device=DEVICE,
        dtype=torch.float32,
    )


    encoder_hidden_states1 = (
        encoder_hidden_states0
        .clone()
    )


    chunk_size = 3


    start_frame = (
        args.chunk_idx
        *
        chunk_size
    )


    frame_seq_length = (
        60 // 2
    ) * (
        104 // 2
    )


    current_start = (
        start_frame
        *
        frame_seq_length
    )


    timesteps = torch.arange(
        args.num_denoise_steps,
        device=DEVICE,
        dtype=torch.long,
    )


    # ========================================================
    # Runtime state:
    #
    # WEIGHTS             shared
    # input               separate
    # KV cache            separate
    # crossattn cache     separate
    # ForwardContext      thread-local
    # ========================================================

    def build_dit_state():

        num_layers = 30

        num_heads = 12

        head_dim = 128


        kv_tokens = (
            frame_seq_length
            *
            21
        )


        kv_cache = []


        for _ in range(
            num_layers
        ):

            kv_cache.append({

                "k":
                    torch.zeros(
                        B,
                        kv_tokens,
                        num_heads,
                        head_dim,
                        device=DEVICE,
                        dtype=DTYPE,
                    ),

                "v":
                    torch.zeros(
                        B,
                        kv_tokens,
                        num_heads,
                        head_dim,
                        device=DEVICE,
                        dtype=DTYPE,
                    ),

                "global_end_index":
                    torch.tensor(
                        [current_start],
                        device=DEVICE,
                        dtype=torch.long,
                    ),

                "local_end_index":
                    torch.tensor(
                        [current_start],
                        device=DEVICE,
                        dtype=torch.long,
                    ),
            })


        crossattn_cache = []


        for _ in range(
            num_layers
        ):

            crossattn_cache.append({

                "k":
                    torch.zeros(
                        B,
                        512,
                        num_heads,
                        head_dim,
                        device=DEVICE,
                        dtype=DTYPE,
                    ),

                "v":
                    torch.zeros(
                        B,
                        512,
                        num_heads,
                        head_dim,
                        device=DEVICE,
                        dtype=DTYPE,
                    ),

                "is_init":
                    True,
            })


        return {
            "kv_cache":
                kv_cache,

            "crossattn_cache":
                crossattn_cache,
        }


    state0 = build_dit_state()

    state1 = build_dit_state()


    # ========================================================
    # Forward
    # ========================================================

    def dit_forward(
        latent,
        encoder_hidden_states,
        timestep,
        ctx_step,
        state,
    ):

        with torch.no_grad():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ), set_forward_context(
                current_timestep=
                    ctx_step,
                attn_metadata=
                    None,
                forward_batch=
                    None,
            ):

                out = (
                    model
                    ._forward_inference(
                        hidden_states=
                            latent,

                        encoder_hidden_states=
                            encoder_hidden_states,

                        timestep=
                            timestep.reshape(
                                1,
                                1,
                            ),

                        encoder_hidden_states_image=
                            None,

                        kv_cache=
                            state[
                                "kv_cache"
                            ],

                        crossattn_cache=
                            state[
                                "crossattn_cache"
                            ],

                        current_start=
                            current_start,

                        cache_start=
                            0,

                        start_frame=
                            start_frame,
                    )
                )

        return out


    # ========================================================
    # One complete DiT task
    # ========================================================

    def run_dit(
        hidden_states,
        encoder_hidden_states,
        state,
        nvtx_name,
    ):

        torch.cuda.nvtx.range_push(
            nvtx_name
        )


        latent = (
            hidden_states
            .clone()
        )


        torch.cuda.nvtx.range_push(
            f"{nvtx_name}_denoise"
        )


        for i, t in enumerate(
            timesteps
        ):

            pred = dit_forward(
                latent,
                encoder_hidden_states,
                t,
                i,
                state,
            )


            latent = (
                latent
                -
                0.1 * pred
            )


        torch.cuda.nvtx.range_pop()


        torch.cuda.nvtx.range_push(
            f"{nvtx_name}_KV"
        )


        _ = dit_forward(
            latent,
            encoder_hidden_states,
            torch.tensor(
                0,
                device=DEVICE,
                dtype=torch.long,
            ),
            0,
            state,
        )


        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()


        return latent


    def run0():

        return run_dit(
            hidden_states0,
            encoder_hidden_states0,
            state0,
            "DiT_stream0",
        )


    def run1():

        return run_dit(
            hidden_states1,
            encoder_hidden_states1,
            state1,
            "DiT_stream1",
        )


# ============================================================
# VAE SAME-MODEL workload
# ============================================================

elif args.workload == "vae":

    print()
    print(
        "========================================"
    )

    print(
        "Build ONE shared VAE"
    )

    print(
        "========================================"
    )


    # ========================================================
    # ONE post_quant_conv
    # ONE decoder
    # ========================================================

    post_quant_conv = WanCausalConv3d(
        16,
        16,
        1,
    )


    decoder = WanDecoder3d(
        dim=96,
        z_dim=16,
        dim_mult=[
            1,
            2,
            4,
            4,
        ],
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=[
            True,
            True,
            False,
        ],
        dropout=0.0,
        out_channels=3,
        is_residual=False,
    )


    post_quant_conv = (
        post_quant_conv
        .cuda()
        .to(dtype=torch.bfloat16)
        .eval()
    )


    decoder = (
        decoder
        .cuda()
        .to(dtype=torch.bfloat16)
        .eval()
    )


    shared_weight_ptr = (
        next(
            decoder.parameters()
        )
        .data_ptr()
    )


    print(
        "decoder object:",
        hex(id(decoder)),
    )

    print(
        "shared first weight:",
        hex(shared_weight_ptr),
    )


    # ========================================================
    # Feature cache
    # ========================================================

    def count_causal_conv(
        module,
    ):

        return sum(
            1
            for m in module.modules()
            if isinstance(
                m,
                WanCausalConv3d,
            )
        )


    num_cache = (
        count_causal_conv(
            decoder
        )
    )


    print(
        "decoder causal conv:",
        num_cache,
    )


    # ========================================================
    # Separate inputs
    # ========================================================

    T = args.num_latents


    # z0 = torch.randn(
    #     1,
    #     16,
    #     T,
    #     60,
    #     104,
    #     device=DEVICE,
    #     dtype=torch.float32,
    # )
    z0 = torch.randn(
        1,
        16,
        T,
        60,
        104,
        device=DEVICE,
        dtype=torch.bfloat16,
    )


    z1 = z0.clone()


    # ========================================================
    # Initialize separate feature caches
    # ========================================================

    def initialize_feature_cache():

        cache = [
            None
            for _ in range(
                num_cache
            )
        ]


        # dummy = torch.randn(
        #     1,
        #     16,
        #     1,
        #     60,
        #     104,
        #     device=DEVICE,
        #     dtype=torch.float32,
        # )

        dummy = torch.randn(
            1,
            16,
            1,
            60,
            104,
            device=DEVICE,
            dtype=torch.bfloat16,
        )


        dummy_x = (
            post_quant_conv(
                dummy
            )
        )


        with torch.no_grad():

            with forward_context(
                feat_cache_arg=
                    cache,
                feat_idx_arg=
                    0,
            ):

                feat_idx.set(
                    0
                )

                first_chunk.set(
                    False
                )

                _ = decoder(
                    dummy_x
                )


        torch.cuda.synchronize()


        return cache


    print(
        "initialize feature cache 0"
    )

    cache0 = (
        initialize_feature_cache()
    )


    print(
        "initialize feature cache 1"
    )

    cache1 = (
        initialize_feature_cache()
    )


    # ========================================================
    # VAE workload
    # ========================================================

    def run_vae(
        z,
        cache,
        nvtx_name,
    ):

        torch.cuda.nvtx.range_push(
            nvtx_name
        )


        with torch.no_grad():

            x = (
                post_quant_conv(
                    z
                )
            )


            outputs = []


            with forward_context(
                feat_cache_arg=
                    cache,
                feat_idx_arg=
                    0,
            ):

                for i in range(T):

                    feat_idx.set(
                        0
                    )

                    first_chunk.set(
                        False
                    )


                    out_i = decoder(
                        x[
                            :,
                            :,
                            i:i + 1,
                            :,
                            :
                        ]
                    )


                    outputs.append(
                        out_i
                    )


            out = torch.cat(
                outputs,
                dim=2,
            )


        torch.cuda.nvtx.range_pop()


        return out


    def run0():

        return run_vae(
            z0,
            cache0,
            "VAE_stream0",
        )


    def run1():

        return run_vae(
            z1,
            cache1,
            "VAE_stream1",
        )


# ============================================================
# Benchmark one Green stream
# ============================================================

def benchmark_stream(
    fn,
    stream,
):

    with torch.cuda.stream(
        stream
    ):

        for _ in range(
            args.warmup_iters
        ):

            fn()


    stream.synchronize()


    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    with torch.cuda.stream(
        stream
    ):

        start.record(
            stream
        )


        for _ in range(
            args.profile_iters
        ):

            fn()


        end.record(
            stream
        )


    end.synchronize()


    return (
        start.elapsed_time(
            end
        )
        /
        args.profile_iters
    )


# ============================================================
# Sequential using SAME TWO Green streams
# ============================================================

def benchmark_sequential():

    # warmup
    for _ in range(
        args.warmup_iters
    ):

        run0()

        torch.cuda.synchronize()

        run1()

        torch.cuda.synchronize()


    values = []


    for _ in range(
        args.profile_iters
    ):

        begin = time.perf_counter()


        run0()

        torch.cuda.synchronize()


        run1()

        torch.cuda.synchronize()


        end = time.perf_counter()


        values.append(
            (
                end - begin
            )
            *
            1000.0
        )


    return (
        sum(values)
        /
        len(values)
    )

# ============================================================
# Single-stream baseline
# ============================================================

print()
print(
    "========================================"
)

print(
    "Single-stream"
)

print(
    "========================================"
)


stream0_time = benchmark_stream(
    run0,
    stream0,
)


stream1_time = benchmark_stream(
    run1,
    stream1,
)


print(
    "stream0 alone:",
    stream0_time,
    "ms",
)


print(
    "stream1 alone:",
    stream1_time,
    "ms",
)


# ============================================================
# Sequential
# ============================================================

print()
print(
    "========================================"
)

print(
    "Sequential"
)

print(
    "========================================"
)


sequential_time = (
    benchmark_sequential()
)


print(
    "Sequential:",
    sequential_time,
    "ms",
)


# ============================================================
# Timeline logger
# ============================================================

LOG_FILE = (
    f"task_timeline_"
    f"{args.workload}_same_model.txt"
)


open(
    LOG_FILE,
    "w",
).close()


log_lock = (
    threading.Lock()
)


def log_task(
    name,
    event,
):

    ts = (
        time.perf_counter()
        *
        1000.0
    )


    with log_lock:

        with open(
            LOG_FILE,
            "a",
        ) as f:

            f.write(
                f"{name},"
                f"{event},"
                f"{ts:.3f}\n"
            )


# ============================================================
# Parallel workers
# ============================================================

def launch0(
    do_log=True,
):

    # Make CUDA device current for this Python thread
    torch.cuda.set_device(
        0
    )

    if args.workload == "dit":

        ptr = next(
            model.parameters()
        ).data_ptr()

    else:

        ptr = next(
            decoder.parameters()
        ).data_ptr()


    print(
        "launch0 thread:",
        threading.current_thread().name,
        "weight ptr:",
        hex(ptr),
    )



    if do_log:
        log_task(
            "stream0",
            "start",
        )


    with torch.cuda.stream(
        stream0
    ):

        run0()


    stream0.synchronize()


    if do_log:
        log_task(
            "stream0",
            "end",
        )


def launch1(
    do_log=True,
):

    torch.cuda.set_device(
        0
    )

    if args.workload == "dit":

        ptr = next(
            model.parameters()
        ).data_ptr()

    else:

        ptr = next(
            decoder.parameters()
        ).data_ptr()


    print(
        "launch1 thread:",
        threading.current_thread().name,
        "weight ptr:",
        hex(ptr),
    )


    if do_log:
        log_task(
            "stream1",
            "start",
        )


    with torch.cuda.stream(
        stream1
    ):

        run1()


    stream1.synchronize()


    if do_log:
        log_task(
            "stream1",
            "end",
        )


# ============================================================
# One parallel run
# ============================================================

def parallel_once(
    do_log=True,
):

    if do_log:

        log_task(
            "Parallel",
            "start",
        )


    begin = (
        time.perf_counter()
    )


    t0 = threading.Thread(
        target=launch0,
        args=(do_log,),
    )


    t1 = threading.Thread(
        target=launch1,
        args=(do_log,),
    )


    # Start both immediately
    t0.start()

    t1.start()


    t0.join()

    t1.join()


    end = (
        time.perf_counter()
    )


    if do_log:

        log_task(
            "Parallel",
            "end",
        )


    return (
        end - begin
    ) * 1000.0


# ============================================================
# Parallel warmup
# ============================================================

print()
print(
    "========================================"
)

print(
    "Parallel warmup"
)

print(
    "========================================"
)


for _ in range(
    args.warmup_iters
):

    parallel_once(
        do_log=False
    )


torch.cuda.synchronize()


# Clear warmup timeline
open(
    LOG_FILE,
    "w",
).close()


# ============================================================
# Parallel measurement
# ============================================================

print()
print(
    "========================================"
)

print(
    "Parallel"
)

print(
    "========================================"
)


parallel_times = []


for _ in range(
    args.profile_iters
):

    parallel_times.append(
        parallel_once(
            do_log=True
        )
    )


parallel_wall = (
    sum(
        parallel_times
    )
    /
    len(
        parallel_times
    )
)


# ============================================================
# Results
# ============================================================

ideal = max(
    stream0_time,
    stream1_time,
)


estimated_sequential = (
    stream0_time
    +
    stream1_time
)


print()
print(
    "========================================"
)

print(
    "Summary"
)

print(
    "========================================"
)


print(
    "workload:",
    args.workload,
)


print(
    "stream0 SMs:",
    stream0_sms,
)


print(
    "stream1 SMs:",
    stream1_sms,
)


print(
    "stream0 alone:",
    stream0_time,
    "ms",
)


print(
    "stream1 alone:",
    stream1_time,
    "ms",
)


print(
    "stream sum:",
    estimated_sequential,
    "ms",
)


print(
    "measured sequential:",
    sequential_time,
    "ms",
)


print(
    "ideal parallel:",
    ideal,
    "ms",
)


print(
    "actual parallel:",
    parallel_wall,
    "ms",
)


print(
    "parallel / ideal:",
    parallel_wall
    /
    ideal,
)


print(
    "speedup vs measured sequential:",
    sequential_time
    /
    parallel_wall,
)


print(
    "timeline:",
    LOG_FILE,
)


# ============================================================
# Verify SAME model weights
# ============================================================

def verify_shared_weights(
    module,
    name,
):

    print()
    print(
        "========================================"
    )

    print(
        name,
        "weight verification"
    )

    print(
        "========================================"
    )


    params = list(
        module.parameters()
    )


    print(
        "module object:",
        hex(id(module))
    )


    print(
        "number of parameters:",
        len(params)
    )


    print(
        "total parameter memory:",
        sum(
            p.numel() * p.element_size()
            for p in params
        )
        /
        1024
        /
        1024,
        "MB"
    )


    print()


    for i, p in enumerate(params[:10]):

        print(
            f"param {i}:",
            hex(p.data_ptr()),
            "shape:",
            tuple(p.shape),
            "dtype:",
            p.dtype
        )



    print(
        "========================================"
    )



if args.workload == "dit":

    verify_shared_weights(
        model,
        "DiT"
    )

else:

    verify_shared_weights(
        decoder,
        "VAE decoder"
    )