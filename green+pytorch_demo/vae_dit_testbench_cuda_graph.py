import os
import argparse
import time

import torch
import greenctx


from torch.distributed.fsdp import MixedPrecisionPolicy


from fastvideo.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)


from fastvideo.utils import set_mixed_precision_policy


from fastvideo.models.dits.causal_wanvideo import (
    CausalWanTransformer3DModel,
)


from fastvideo.configs.models.dits import (
    WanVideoConfig,
)


from fastvideo.forward_context import (
    set_forward_context,
)


from fastvideo.models.vaes.wanvae import (
    WanDecoder3d,
    WanCausalConv3d,
    forward_context,
    first_chunk,
    feat_idx,
)

# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dit-sms",
        type=int,
        default=80,
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


    parser.add_argument(
        "--ignore-sm-coscheduling",
        action="store_true",
        help=(
            "Pass CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING "
            "when partitioning SM resources"
        ),
    )




    parser.add_argument(
        "--vae-first-chunk",
        action="store_true",
        help=(
            "Benchmark and capture the VAE cold first-chunk path. The first "
            "latent uses first_chunk=True and the feature cache is reset "
            "before each eager sample."
        ),
    )

    parser.add_argument(
        "--sequential-only",
        action="store_true",
        help=(
            "Run only the full-device sequential DiT-then-VAE baseline "
            "and exit before Green Context and CUDA Graph benchmarks."
        ),
    )

    return parser.parse_args()



args = parse_args()



# ============================================================
# Runtime
# ============================================================

DEVICE="cuda:0"

DTYPE=torch.bfloat16


torch.cuda.set_device(0)


torch.manual_seed(0)

torch.cuda.manual_seed_all(0)



# ============================================================
# Your Green Context
# ============================================================

print("create green context")


gc = greenctx.GreenContext(
    args.dit_sms,
    0,
    args.ignore_sm_coscheduling,
)


dit_stream = torch.cuda.ExternalStream(
    gc.dit_stream(),
    device=0,
)


vae_stream = torch.cuda.ExternalStream(
    gc.vae_stream(),
    device=0,
)



print(
    "dit stream:",
    hex(gc.dit_stream())
)


print(
    "vae stream:",
    hex(gc.vae_stream())
)



# ============================================================
# Distributed
# ============================================================

os.environ["MASTER_ADDR"]="127.0.0.1"

os.environ["MASTER_PORT"]="29501"



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
# Build DiT
# ============================================================

print("init DiT")



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
        [1,2,2],

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


config.patch_size = [1,2,2]


config.text_dim = 4096

config.freq_dim = 256


config.cross_attn_norm=True

config.qk_norm="rms_norm_across_heads"

config.eps=1e-6


config.num_frames_per_block=3

config.sliding_window_num_frames=21



model = CausalWanTransformer3DModel(
    config=config,
    hf_config=hf_config,
)


model = model.to(
    DEVICE,
    dtype=DTYPE
)


model.eval()



# ============================================================
# DiT Input
# ============================================================

B=1


hidden_states=torch.randn(
    B,
    16,
    3,
    60,
    104,
    device=DEVICE,
    dtype=DTYPE,
)



encoder_hidden_states=torch.randn(
    B,
    512,
    4096,
    device=DEVICE,
    dtype=torch.float32,
)



chunk_size=3


start_frame = (
    args.chunk_idx *
    chunk_size
)



frame_seq_length = (
    60//2
) * (
    104//2
)



current_start = (
    start_frame *
    frame_seq_length
)



print(
    "chunk:",
    args.chunk_idx,
    "current_start:",
    current_start
)



# ============================================================
# KV cache
# ============================================================

num_layers=30

num_heads=12

head_dim=128



kv_tokens = (
    frame_seq_length *
    21
)



kv_cache=[]


for _ in range(num_layers):

    kv_cache.append({

        "k":torch.zeros(
            B,
            kv_tokens,
            num_heads,
            head_dim,
            device=DEVICE,
            dtype=DTYPE,
        ),

        "v":torch.zeros(
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




crossattn_cache=[]


for _ in range(num_layers):

    crossattn_cache.append({

        "k":torch.zeros(
            B,
            512,
            num_heads,
            head_dim,
            device=DEVICE,
            dtype=DTYPE,
        ),

        "v":torch.zeros(
            B,
            512,
            num_heads,
            head_dim,
            device=DEVICE,
            dtype=DTYPE,
        ),

        "is_init":True,

    })



# ============================================================
# timesteps
# ============================================================

timesteps=torch.arange(
    args.num_denoise_steps,
    device=DEVICE,
    dtype=torch.long,
)


# ============================================================
# One DiT forward
# ============================================================

def dit_forward(
    latent,
    timestep,
    ctx_step,
):

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ), set_forward_context(
            current_timestep=ctx_step,
            attn_metadata=None,
            forward_batch=None,
        ):


            out = model._forward_inference(

                hidden_states=latent,

                encoder_hidden_states=
                    encoder_hidden_states,

                timestep=
                    timestep.reshape(1,1),

                encoder_hidden_states_image=None,


                kv_cache=kv_cache,

                crossattn_cache=
                    crossattn_cache,


                current_start=current_start,

                cache_start=0,

                start_frame=start_frame,
            )


    return out



# ============================================================
# One causal chunk
#
# 4 denoise + 1 KV update
# ============================================================

def run_dit_chunk():


    torch.cuda.nvtx.range_push(
        "DiT_chunk"
    )


    latent = hidden_states.clone()


    torch.cuda.nvtx.range_push(
        "DiT_denoise"
    )


    for i,t in enumerate(timesteps):

        pred = dit_forward(
            latent,
            t,
            i,
        )

        latent = latent - 0.1 * pred


    torch.cuda.nvtx.range_pop()



    torch.cuda.nvtx.range_push(
        "DiT_KV_update"
    )


    _ = dit_forward(
        latent,
        torch.tensor(
            0,
            device=DEVICE,
            dtype=torch.long,
        ),
        0,
    )


    torch.cuda.nvtx.range_pop()


    torch.cuda.nvtx.range_pop()


    return latent


# ============================================================
# Build VAE
# ============================================================


print("init VAE")


post_quant_conv = WanCausalConv3d(
    16,
    16,
    1,
)


decoder = WanDecoder3d(
    dim=96,
    z_dim=16,
    dim_mult=[1,2,4,4],
    num_res_blocks=2,
    attn_scales=(),
    temperal_upsample=[
        True,
        True,
        False
    ],
    dropout=0.0,
    out_channels=3,
    is_residual=False,
)


post_quant_conv = (
    post_quant_conv
    .cuda()
    .eval()
)


decoder = (
    decoder
    .cuda()
    .eval()
)



# ============================================================
# VAE feature cache
# ============================================================


def count_causal_conv(model):

    return sum(
        1
        for m in model.modules()
        if isinstance(
            m,
            WanCausalConv3d
        )
    )



num_cache = count_causal_conv(
    decoder
)


print(
    "decoder causal conv:",
    num_cache
)



feat_cache_map = [
    None
    for _ in range(num_cache)
]



# ============================================================
# VAE input
# ============================================================


T = args.num_latents


z = torch.randn(
    1,
    16,
    T,
    60,
    104,
    device=DEVICE,
    dtype=torch.float32,
)



# ============================================================
# initialize VAE cache
# ============================================================


print(
    "initialize VAE cache"
)



dummy = torch.randn(
    1,
    16,
    1,
    60,
    104,
    device=DEVICE,
    dtype=torch.float32,
)



dummy_x = post_quant_conv(
    dummy
)



if not args.vae_first_chunk:

    with torch.no_grad():

        with forward_context(
            feat_cache_arg=feat_cache_map,
            feat_idx_arg=0,
        ):

            feat_idx.set(0)

            first_chunk.set(False)


            _ = decoder(
                dummy_x
            )



torch.cuda.synchronize()



print(
    "VAE cache ready"
    if not args.vae_first_chunk
    else "VAE cold first-chunk mode: cache starts empty"
)


def reset_vae_feature_cache():

    for cache_idx in range(len(feat_cache_map)):

        feat_cache_map[cache_idx] = None



# ============================================================
# One VAE chunk
#
# post_quant_conv
# +
# 3 latent frame decode
# ============================================================


def run_vae_chunk():


    torch.cuda.nvtx.range_push(
        "VAE_chunk"
    )


    with torch.no_grad():


        torch.cuda.nvtx.range_push(
            "VAE_post_quant_conv"
        )


        x = post_quant_conv(z)


        torch.cuda.nvtx.range_pop()



        outputs=[]


        torch.cuda.nvtx.range_push(
            "VAE_decode"
        )


        with forward_context(
            feat_cache_arg=feat_cache_map,
            feat_idx_arg=0,
        ):


            for i in range(T):


                feat_idx.set(0)

                first_chunk.set(
                    args.vae_first_chunk
                    and i == 0
                )


                out_i = decoder(
                    x[:,:,i:i+1,:,:]
                )


                outputs.append(
                    out_i
                )


        torch.cuda.nvtx.range_pop()



        out = torch.cat(
            outputs,
            dim=2
        )


    torch.cuda.nvtx.range_pop()


    return out


# ============================================================
# Benchmark helper
# ============================================================


def benchmark_stream(
    fn,
    stream,
    warmup_iters,
    profile_iters,
    profile_nvtx_range=None,
    prepare_fn=None,
):
    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            if prepare_fn is not None:
                prepare_fn()
            fn()

    stream.synchronize()
    if profile_nvtx_range is not None:
        torch.cuda.nvtx.range_push(profile_nvtx_range)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(profile_iters):
            if prepare_fn is not None:
                prepare_fn()
            fn()
        end.record(stream)

    end.synchronize()
    if profile_nvtx_range is not None:
        torch.cuda.nvtx.range_pop()
    return start.elapsed_time(end) / profile_iters


# ============================================================
# Full-device sequential baseline
#
# The ordinary CUDA default stream is not one of the two Green Context
# streams. DiT and VAE are submitted to this same stream, so they execute
# sequentially while using the full set of SMs visible to the process.
# ============================================================


def benchmark_full_sm_sequential(
    stream,
    warmup_iters,
    profile_iters,
):
    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            if args.vae_first_chunk:
                reset_vae_feature_cache()
            run_dit_chunk()
            run_vae_chunk()

    stream.synchronize()

    iteration_events = []
    torch.cuda.nvtx.range_push("Full_SM_sequential")

    with torch.cuda.stream(stream):
        for _ in range(profile_iters):
            if args.vae_first_chunk:
                reset_vae_feature_cache()

            iteration_start = torch.cuda.Event(enable_timing=True)
            dit_end = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)

            iteration_start.record(stream)

            torch.cuda.nvtx.range_push("Full_SM_DiT")
            run_dit_chunk()
            torch.cuda.nvtx.range_pop()
            dit_end.record(stream)

            torch.cuda.nvtx.range_push("Full_SM_VAE")
            run_vae_chunk()
            torch.cuda.nvtx.range_pop()
            vae_end.record(stream)

            iteration_events.append(
                (iteration_start, dit_end, vae_end)
            )

    iteration_events[-1][2].synchronize()
    torch.cuda.nvtx.range_pop()

    dit_times = []
    vae_times = []
    total_times = []

    for iteration_start, dit_end, vae_end in iteration_events:
        dit_times.append(iteration_start.elapsed_time(dit_end))
        vae_times.append(dit_end.elapsed_time(vae_end))
        total_times.append(iteration_start.elapsed_time(vae_end))

    return (
        sum(dit_times) / profile_iters,
        sum(vae_times) / profile_iters,
        sum(total_times) / profile_iters,
    )


# ============================================================
# Full-SM sequential baseline
# ============================================================


full_sm_stream = torch.cuda.default_stream(device=0)

print("benchmark full-SM sequential DiT then VAE")
(
    full_sm_dit_time,
    full_sm_vae_time,
    full_sm_sequential_time,
) = benchmark_full_sm_sequential(
    full_sm_stream,
    args.warmup_iters,
    args.profile_iters,
)

print("Full-SM sequential DiT:", full_sm_dit_time, "ms")
print("Full-SM sequential VAE:", full_sm_vae_time, "ms")
print("Full-SM sequential total:", full_sm_sequential_time, "ms")

if args.sequential_only:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    raise SystemExit(0)


# ============================================================
# Green Context eager baselines
# ============================================================


print("benchmark eager DiT alone")
dit_time = benchmark_stream(
    run_dit_chunk,
    dit_stream,
    args.warmup_iters,
    args.profile_iters,
    profile_nvtx_range="DiT_eager_alone",
)


print("benchmark eager VAE alone")
vae_eager_time = benchmark_stream(
    run_vae_chunk,
    vae_stream,
    args.warmup_iters,
    args.profile_iters,
    prepare_fn=(
        reset_vae_feature_cache
        if args.vae_first_chunk
        else None
    ),
)


print("Green DiT eager alone:", dit_time, "ms")
print("Green VAE eager alone:", vae_eager_time, "ms")


# ============================================================
# Capture VAE on its Green Context stream
# ============================================================


print("warm up VAE before CUDA Graph capture")

with torch.cuda.stream(vae_stream):
    for _ in range(args.warmup_iters):
        if args.vae_first_chunk:
            reset_vae_feature_cache()
        run_vae_chunk()

vae_stream.synchronize()
torch.cuda.synchronize()


print("capture VAE CUDA Graph")

if args.vae_first_chunk:
    reset_vae_feature_cache()

vae_graph = torch.cuda.CUDAGraph()
captured_vae_output = None

# The captured output is kept alive because it resides in the graph-private
# memory pool. Replay mutates the same graph-owned output and cache buffers.
with torch.cuda.graph(
    vae_graph,
    stream=vae_stream,
    capture_error_mode="global",
):
    captured_vae_output = run_vae_chunk()

vae_stream.synchronize()

if captured_vae_output is None:
    raise RuntimeError("VAE CUDA Graph capture did not produce an output")


print("VAE CUDA Graph capture complete")


def replay_vae_graph():
    torch.cuda.nvtx.range_push("VAE_graph_replay_host")
    vae_graph.replay()
    torch.cuda.nvtx.range_pop()


# Warm up graph replay separately. This avoids including first-replay driver
# setup in either the graph-alone or colocation measurements.
with torch.cuda.stream(vae_stream):
    replay_vae_graph()
vae_stream.synchronize()


torch.cuda.nvtx.range_push("VAE_graph_alone")

vae_graph_time = benchmark_stream(
    replay_vae_graph,
    vae_stream,
    0,
    args.profile_iters,
)

torch.cuda.nvtx.range_pop()

print("Green VAE graph alone:", vae_graph_time, "ms")


# ============================================================
# Colocation protocol
#
# 1. Replay returns after one graph launch API call.
# 2. VAE graph kernels remain queued/running on the VAE Green Context.
# 3. The same CPU thread immediately submits the DiT workload.
#
# There are no per-op VAE/cuDNN host calls concurrent with DiT submission.
# ============================================================


parallel_wall_times = []
parallel_dit_times = []
parallel_vae_times = []
graph_replay_cpu_times = []


torch.cuda.synchronize()

for iteration in range(args.profile_iters):
    vae_start = torch.cuda.Event(enable_timing=True)
    vae_end = torch.cuda.Event(enable_timing=True)
    dit_start = torch.cuda.Event(enable_timing=True)
    dit_end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()

    torch.cuda.nvtx.range_push("Colocated_iteration")
    with torch.cuda.stream(vae_stream):
        vae_start.record(vae_stream)
        replay_cpu_start = time.perf_counter()
        replay_vae_graph()
        replay_cpu_end = time.perf_counter()
        vae_end.record(vae_stream)

    torch.cuda.nvtx.range_push("DiT_after_VAE_graph_replay")
    with torch.cuda.stream(dit_stream):
        dit_start.record(dit_stream)
        run_dit_chunk()
        dit_end.record(dit_stream)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()

    dit_end.synchronize()
    vae_end.synchronize()

    wall_end = time.perf_counter()

    graph_replay_cpu_times.append(
        (replay_cpu_end - replay_cpu_start) * 1000.0
    )
    parallel_wall_times.append(
        (wall_end - wall_start) * 1000.0
    )
    parallel_dit_times.append(
        dit_start.elapsed_time(dit_end)
    )
    parallel_vae_times.append(
        vae_start.elapsed_time(vae_end)
    )

    print(
        "iteration",
        iteration,
        "graph_replay_cpu_ms=",
        graph_replay_cpu_times[-1],
        "dit_ms=",
        parallel_dit_times[-1],
        "vae_graph_ms=",
        parallel_vae_times[-1],
        "wall_ms=",
        parallel_wall_times[-1],
    )


parallel_wall = sum(parallel_wall_times) / len(parallel_wall_times)
parallel_dit = sum(parallel_dit_times) / len(parallel_dit_times)
parallel_vae = sum(parallel_vae_times) / len(parallel_vae_times)
graph_replay_cpu = (
    sum(graph_replay_cpu_times)
    / len(graph_replay_cpu_times)
)
overlap_lower_bound = (
    parallel_dit
    + parallel_vae
    - parallel_wall
)


print("=" * 72)
print("VAE CUDA Graph colocation experiment")
print("attention backend env:", os.getenv("FASTVIDEO_ATTENTION_BACKEND"))
print("Full-SM sequential DiT:", full_sm_dit_time, "ms")
print("Full-SM sequential VAE:", full_sm_vae_time, "ms")
print("Full-SM sequential total:", full_sm_sequential_time, "ms")
print("VAE graph replay CPU:", graph_replay_cpu, "ms")
print("DiT eager alone:", dit_time, "ms")
print("DiT with VAE graph:", parallel_dit, "ms")
print("DiT slowdown ratio:", parallel_dit / dit_time)
print("VAE eager alone:", vae_eager_time, "ms")
print("VAE graph alone:", vae_graph_time, "ms")
print("VAE graph colocated:", parallel_vae, "ms")
print("parallel wall:", parallel_wall, "ms")
print("overlap lower bound:", overlap_lower_bound, "ms")
print("=" * 72)
