import os
import argparse
import threading
import time

import torch
import greenctx
from torch.utils.cpp_extension import load_inline


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
        "--fake-kernel-ms",
        type=float,
        default=0.25,
        help="Target duration of one fake-VAE kernel.",
    )


    parser.add_argument(
        "--fake-kernels",
        type=int,
        default=6400,
        help="Number of kernels used for the fake-only baseline.",
    )


    parser.add_argument(
        "--fake-blocks-per-sm",
        type=int,
        default=1,
    )


    parser.add_argument(
        "--fake-threads",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--fake-read-mib",
        type=int,
        default=128,
        help="Size of the read-only global-memory working set in MiB.",
    )

    parser.add_argument(
        "--fake-launch-batch",
        type=int,
        default=32,
        help="Kernels launched before waiting for the fake-VAE stream.",
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
# Fake VAE: concurrently submitted, global-memory read-heavy kernels
# ============================================================


cpp_source = r"""
#include <torch/extension.h>
#include <cstdint>

void launch_fake_vae_kernel(
    uint64_t stream_ptr,
    uintptr_t input_ptr,
    int64_t input_elements,
    uintptr_t output_ptr,
    int blocks,
    int threads,
    int64_t target_cycles
);
"""


cuda_source = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void fake_vae_register_spin(
    const float* input,
    unsigned long long input_elements,
    float* output,
    unsigned long long target_cycles
) {
    const unsigned long long start = clock64();
    const unsigned long long linear_tid =
        blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long grid_threads = gridDim.x * blockDim.x;
    unsigned long long index = linear_tid % input_elements;
    float x = 0.0f;

    while (clock64() - start < target_cycles) {
        x += __ldg(input + index);
        index += grid_threads;
        if (index >= input_elements) {
            index -= input_elements;
        }
        asm volatile("");
    }

    output[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

void launch_fake_vae_kernel(
    uint64_t stream_ptr,
    uintptr_t input_ptr,
    int64_t input_elements,
    uintptr_t output_ptr,
    int blocks,
    int threads,
    int64_t target_cycles
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const float* input = reinterpret_cast<const float*>(input_ptr);
    float* output = reinterpret_cast<float*>(output_ptr);

    fake_vae_register_spin<<<blocks, threads, 0, stream>>>(
        input,
        static_cast<unsigned long long>(input_elements),
        output,
        static_cast<unsigned long long>(target_cycles)
    );

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
"""


print("compile fake-VAE CUDA kernel")

fake_vae_extension = load_inline(
    name="fastvideo_fake_vae_read_heavy",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_fake_vae_kernel"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


total_sms = torch.cuda.get_device_properties(0).multi_processor_count
fake_vae_sms = total_sms - args.dit_sms

if fake_vae_sms <= 0:
    raise ValueError(
        f"--dit-sms must be smaller than total SM count {total_sms}"
    )
if args.fake_kernel_ms <= 0:
    raise ValueError("--fake-kernel-ms must be positive")
if args.fake_kernels <= 0:
    raise ValueError("--fake-kernels must be positive")
if args.fake_blocks_per_sm <= 0:
    raise ValueError("--fake-blocks-per-sm must be positive")
if args.fake_threads <= 0 or args.fake_threads > 1024:
    raise ValueError("--fake-threads must be in [1, 1024]")
if args.fake_read_mib <= 0:
    raise ValueError("--fake-read-mib must be positive")
if args.fake_launch_batch <= 0:
    raise ValueError("--fake-launch-batch must be positive")


fake_vae_blocks = fake_vae_sms * args.fake_blocks_per_sm
fake_vae_input_elements = args.fake_read_mib * 1024 * 1024 // 4
fake_vae_input = torch.rand(
    fake_vae_input_elements,
    device=DEVICE,
    dtype=torch.float32,
)
fake_vae_output = torch.empty(
    fake_vae_blocks * args.fake_threads,
    device=DEVICE,
    dtype=torch.float32,
)

# cudaDeviceProp.clock_rate is in kHz. This is only an initial estimate;
# CUDA Events below calibrate it against the requested wall-clock duration.
clock_rate_khz = torch.cuda.get_device_properties(0).clock_rate
fake_vae_cycles = max(
    1,
    int(args.fake_kernel_ms * clock_rate_khz),
)


print("total SM:", total_sms)
print("DiT SM request:", args.dit_sms)
print("fake-VAE SM remainder:", fake_vae_sms)
print("fake-VAE blocks:", fake_vae_blocks)
print("fake-VAE threads:", args.fake_threads)
print("fake-VAE read working set:", args.fake_read_mib, "MiB")
print("fake-VAE launch batch:", args.fake_launch_batch)
print("attention backend env:", os.getenv("FASTVIDEO_ATTENTION_BACKEND"))


def launch_one_fake_vae_kernel(target_cycles):
    fake_vae_extension.launch_fake_vae_kernel(
        gc.vae_stream(),
        fake_vae_input.data_ptr(),
        fake_vae_input.numel(),
        fake_vae_output.data_ptr(),
        fake_vae_blocks,
        args.fake_threads,
        target_cycles,
    )


def measure_one_fake_kernel(target_cycles):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(vae_stream):
        start.record(vae_stream)
        launch_one_fake_vae_kernel(target_cycles)
        end.record(vae_stream)

    end.synchronize()
    return start.elapsed_time(end)


# Calibrate three times because clock64 advances at the active SM clock and
# the device may change clocks after the first launch.
for calibration_step in range(3):
    measured_ms = measure_one_fake_kernel(fake_vae_cycles)
    fake_vae_cycles = max(
        1,
        int(
            fake_vae_cycles
            * args.fake_kernel_ms
            / measured_ms
        ),
    )
    print(
        "fake-VAE calibration",
        calibration_step,
        "measured_ms=",
        measured_ms,
        "next_cycles=",
        fake_vae_cycles,
    )

calibrated_fake_kernel_ms = measure_one_fake_kernel(fake_vae_cycles)
print("calibrated fake kernel:", calibrated_fake_kernel_ms, "ms")
print(
    "target fake chunk:",
    calibrated_fake_kernel_ms * args.fake_kernels,
    "ms",
)


def run_fake_vae_chunk():
    """Submit a fixed read-heavy chunk for the fake-only baseline."""
    torch.cuda.nvtx.range_push("FakeVAE_read_heavy_chunk")

    for _ in range(args.fake_kernels):
        launch_one_fake_vae_kernel(fake_vae_cycles)

    torch.cuda.nvtx.range_pop()


# ============================================================
# Timing helpers
# ============================================================


def benchmark_stream(fn, stream, warmup_iters, profile_iters):
    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            fn()
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(profile_iters):
            fn()
        end.record(stream)

    end.synchronize()
    return start.elapsed_time(end) / profile_iters


print("benchmark DiT alone")
dit_time = benchmark_stream(
    run_dit_chunk,
    dit_stream,
    args.warmup_iters,
    args.profile_iters,
)

# Calibration already warmed up the fake kernel. Avoid adding several long
# fake chunks solely for warmup.
print("benchmark fake VAE alone")
fake_vae_time = benchmark_stream(
    run_fake_vae_chunk,
    vae_stream,
    0,
    args.profile_iters,
)


print("Green DiT alone:", dit_time, "ms")
print("Green fake VAE alone:", fake_vae_time, "ms")


# ============================================================
# Colocation: concurrently submit read-heavy fake VAE and DiT
# ============================================================


parallel_wall_times = []
parallel_dit_times = []
parallel_fake_vae_times = []
fake_launch_counts = []
fake_launch_cpu_times = []


torch.cuda.synchronize()

for iteration in range(args.profile_iters):
    dit_start = torch.cuda.Event(enable_timing=True)
    dit_end = torch.cuda.Event(enable_timing=True)
    stop_fake = threading.Event()
    fake_ready = threading.Event()
    fake_stats = {}

    wall_start = time.perf_counter()

    def fake_submitter():
        torch.cuda.set_device(0)
        fake_start = torch.cuda.Event(enable_timing=True)
        fake_end = torch.cuda.Event(enable_timing=True)
        launch_count = 0
        submit_start = time.perf_counter()

        torch.cuda.nvtx.range_push("FakeVAE_concurrent_read_submit")
        with torch.cuda.stream(vae_stream):
            fake_start.record(vae_stream)
            while not stop_fake.is_set():
                for _ in range(args.fake_launch_batch):
                    launch_one_fake_vae_kernel(fake_vae_cycles)
                    launch_count += 1
                fake_ready.set()

                # Bound outstanding work. This deliberately resumes host
                # submission after each short batch while DiT is active.
                vae_stream.synchronize()
            fake_end.record(vae_stream)
        fake_end.synchronize()
        torch.cuda.nvtx.range_pop()

        fake_stats["launches"] = launch_count
        fake_stats["cpu_ms"] = (
            time.perf_counter() - submit_start
        ) * 1000.0
        fake_stats["gpu_ms"] = fake_start.elapsed_time(fake_end)

    fake_thread = threading.Thread(
        target=fake_submitter,
        name="fake-vae-read-submitter",
    )
    fake_thread.start()
    if not fake_ready.wait(timeout=30.0):
        stop_fake.set()
        fake_thread.join()
        raise RuntimeError("fake-VAE submitter did not become ready")

    torch.cuda.nvtx.range_push("DiT_with_concurrent_read_submit")
    with torch.cuda.stream(dit_stream):
        dit_start.record(dit_stream)
        run_dit_chunk()
        dit_end.record(dit_stream)
    torch.cuda.nvtx.range_pop()

    dit_end.synchronize()
    stop_fake.set()
    fake_thread.join()

    wall_end = time.perf_counter()

    parallel_wall_times.append(
        (wall_end - wall_start) * 1000.0
    )
    parallel_dit_times.append(
        dit_start.elapsed_time(dit_end)
    )
    parallel_fake_vae_times.append(fake_stats["gpu_ms"])
    fake_launch_counts.append(fake_stats["launches"])
    fake_launch_cpu_times.append(fake_stats["cpu_ms"])

    print(
        "iteration",
        iteration,
        "fake_launches=",
        fake_launch_counts[-1],
        "fake_submit_cpu_ms=",
        fake_launch_cpu_times[-1],
        "dit_ms=",
        parallel_dit_times[-1],
        "fake_vae_ms=",
        parallel_fake_vae_times[-1],
        "wall_ms=",
        parallel_wall_times[-1],
    )


parallel_wall = sum(parallel_wall_times) / len(parallel_wall_times)
parallel_dit = sum(parallel_dit_times) / len(parallel_dit_times)
parallel_fake_vae = (
    sum(parallel_fake_vae_times)
    / len(parallel_fake_vae_times)
)
fake_launch_count = sum(fake_launch_counts) / len(fake_launch_counts)
fake_launch_cpu = (
    sum(fake_launch_cpu_times) / len(fake_launch_cpu_times)
)


print("=" * 72)
print("fake-VAE concurrent read-heavy launch experiment")
print("fake launches:", fake_launch_count)
print("fake submit CPU:", fake_launch_cpu, "ms")
print("DiT alone:", dit_time, "ms")
print("DiT colocated:", parallel_dit, "ms")
print("DiT slowdown:", parallel_dit / dit_time)
print("fake VAE alone:", fake_vae_time, "ms")
print("fake VAE colocated:", parallel_fake_vae, "ms")
print("parallel wall:", parallel_wall, "ms")
print(
    "overlap lower bound:",
    parallel_dit + parallel_fake_vae - parallel_wall,
    "ms",
)
print("=" * 72)
