import argparse
import threading
import time

import torch
import greenctx

from torch.utils.cpp_extension import load_inline



# ============================================================
# args
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dit-sms",
    type=int,
    default=88
)

parser.add_argument(
    "--blocks-per-sm",
    type=int,
    default=4
)

parser.add_argument(
    "--threads",
    type=int,
    default=256
)

parser.add_argument(
    "--iterations",
    type=int,
    default=50_000_000
)

parser.add_argument(
    "--warmup",
    type=int,
    default=5
)

parser.add_argument(
    "--repeat",
    type=int,
    default=10
)

args = parser.parse_args()



torch.cuda.set_device(0)



# ============================================================
# CUDA extension
# ============================================================

cpp_source = r"""

#include <torch/extension.h>
#include <cstdint>


void launch_register_kernel(
    uint64_t stream_ptr,
    uintptr_t output_ptr,
    int blocks,
    int threads,
    int64_t iterations
);

"""


cuda_source = r"""

#include <cuda.h>
#include <cuda_runtime.h>


__global__
void register_kernel(
    float* output,
    long long iterations
)
{

    float x0 = 1.001f;
    float x1 = 2.001f;
    float x2 = 3.001f;
    float x3 = 4.001f;


    #pragma unroll 1
    for(
        long long i = 0;
        i < iterations;
        i++
    )
    {

        x0 = fmaf(
            x0,
            1.000001f,
            0.000001f
        );


        x1 = fmaf(
            x1,
            1.000002f,
            0.000002f
        );


        x2 = fmaf(
            x2,
            1.000003f,
            0.000003f
        );


        x3 = fmaf(
            x3,
            1.000004f,
            0.000004f
        );

    }


    int idx =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;


    output[idx] =
        x0+x1+x2+x3;
}



void launch_register_kernel(
    uint64_t stream_ptr,
    uintptr_t output_ptr,
    int blocks,
    int threads,
    int64_t iterations
)
{

    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(
            stream_ptr
        );


    float* output =
        reinterpret_cast<float*>(
            output_ptr
        );


    register_kernel<<<
        blocks,
        threads,
        0,
        stream
    >>>(
        output,
        iterations
    );


    cudaError_t err =
        cudaGetLastError();


    if(err != cudaSuccess)
    {
        throw std::runtime_error(
            cudaGetErrorString(err)
        );
    }


    // ==================================================
    // IMPORTANT
    //
    // DO NOT synchronize here.
    //
    // The caller controls synchronization.
    //
    // ==================================================
}

"""


print(
    "compile cuda kernel..."
)


kernel = load_inline(
    name="register_only_kernel_async",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "launch_register_kernel"
    ],
    extra_cflags=[
        "-O3"
    ],
    extra_cuda_cflags=[
        "-O3"
    ],
    verbose=False
)


print(
    "compile done"
)



# ============================================================
# Green Context
# ============================================================

total_sms = (
    torch.cuda
    .get_device_properties(0)
    .multi_processor_count
)


dit_sms = args.dit_sms

vae_sms = (
    total_sms
    -
    dit_sms
)



print(
    "total SM:",
    total_sms
)

print(
    "dit SM:",
    dit_sms
)

print(
    "vae SM:",
    vae_sms
)



gc = greenctx.GreenContext(
    dit_sms,
    0
)



dit_stream_ptr = gc.dit_stream()

vae_stream_ptr = gc.vae_stream()



dit_stream = torch.cuda.ExternalStream(
    dit_stream_ptr,
    device=0
)


vae_stream = torch.cuda.ExternalStream(
    vae_stream_ptr,
    device=0
)



# ============================================================
# workload
# ============================================================

dit_blocks = (
    dit_sms
    *
    args.blocks_per_sm
)


vae_blocks = (
    vae_sms
    *
    args.blocks_per_sm
)



output = torch.empty(
    max(
        dit_blocks,
        vae_blocks
    )
    *
    args.threads,
    device="cuda",
    dtype=torch.float32
)



print(
    "dit blocks:",
    dit_blocks
)

print(
    "vae blocks:",
    vae_blocks
)

print(
    "iterations:",
    args.iterations
)



def dit_work():

    kernel.launch_register_kernel(
        dit_stream_ptr,
        output.data_ptr(),
        dit_blocks,
        args.threads,
        args.iterations
    )



def vae_work():

    kernel.launch_register_kernel(
        vae_stream_ptr,
        output.data_ptr(),
        vae_blocks,
        args.threads,
        args.iterations
    )

# ============================================================
# logger
# ============================================================

LOG_FILE = "task_timeline.txt"


open(
    LOG_FILE,
    "w"
).close()



log_lock = threading.Lock()



def log_task(
    name,
    event
):

    t = (
        time.perf_counter()
        *
        1000.0
    )


    with log_lock:

        with open(
            LOG_FILE,
            "a"
        ) as f:

            f.write(
                f"{name},{event},{t:.3f}\n"
            )



# ============================================================
# CUDA stream timing
# ============================================================

def measure_stream(
    fn,
    stream
):


    # -----------------------------
    # warmup
    # -----------------------------

    for _ in range(args.warmup):

        fn()


    stream.synchronize()



    start = torch.cuda.Event(
        enable_timing=True
    )


    end = torch.cuda.Event(
        enable_timing=True
    )



    with torch.cuda.stream(stream):


        start.record()


        for _ in range(args.repeat):

            fn()


        end.record()



    end.synchronize()



    return (
        start.elapsed_time(end)
        /
        args.repeat
    )



# ============================================================
# sequential timing
#
# IMPORTANT:
#
# launch async
# synchronize manually
#
# ============================================================

def measure_sequential():


    # warmup

    for _ in range(args.warmup):

        dit_work()

        dit_stream.synchronize()


        vae_work()

        vae_stream.synchronize()



    torch.cuda.synchronize()



    start = torch.cuda.Event(
        enable_timing=True
    )


    end = torch.cuda.Event(
        enable_timing=True
    )



    start.record()



    for _ in range(args.repeat):


        dit_work()

        dit_stream.synchronize()



        vae_work()

        vae_stream.synchronize()



    end.record()



    end.synchronize()



    return (
        start.elapsed_time(end)
        /
        args.repeat
    )



# ============================================================
# parallel
#
# Two CPU threads:
#
# thread0:
#       enqueue DiT kernel
#       wait dit stream
#
# thread1:
#       enqueue VAE kernel
#       wait vae stream
#
# ============================================================


def parallel_once():


    log_task(
        "Parallel",
        "start"
    )



    begin = time.perf_counter()



    def worker(
        name,
        fn,
        stream
    ):


        log_task(
            name,
            "start"
        )


        fn()


        #
        # GPU completion
        #

        stream.synchronize()



        log_task(
            name,
            "end"
        )



    t0 = threading.Thread(
        target=worker,
        args=(
            "DiT",
            dit_work,
            dit_stream
        )
    )


    t1 = threading.Thread(
        target=worker,
        args=(
            "VAE",
            vae_work,
            vae_stream
        )
    )



    t0.start()

    t1.start()



    t0.join()

    t1.join()



    end = time.perf_counter()



    log_task(
        "Parallel",
        "end"
    )



    return (
        end - begin
    ) * 1000.0



def measure_parallel():


    # warmup

    for _ in range(args.warmup):

        parallel_once()



    torch.cuda.synchronize()



    values = []



    # clear old log

    open(
        LOG_FILE,
        "w"
    ).close()



    for _ in range(args.repeat):

        values.append(
            parallel_once()
        )



    return (
        sum(values)
        /
        len(values)
    )



# ============================================================
# Run benchmark
# ============================================================


print()
print(
    "================"
)

print(
    "DiT stream"
)

print(
    "================"
)



dit_time = measure_stream(
    dit_work,
    dit_stream
)



print(
    "DiT stream:",
    dit_time,
    "ms"
)



print()
print(
    "================"
)

print(
    "VAE stream"
)

print(
    "================"
)



vae_time = measure_stream(
    vae_work,
    vae_stream
)



print(
    "VAE stream:",
    vae_time,
    "ms"
)



print()
print(
    "================"
)

print(
    "Sequential"
)

print(
    "================"
)



seq_time = measure_sequential()



print(
    "Sequential:",
    seq_time,
    "ms"
)



print()
print(
    "================"
)

print(
    "Parallel"
)

print(
    "================"
)



parallel_time = measure_parallel()



print(
    "Parallel:",
    parallel_time,
    "ms"
)



# ============================================================
# Summary
# ============================================================


ideal = max(
    dit_time,
    vae_time
)



print()
print(
    "================"
)

print(
    "Summary"
)

print(
    "================"
)



print(
    "DiT:",
    dit_time,
    "ms"
)


print(
    "VAE:",
    vae_time,
    "ms"
)


print(
    "Sequential:",
    seq_time,
    "ms"
)


print(
    "Parallel:",
    parallel_time,
    "ms"
)


print(
    "Ideal:",
    ideal,
    "ms"
)


print(
    "Parallel / ideal:",
    parallel_time / ideal
)


print(
    "Speedup:",
    seq_time / parallel_time
)


print(
    "timeline:",
    LOG_FILE
)