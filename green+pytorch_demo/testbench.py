import torch
import greenctx


torch.manual_seed(0)


device = "cuda:0"


WARMUP = 20
ITER = 10



# ==================================================
# workload
# ==================================================

# DiT-like GEMM

M = 1024
K = 4096
N = 4096


a = torch.randn(
    M,
    K,
    device=device,
    dtype=torch.float16,
)


b = torch.randn(
    K,
    N,
    device=device,
    dtype=torch.float16,
)



# VAE-like memory workload

SIZE = 256 * 1024 * 1024 // 2


x = torch.randn(
    SIZE,
    device=device,
    dtype=torch.float16,
)



# ==================================================
# functions
# ==================================================

def gemm():

    torch.cuda.nvtx.range_push(
        "GEMM"
    )

    y = a @ b

    torch.cuda.nvtx.range_pop()



def memory_op():

    torch.cuda.nvtx.range_push(
        "MEMORY"
    )

    y = x * 1.001

    torch.cuda.nvtx.range_pop()



# ==================================================
# timer
# ==================================================

def measure_default(fn):

    for _ in range(WARMUP):
        fn()


    torch.cuda.synchronize()


    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    start.record()


    for _ in range(ITER):
        fn()


    end.record()


    end.synchronize()


    return (
        start.elapsed_time(end)
        /
        ITER
    )



def measure_stream(
        fn,
        stream):


    with torch.cuda.stream(stream):

        for _ in range(WARMUP):
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


        for _ in range(ITER):

            fn()


        end.record()


    end.synchronize()


    return (
        start.elapsed_time(end)
        /
        ITER
    )



def measure_parallel(
        fn0,
        s0,
        fn1,
        s1):


    torch.cuda.synchronize()


    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    sync_event = torch.cuda.Event()



    # =====================================
    # interleave submit:
    #
    # G M G M G M ...
    #
    # =====================================

    with torch.cuda.stream(s0):

        start.record(
            s0
        )



    for _ in range(ITER):


        with torch.cuda.stream(s0):

            fn0()



        with torch.cuda.stream(s1):

            fn1()



    # =====================================
    # wait both streams
    # =====================================

    with torch.cuda.stream(s1):

        sync_event.record(
            s1
        )



    with torch.cuda.stream(s0):

        s0.wait_event(
            sync_event
        )


        end.record(
            s0
        )


    end.synchronize()


    return (
        start.elapsed_time(end)
        /
        ITER
    )



# ==================================================
# baseline
# ==================================================

baseline = measure_default(
    lambda:
    (
        gemm(),
        memory_op()
    )
)


print(
    "baseline sequential:",
    baseline,
    "ms"
)



# ==================================================
# Green context
# ==================================================

gc = greenctx.GreenContext(
    72,
    0
)



dit_stream = torch.cuda.ExternalStream(
    gc.dit_stream(),
    device=0
)


vae_stream = torch.cuda.ExternalStream(
    gc.vae_stream(),
    device=0
)



print(
    "DIT stream:",
    hex(gc.dit_stream())
)


print(
    "VAE stream:",
    hex(gc.vae_stream())
)



# ==================================================
# single workload
# ==================================================

dit = measure_stream(
    gemm,
    dit_stream
)


vae = measure_stream(
    memory_op,
    vae_stream
)



print(
    "green GEMM:",
    dit,
    "ms"
)


print(
    "green memory:",
    vae,
    "ms"
)



# ==================================================
# parallel
# ==================================================

parallel = measure_parallel(
    gemm,
    dit_stream,
    memory_op,
    vae_stream
)



print(
    "green parallel:",
    parallel,
    "ms"
)


print(
    "speedup:",
    baseline / parallel
)


#   nsys profile \
#     --trace=cuda,nvtx,osrt,cudnn,cublas \
#     --sample=none \
#     --cpuctxsw=none \
#     -o green+pytorch_demo/fake_vae_dit \
#     /opt/venv/bin/python \
#     green+pytorch_demo/fake_vae_dit_testbench.py \
#     --dit-sms 112 \
#     --fake-kernel-ms 25 \
#     --fake-kernels 80 \
#     --fake-blocks-per-sm 1 \
#     --fake-threads 256 \
#     --warmup-iters 3 \
#     --profile-iters 3