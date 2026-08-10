import argparse
import torch
import greenctx

import threading
import time



# ==================================================
# args
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dit-sms",
    type=int,
    default=72
)

args = parser.parse_args()



torch.manual_seed(0)


device = "cuda:0"


WARMUP = 10
ITER = 10



# ==================================================
# workload
# ==================================================

# --------------------------
# GEMM
# --------------------------

# GEMM
M = 8192
K = 4096
N = 4096


a = torch.randn(
    M,
    K,
    device=device,
    dtype=torch.float16
)


b = torch.randn(
    K,
    N,
    device=device,
    dtype=torch.float16
)



# --------------------------
# Memory
# --------------------------

SIZE = 512 * 1024 * 1024 // 2

x = torch.randn(
    SIZE,
    device=device,
    dtype=torch.float16
)



# ==================================================
# workload functions
# ==================================================

def gemm():

    torch.cuda.nvtx.range_push("GEMM")

    for _ in range(600):
        y = a @ b

    torch.cuda.nvtx.range_pop()


def memory_op():

    torch.cuda.nvtx.range_push("MEMORY")

    y = x

    for _ in range(1600):
        y = y * 1.001

    torch.cuda.nvtx.range_pop()


# ==================================================
# timers
# ==================================================

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



# ==================================================
# strict sequential baseline
# ==================================================

def measure_sequential():

    # warmup

    for _ in range(WARMUP):

        gemm()

        torch.cuda.synchronize()

        memory_op()

        torch.cuda.synchronize()



    torch.cuda.synchronize()


    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    start.record()


    for _ in range(ITER):

        gemm()

        torch.cuda.synchronize()


        memory_op()

        torch.cuda.synchronize()



    end.record()


    end.synchronize()


    return (
        start.elapsed_time(end)
        /
        ITER
    )



# ==================================================
# logger
# ==================================================

LOG_FILE = "task_timeline.txt"


open(
    LOG_FILE,
    "w"
).close()


log_lock = threading.Lock()



def log_task(
        name,
        event):

    t = time.perf_counter() * 1000


    with log_lock:

        with open(
            LOG_FILE,
            "a"
        ) as f:

            f.write(
                f"{name},{event},{t:.3f}\n"
            )



# ==================================================
# threaded parallel
# ==================================================

def measure_parallel():



    # thread warmup
    def warm_worker(
            fn,
            stream):

        with torch.cuda.stream(stream):

            fn()

        stream.synchronize()



    t0 = threading.Thread(
        target=warm_worker,
        args=(
            gemm,
            dit_stream
        )
    )


    t1 = threading.Thread(
        target=warm_worker,
        args=(
            memory_op,
            vae_stream
        )
    )


    t0.start()
    t1.start()

    t0.join()
    t1.join()



    torch.cuda.synchronize()



    times = []



    for _ in range(ITER):


        log_task(
            "Parallel",
            "start"
        )


        begin = time.perf_counter()



        def worker(
                fn,
                stream,
                name):


            log_task(
                name,
                "start"
            )


            with torch.cuda.stream(stream):

                fn()


            stream.synchronize()


            log_task(
                name,
                "end"
            )



        t0 = threading.Thread(
            target=worker,
            args=(
                gemm,
                dit_stream,
                "GEMM"
            )
        )


        t1 = threading.Thread(
            target=worker,
            args=(
                memory_op,
                vae_stream,
                "MEMORY"
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


        times.append(
            (end-begin)
            *
            1000
        )


    return sum(times)/len(times)



# ==================================================
# Green Context
# ==================================================

print(
    "create green context:",
    args.dit_sms,
    "SMs for GEMM"
)



gc = greenctx.GreenContext(
    args.dit_sms,
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
# initialize CUDA context
# ==================================================

gemm()

memory_op()

torch.cuda.synchronize()



# ==================================================
# run
# ==================================================

seq = measure_sequential()


print(
    "sequential:",
    seq,
    "ms"
)



gemm_time = measure_stream(
    gemm,
    dit_stream
)


memory_time = measure_stream(
    memory_op,
    vae_stream
)



print(
    "green GEMM:",
    gemm_time,
    "ms"
)


print(
    "green MEMORY:",
    memory_time,
    "ms"
)



parallel = measure_parallel()



print(
    "parallel:",
    parallel,
    "ms"
)


print(
    "ideal:",
    max(
        gemm_time,
        memory_time
    ),
    "ms"
)


print(
    "speedup:",
    seq / parallel
)


print(
    "overlap efficiency:",
    max(
        gemm_time,
        memory_time
    )
    /
    parallel
)


print(
    "log:",
    LOG_FILE
)