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
)



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

                first_chunk.set(False)


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
):


    # warmup

    with torch.cuda.stream(stream):

        for _ in range(args.warmup_iters):

            fn()



    stream.synchronize()



    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    with torch.cuda.stream(stream):

        start.record(
            stream
        )


        for _ in range(args.profile_iters):

            fn()


        end.record(
            stream
        )


    end.synchronize()


    return (
        start.elapsed_time(end)
        /
        args.profile_iters
    )

# ============================================================
# baseline:
# DiT -> VAE
# default stream
# ============================================================


torch.cuda.synchronize()


start = time.time()



for _ in range(args.profile_iters):

    run_dit_chunk()

    run_vae_chunk()



torch.cuda.synchronize()



baseline = (
    (time.time()-start)
    *
    1000
    /
    args.profile_iters
)



print(
    "baseline sequential:",
    baseline,
    "ms"
)


# ============================================================
# single
# ============================================================


dit_time = benchmark_stream(
    run_dit_chunk,
    dit_stream,
)



vae_time = benchmark_stream(
    run_vae_chunk,
    vae_stream,
)



print(
    "Green DiT:",
    dit_time,
    "ms"
)


print(
    "Green VAE:",
    vae_time,
    "ms"
)

# ============================================================
# parallel: dual CPU thread submission + timeline logging
# ============================================================

import threading


task_log_file = "task_timeline.txt"

# clear old log
open(task_log_file, "w").close()


log_lock = threading.Lock()


def log_task(name, event):

    ts = time.perf_counter() * 1000.0

    with log_lock:
        with open(task_log_file, "a") as f:
            f.write(
                f"{name},{event},{ts:.3f}\n"
            )


# ============================================================
# worker
# ============================================================


def launch_dit():

    log_task(
        "DiT",
        "start"
    )


    with torch.cuda.stream(dit_stream):

        run_dit_chunk()


    # GPU execution finished
    dit_stream.synchronize()


    log_task(
        "DiT",
        "end"
    )

# fake 1: totally empty

# def run_fake_vae():
#     time.sleep(1.5)

#  fake 2 launch empty task
# fake_tensor = torch.empty(
#     1,
#     device="cuda"
# )


# def run_fake_vae():

#     with torch.cuda.stream(vae_stream):

#         fake_tensor.fill_(0)

#  fake big GEMM

# fake_a = torch.randn(
#     4096,
#     4096,
#     device="cuda",
#     dtype=torch.float16
# )

# fake_b = torch.randn(
#     4096,
#     4096,
#     device="cuda",
#     dtype=torch.float16
# )


# def run_fake_vae():

#     with torch.cuda.stream(vae_stream):

#         for _ in range(10):
#             torch.matmul(
#                 fake_a,
#                 fake_b
#             )

# fake memory workload

# FAKE_SIZE = 4 * 1024 * 1024 * 1024 // 2  # fp16 约4GB

# fake_x = torch.randn(
#     FAKE_SIZE,
#     dtype=torch.float16,
#     device="cuda"
# )


# def run_fake_memory():

#     with torch.cuda.stream(vae_stream):

#         for _ in range(200):

#             # streaming read/write
#             fake_x.mul_(1.001)


def launch_vae():

    log_task(
        "VAE",
        "start"
    )


    with torch.cuda.stream(vae_stream):

        run_vae_chunk()
        # run_fake_vae()
        # run_fake_memory()


    # GPU execution finished
    vae_stream.synchronize()


    log_task(
        "VAE",
        "end"
    )



# ============================================================
# benchmark
# ============================================================


torch.cuda.synchronize()


parallel_times = []


for _ in range(args.profile_iters):


    log_task(
        "Parallel",
        "start"
    )


    t0 = time.perf_counter()


    t_dit = threading.Thread(
        target=launch_dit
    )


    t_vae = threading.Thread(
        target=launch_vae
    )

    t_vae.start()
    t_dit.start()


    t_vae.join()
    t_dit.join()


    t1 = time.perf_counter()


    log_task(
        "Parallel",
        "end"
    )


    parallel_times.append(
        (t1 - t0) * 1000
    )



parallel_wall = (
    sum(parallel_times)
    /
    len(parallel_times)
)



# ============================================================
# print
# ============================================================


print(
    "Green parallel wall:",
    parallel_wall,
    "ms"
)


print(
    "Sequential baseline:",
    baseline,
    "ms"
)


print(
    "speedup:",
    baseline / parallel_wall
)


print(
    "timeline saved:",
    task_log_file
)