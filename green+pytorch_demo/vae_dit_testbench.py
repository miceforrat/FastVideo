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
# parallel: dual CPU thread submission
# ============================================================

import threading


torch.cuda.synchronize()


# ============================================================
# helper
# ============================================================

def launch_dit(events):

    dit_start = torch.cuda.Event(
        enable_timing=True
    )

    dit_end = torch.cuda.Event(
        enable_timing=True
    )


    with torch.cuda.stream(dit_stream):

        dit_start.record(
            dit_stream
        )

        run_dit_chunk()

        dit_end.record(
            dit_stream
        )


    events.append(
        (
            dit_start,
            dit_end
        )
    )



def launch_vae(events):

    vae_start = torch.cuda.Event(
        enable_timing=True
    )

    vae_end = torch.cuda.Event(
        enable_timing=True
    )


    with torch.cuda.stream(vae_stream):

        vae_start.record(
            vae_stream
        )

        run_vae_chunk()

        vae_end.record(
            vae_stream
        )


    events.append(
        (
            vae_start,
            vae_end
        )
    )



# ============================================================
# benchmark
# ============================================================


start_wall = time.time()


dit_events = []
vae_events = []


for _ in range(args.profile_iters):


    # ----------------------------------------
    # submit from two CPU threads
    # ----------------------------------------

    dit_events_iter = []
    vae_events_iter = []


    t_dit = threading.Thread(
        target=launch_dit,
        args=(dit_events_iter,)
    )


    t_vae = threading.Thread(
        target=launch_vae,
        args=(vae_events_iter,)
    )


    t_dit.start()
    t_vae.start()


    t_dit.join()
    t_vae.join()


    dit_events.extend(
        dit_events_iter
    )

    vae_events.extend(
        vae_events_iter
    )



# ============================================================
# wait GPU
# ============================================================

dit_stream.synchronize()

vae_stream.synchronize()


end_wall = time.time()



# ============================================================
# E2E wall clock
# ============================================================

parallel_wall = (
    (end_wall-start_wall)
    *
    1000
    /
    args.profile_iters
)



# ============================================================
# GPU kernel timeline time
# ============================================================

dit_times=[]

vae_times=[]


for s,e in dit_events:

    dit_times.append(
        s.elapsed_time(e)
    )


for s,e in vae_events:

    vae_times.append(
        s.elapsed_time(e)
    )



dit_parallel = (
    sum(dit_times)
    /
    len(dit_times)
)


vae_parallel = (
    sum(vae_times)
    /
    len(vae_times)
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
    "Green parallel DiT GPU:",
    dit_parallel,
    "ms"
)


print(
    "Green parallel VAE GPU:",
    vae_parallel,
    "ms"
)


print(
    "ideal overlap:",
    max(
        dit_parallel,
        vae_parallel
    ),
    "ms"
)


print(
    "GPU overlap efficiency:",
    (
        max(
            dit_parallel,
            vae_parallel
        )
        /
        parallel_wall
    ),
)


print(
    "speedup:",
    baseline / parallel_wall
)