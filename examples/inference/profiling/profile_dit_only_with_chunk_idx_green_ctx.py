import os
import argparse
import torch

from torch.distributed.fsdp import MixedPrecisionPolicy

from fastvideo.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

from fastvideo.utils import set_mixed_precision_policy

from fastvideo.models.dits.causal_wanvideo import (
    CausalWanTransformer3DModel,
)

from fastvideo.configs.models.dits import WanVideoConfig

from fastvideo.forward_context import set_forward_context

from torch.cuda.green_contexts import GreenContext



# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Profile Causal Wan DiT with GreenContext"
    )

    parser.add_argument(
        "--num-sms",
        type=int,
        default=170,
        help="GreenContext SM number",
    )


    # chunk idx 0-6均可
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=5,
        help="AR chunk index",
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
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

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29501"


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.cuda.set_device(0)



# ============================================================
# Green Context
# ============================================================

green_ctx = GreenContext.create(
    num_sms=args.num_sms
)

green_stream = green_ctx.Stream()



# ============================================================
# Distributed init
# ============================================================

os.environ.setdefault(
    "MASTER_ADDR",
    MASTER_ADDR
)

os.environ.setdefault(
    "MASTER_PORT",
    MASTER_PORT
)


distributed_init_method = (
    f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
)


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
# Model config
# ============================================================

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

config.cross_attn_norm = True
config.qk_norm = "rms_norm_across_heads"
config.eps = 1e-6

config.num_frames_per_block = 3
config.sliding_window_num_frames = 21



# ============================================================
# Build model
# ============================================================

print("init dit......")


model = CausalWanTransformer3DModel(
    config=config,
    hf_config=hf_config,
)


model = model.to(
    device=DEVICE,
    dtype=DTYPE,
)

model.eval()


print("model ready")



# ============================================================
# Input
# ============================================================

B = 1


hidden_states = torch.randn(
    B,
    16,
    3,
    60,
    104,
    device=DEVICE,
    dtype=DTYPE,
)



encoder_hidden_states = torch.randn(
    B,
    512,
    4096,
    device=DEVICE,
    dtype=torch.float32,
)



timestep = torch.randint(
    0,
    1000,
    (B,1),
    device=DEVICE,
    dtype=torch.long,
)



# ============================================================
# Chunk state
# ============================================================

chunk_size = 3

chunk_idx = args.chunk_idx


start_frame = (
    chunk_idx *
    chunk_size
)


frame_seq_length = (
    60 // 2
) * (
    104 // 2
)


current_start = (
    start_frame *
    frame_seq_length
)



print("\n===== Runtime Config =====")
print("num sms       :", args.num_sms)
print("chunk idx     :", chunk_idx)
print("start frame   :", start_frame)
print("current start :", current_start)
print("==========================\n")



# ============================================================
# KV cache
# ============================================================

num_layers = 30

num_heads = 12
head_dim = 128


local_heads = num_heads


kv_cache_frames = 21


kv_cache_tokens = (
    frame_seq_length *
    kv_cache_frames
)



kv_cache = []


for _ in range(num_layers):

    kv_cache.append({

        "k":
        torch.zeros(
            [
                B,
                kv_cache_tokens,
                local_heads,
                head_dim
            ],
            device=DEVICE,
            dtype=DTYPE,
        ),

        "v":
        torch.zeros(
            [
                B,
                kv_cache_tokens,
                local_heads,
                head_dim
            ],
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



# ============================================================
# Cross attention cache
# ============================================================

crossattn_cache = []


for _ in range(num_layers):

    crossattn_cache.append({

        "k":
        torch.zeros(
            [
                B,
                512,
                num_heads,
                head_dim,
            ],
            device=DEVICE,
            dtype=DTYPE,
        ),

        "v":
        torch.zeros(
            [
                B,
                512,
                num_heads,
                head_dim,
            ],
            device=DEVICE,
            dtype=DTYPE,
        ),

        "is_init":
        True,

    })



# ============================================================
# Forward helper
# ============================================================

def run_forward():

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ), set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):

            out = model._forward_inference(
                hidden_states=hidden_states,

                encoder_hidden_states=encoder_hidden_states,

                timestep=timestep,

                encoder_hidden_states_image=None,

                kv_cache=kv_cache,

                crossattn_cache=crossattn_cache,

                current_start=current_start,

                cache_start=0,

                start_frame=start_frame,
            )

    return out



# ============================================================
# Warmup
# ============================================================

for i in range(args.warmup_iters):

    with torch.cuda.stream(green_stream):

        out = run_forward()


torch.cuda.synchronize()


print("warmup finished")



# ============================================================
# Profile
# ============================================================

start_event = torch.cuda.Event(
    enable_timing=True
)

end_event = torch.cuda.Event(
    enable_timing=True
)


times = []


for i in range(args.profile_iters):

    with torch.cuda.stream(green_stream):

        start_event.record(
            stream=green_stream
        )

        out = run_forward()

        end_event.record(
            stream=green_stream
        )


    end_event.synchronize()


    elapsed = start_event.elapsed_time(
        end_event
    )

    times.append(elapsed)


    print(
        f"iter {i}: {elapsed:.3f} ms"
    )



avg = sum(times) / len(times)


print("\n====================")
print(
    f"SMs={args.num_sms}, "
    f"chunk={args.chunk_idx}"
)

print(
    f"RESULT "
    f"chunk={args.chunk_idx} "
    f"sms={args.num_sms} "
    f"latency={avg:.3f}"
)

print("====================")


print(
    "output:",
    out.shape
)