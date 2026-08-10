import argparse
import torch

from fastvideo.models.vaes.wanvae import (
    WanDecoder3d,
    WanCausalConv3d,
    forward_context,
)

from torch.cuda.green_contexts import GreenContext



# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Profile Wan VAE Decoder with Feature Cache"
    )

    parser.add_argument(
        "--num-sms",
        type=int,
        default=170,
        help="GreenContext SM number",
    )

    parser.add_argument(
        "--num-chunks",
        type=int,
        default=21,
        help="number of streaming chunks",
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--profile-iters",
        type=int,
        default=5,
    )

    return parser.parse_args()



args = parse_args()



# ============================================================
# Runtime
# ============================================================

DEVICE = "cuda:0"
DTYPE = torch.float32


torch.cuda.set_device(0)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



# ============================================================
# Green Context
# ============================================================

green_ctx = GreenContext.create(
    num_sms=args.num_sms
)

green_stream = green_ctx.Stream()



# ============================================================
# Build decoder
# ============================================================

print("init VAE decoder......")


decoder = WanDecoder3d(
    dim=96,
    z_dim=16,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=(),
    temperal_upsample=[True, True, False],
    dropout=0.0,
    out_channels=3,
    is_residual=False,
)


decoder = (
    decoder
    .cuda()
    .eval()
)


print("decoder ready")



# ============================================================
# Feature cache init
# ============================================================

def count_causal_conv(model):

    return sum(
        1
        for m in model.modules()
        if isinstance(m, WanCausalConv3d)
    )



num_cache = count_causal_conv(decoder)


print(
    "causal conv num:",
    num_cache
)



# streaming cache

feat_cache_map = [
    None
    for _ in range(num_cache)
]



# ============================================================
# Input
# ============================================================

B = 1
C = 16

# one latent frame per chunk
T = 1

H = 60
W = 104



# prepare multiple chunks
z_chunks = []


for i in range(args.num_chunks):

    z_chunks.append(
        torch.randn(
            B,
            C,
            T,
            H,
            W,
            device=DEVICE,
            dtype=DTYPE,
        )
    )



print("\n===== Runtime Config =====")
print("SMs       :", args.num_sms)
print("chunks    :", args.num_chunks)
print("input     :", z_chunks[0].shape)
print("==========================\n")



# ============================================================
# Forward helper
# ============================================================


def run_chunk(z):

    global feat_cache_map


    with torch.no_grad():

        with forward_context(
            feat_cache_arg=feat_cache_map,
            feat_idx_arg=0,
            first_chunk_arg=False,
        ):

            out = decoder(z)


    return out



# ============================================================
# Warmup
# ============================================================

for _ in range(args.warmup_iters):

    # reset cache
    feat_cache_map = [
        None
        for _ in range(num_cache)
    ]


    for chunk in range(args.num_chunks):

        with torch.cuda.stream(green_stream):

            out = run_chunk(
                z_chunks[chunk]
            )


green_stream.synchronize()


print("warmup finished")



# ============================================================
# Profile
# ============================================================


all_times = []



for chunk in range(args.num_chunks):


    times = []


    for i in range(args.profile_iters):


        # 每次重新模拟一次完整 streaming
        feat_cache_map = [
            None
            for _ in range(num_cache)
        ]


        # 先推进到当前 chunk
        for prev in range(chunk):

            with torch.cuda.stream(green_stream):

                _ = run_chunk(
                    z_chunks[prev]
                )


        torch.cuda.synchronize()



        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )



        with torch.cuda.stream(green_stream):

            torch.cuda.nvtx.range_push(
                f"WanDecoder_featcache_chunk{chunk}_SM{args.num_sms}"
            )


            start_event.record(
                stream=green_stream
            )


            out = run_chunk(
                z_chunks[chunk]
            )


            end_event.record(
                stream=green_stream
            )


            torch.cuda.nvtx.range_pop()



        end_event.synchronize()


        elapsed = start_event.elapsed_time(
            end_event
        )


        times.append(elapsed)



    avg = sum(times) / len(times)

    all_times.append(avg)


    print(
        f"chunk {chunk}: "
        f"{avg:.3f} ms"
    )


    print(
        f"RESULT "
        f"chunk={chunk} "
        f"sms={args.num_sms} "
        f"latency={avg:.3f}"
    )



# ============================================================
# Result
# ============================================================


print("\n====================")

for i, t in enumerate(all_times):

    print(
        f"chunk={i}: {t:.3f} ms"
    )


print("====================")

print(
    "output:",
    out.shape
)