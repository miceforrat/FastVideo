import argparse
import torch

from fastvideo.models.vaes.wanvae import (
    WanDecoder3d,
    WanCausalConv3d,
    forward_context,
    first_chunk,
    feat_idx,
)

from torch.cuda.green_contexts import GreenContext



# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Profile Wan VAE streaming decode steady state"
    )

    parser.add_argument(
        "--num-sms",
        type=int,
        default=170,
    )

    parser.add_argument(
        "--num-latents",
        type=int,
        default=3,
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
# Build VAE
# ============================================================

print("init VAE......")


post_quant_conv = WanCausalConv3d(
    16,
    16,
    1,
)


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


print("VAE ready")



# ============================================================
# Feature cache
# ============================================================

def count_causal_conv(model):

    return sum(
        1
        for m in model.modules()
        if isinstance(m, WanCausalConv3d)
    )


num_cache = count_causal_conv(decoder)


print(
    "decoder causal conv:",
    num_cache
)



feat_cache_map = [
    None
    for _ in range(num_cache)
]



# ============================================================
# Input
# ============================================================

B = 1
C = 16

T = args.num_latents

H = 60
W = 104



z = torch.randn(
    B,
    C,
    T,
    H,
    W,
    device=DEVICE,
    dtype=DTYPE,
)


print("\n========== Config ==========")
print("SMs:", args.num_sms)
print("latent:", z.shape)
print("============================")



# ============================================================
# Prepare cache
# ============================================================

# 对应 AutoencoderKLWan:
#
# self.clear_cache()
# x = self.post_quant_conv(z)
#
# decoder streaming
#
# 这里人为先跑一个 dummy latent
# 让所有 conv cache 非空


print("initialize feature cache...")


dummy = torch.randn(
    B,
    C,
    1,
    H,
    W,
    device=DEVICE,
    dtype=DTYPE,
)


dummy_x = post_quant_conv(dummy)


with torch.no_grad():

    with forward_context(
        feat_cache_arg=feat_cache_map,
        feat_idx_arg=0,
    ):

        feat_idx.set(0)

        # 非 first
        first_chunk.set(False)

        _ = decoder(
            dummy_x
        )


torch.cuda.synchronize()


print("cache initialized")



# ============================================================
# Warmup
# ============================================================

x = post_quant_conv(z)


for _ in range(args.warmup_iters):

    with torch.cuda.stream(
        green_stream
    ):

        with torch.no_grad():

            with forward_context(
                feat_cache_arg=feat_cache_map,
                feat_idx_arg=0,
            ):

                for i in range(T):

                    feat_idx.set(0)

                    # 所有 latent 非 first
                    first_chunk.set(False)


                    _ = decoder(
                        x[:, :, i:i+1, :, :]
                    )


green_stream.synchronize()


print("warmup finished")



# ============================================================
# Profile whole chunk
# post_quant_conv + T latent decoder
# ============================================================

times = []


for rep in range(args.profile_iters):

    start_event = torch.cuda.Event(
        enable_timing=True
    )

    end_event = torch.cuda.Event(
        enable_timing=True
    )


    with torch.cuda.stream(
        green_stream
    ):

        torch.cuda.nvtx.range_push(
            f"vae_chunk_decode_SM{args.num_sms}"
        )


        start_event.record(
            stream=green_stream
        )


        with torch.no_grad():


            # post quant conv
            x = post_quant_conv(z)


            outputs = []


            with forward_context(
                feat_cache_arg=feat_cache_map,
                feat_idx_arg=0,
            ):


                for i in range(T):

                    feat_idx.set(0)

                    # 所有 latent 非 first
                    first_chunk.set(False)


                    out_i = decoder(
                        x[:, :, i:i+1, :, :]
                    )


                    outputs.append(
                        out_i
                    )


            out = torch.cat(
                outputs,
                dim=2
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


    # 单次打印
    print(
        f"iter {rep}: "
        f"{elapsed:.3f} ms"
    )



avg = sum(times) / len(times)


print("\n====================")

print(
    f"RESULT "
    f"frames={T} "
    f"sms={args.num_sms} "
    f"latency={avg:.3f} ms"
)


print(
    "output:",
    out.shape
)

print("====================")