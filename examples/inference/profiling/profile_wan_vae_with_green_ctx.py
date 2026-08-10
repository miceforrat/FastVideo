import argparse
import torch

from fastvideo.models.vaes.wanvae import WanDecoder3d

from torch.cuda.green_contexts import GreenContext



# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Profile Wan VAE Decoder with GreenContext"
    )

    parser.add_argument(
        "--num-sms",
        type=int,
        default=170,
        help="GreenContext SM number",
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
# Input
# ============================================================

B = 1
C = 16
T = 1
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



print("\n===== Runtime Config =====")
print("num sms :", args.num_sms)
print("input   :", z.shape)
print("==========================\n")



# ============================================================
# Forward helper
# ============================================================

def run_forward():

    with torch.no_grad():

        out = decoder(z)

    return out



# ============================================================
# Warmup
# ============================================================

for _ in range(args.warmup_iters):

    with torch.cuda.stream(green_stream):

        out = run_forward()



green_stream.synchronize()


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

        torch.cuda.nvtx.range_push(
            f"WanDecoder_green_{args.num_sms}SM"
        )


        start_event.record(
            stream=green_stream
        )


        out = run_forward()


        end_event.record(
            stream=green_stream
        )


        torch.cuda.nvtx.range_pop()


    end_event.synchronize()


    elapsed = start_event.elapsed_time(
        end_event
    )


    times.append(elapsed)


    print(
        f"iter {i}: {elapsed:.3f} ms"
    )



avg = sum(times) / len(times)



# ============================================================
# Result
# ============================================================

print("\n====================")
print(
    f"RESULT "
    f"sms={args.num_sms} "
    f"latency={avg:.3f} ms"
)

print(
    "input :",
    z.shape
)

print(
    "output:",
    out.shape
)

print("====================")