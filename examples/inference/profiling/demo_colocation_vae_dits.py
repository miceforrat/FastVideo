import time
import torch

from torch.cuda.green_contexts import GreenContext


torch.cuda.set_device(0)


# ============================================================
# Green Context partition
# ============================================================

TOTAL_SMS = 170

DIT_SMS = 170
VAE_SMS = 170


dit_ctx = GreenContext.create(
    num_sms=DIT_SMS
)

vae_ctx = GreenContext.create(
    num_sms=VAE_SMS
)


dit_stream = dit_ctx.Stream()
vae_stream = vae_ctx.Stream()



# ============================================================
# Workload
# ============================================================

A = torch.randn(
    8192,
    8192,
    device="cuda",
    dtype=torch.float16,
)

B = torch.randn(
    8192,
    8192,
    device="cuda",
    dtype=torch.float16,
)



def dit_work():

    out = None

    for _ in range(20):

        out = torch.matmul(
            A,
            B
        )

    return out



def vae_work():

    out = None

    for _ in range(20):

        out = torch.matmul(
            A,
            B
        )

    return out



# ============================================================
# Warmup
# ============================================================


with torch.cuda.stream(dit_stream):

    dit_work()


with torch.cuda.stream(vae_stream):

    vae_work()


dit_stream.synchronize()
vae_stream.synchronize()



# ============================================================
# Baseline:
# one normal CUDA stream
# ============================================================


torch.cuda.synchronize()


start = torch.cuda.Event(
    enable_timing=True
)

end = torch.cuda.Event(
    enable_timing=True
)


start.record()


dit_work()

vae_work()


end.record()


end.synchronize()


seq_ms = start.elapsed_time(end)



# ============================================================
# GreenContext concurrent
# ============================================================


torch.cuda.synchronize()


start.record()


with torch.cuda.stream(dit_stream):

    dit_work()


with torch.cuda.stream(vae_stream):

    vae_work()



dit_stream.synchronize()
vae_stream.synchronize()



end.record()


end.synchronize()


con_ms = start.elapsed_time(end)



print("==========================")
print(
    f"DIT SMs : {DIT_SMS}"
)

print(
    f"VAE SMs : {VAE_SMS}"
)

print(
    f"sequential : {seq_ms:.3f} ms"
)

print(
    f"concurrent : {con_ms:.3f} ms"
)

print(
    f"speedup    : {seq_ms/con_ms:.3f}x"
)

print("==========================")