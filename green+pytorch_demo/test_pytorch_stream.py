import torch
import greenctx


gc = greenctx.GreenContext(
    80,
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


a = torch.randn(
    4096,
    4096,
    device="cuda"
)

b = torch.randn(
    4096,
    4096,
    device="cuda"
)


with torch.cuda.stream(dit_stream):
    c = a @ b


torch.cuda.synchronize()

print(c)