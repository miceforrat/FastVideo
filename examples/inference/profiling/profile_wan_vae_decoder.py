import torch

from fastvideo.models.vaes.wanvae import WanDecoder3d


# dim: 96
# z_dim: 16
# dim_mult: [1, 2, 4, 4]
# num_res_blocks: 2
# temperal_upsample: [True, True, False]
# out_channels: 3
# is_residual: False
# dims: [384, 384, 384, 192, 96]
# (Worker pid=347049) iter: 21
# (Worker pid=347049) decoder input: torch.Size([1, 16, 1, 60, 104]) torch.float32 


torch.cuda.set_device(0)


# =========================
# build decoder (random weight)
# =========================

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

decoder = decoder.cuda()
decoder.eval()


# =========================
# input from real decode
# =========================

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
    device="cuda",
    dtype=torch.float32,
)


# =========================
# warmup
# =========================

with torch.no_grad():
    for _ in range(5):
        out = decoder(z)

torch.cuda.synchronize()


# =========================
# nvtx range
# =========================

with torch.no_grad():
    with torch.cuda.nvtx.range("WanDecoder_only"):
        out = decoder(z)

torch.cuda.synchronize()


print("input:", z.shape)
print("output:", out.shape)