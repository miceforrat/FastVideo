import greenctx


print("create green context")


gc = greenctx.GreenContext(
    100,
    0
)


print("GC created")


s0 = gc.dit_stream()

s1 = gc.vae_stream()


print("DiT stream:", hex(s0))
print("VAE stream:", hex(s1))