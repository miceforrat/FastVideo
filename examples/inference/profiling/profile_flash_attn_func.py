import statistics
import torch
from flash_attn import flash_attn_func

torch.manual_seed(0)
torch.cuda.set_device(0)

B = 1
Sq = 4680
MAX_CHUNKS = 7   # 测 chunk 0 ~ 6
H = 12
D = 128

dtype = torch.float16
device = "cuda"
causal = False
softmax_scale = None

iters = 1       # 每次 timing 内部跑多少轮
warmup = 3       # 每个 chunk 正式计时前 warmup
repeats = 5      # 每个 chunk 重复测几次

q = torch.randn(B, Sq, H, D, device=device, dtype=dtype)

# 一次性生成最大 KV，后面按 chunk 截取
Sk_max = Sq * MAX_CHUNKS
k_all = torch.randn(B, Sk_max, H, D, device=device, dtype=dtype)
v_all = torch.randn(B, Sk_max, H, D, device=device, dtype=dtype)

results = {}

with torch.inference_mode():
    for chunk_id in range(MAX_CHUNKS):
        num_chunks = chunk_id + 1
        Sk = Sq * num_chunks

        # 当前 chunk 生成时能看到的 KV 长度
        # contiguous() 不放进计时区
        k = k_all[:, :Sk].contiguous()
        v = v_all[:, :Sk].contiguous()

        print(f"\n===== chunk {chunk_id} / KV chunks={num_chunks}, Sk={Sk} =====")

        # ===== warmup =====
        for _ in range(warmup):
            out = flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=causal
            )

        torch.cuda.synchronize()

        chunk_times = []

        # ===== repeated timing =====
        torch.cuda.nvtx.range_push(f"flash_attn_chunk_{chunk_id}")

        for r in range(repeats):
            torch.cuda.nvtx.range_push(f"repeat_{r}")

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()

            for it in range(iters):
                torch.cuda.nvtx.range_push(f"iter_{it}")

                out = flash_attn_func(
                    q, k, v,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=causal
                )

                torch.cuda.nvtx.range_pop()

            end.record()
            torch.cuda.synchronize()

            avg_ms = start.elapsed_time(end) / iters
            chunk_times.append(avg_ms)

            torch.cuda.nvtx.range_pop()

            print(f"repeat {r}: {avg_ms:.4f} ms")

        torch.cuda.nvtx.range_pop()

        mean_ms = statistics.mean(chunk_times)
        std_ms = statistics.stdev(chunk_times) if len(chunk_times) > 1 else 0.0

        results[chunk_id] = {
            "Sk": Sk,
            "times": chunk_times,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        }

        print(f"chunk {chunk_id} mean: {mean_ms:.4f} ms, std: {std_ms:.4f} ms")

print("\n========== Summary ==========")
print("chunk_id,Sk,mean_ms,std_ms,times")

for chunk_id, item in results.items():
    times_str = "[" + ", ".join(f"{x:.4f}" for x in item["times"]) + "]"
    print(
        f"{chunk_id},"
        f"{item['Sk']},"
        f"{item['mean_ms']:.4f},"
        f"{item['std_ms']:.4f},"
        f"{times_str}"
    )

print("\nout:", out.shape, out.dtype)