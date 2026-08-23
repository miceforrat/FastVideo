# DiT/VAE 共置关键 Kernel 的 NCU 采集命令

## 1. 目标与前置条件

本文给出以下六个高贡献 kernel signature 的 Nsight Compute（NCU）采集命令，并分别采集 alone 与 colocated：

| 模块 | 排名 | Kernel | Grid | Block | Nsys绝对膨胀 |
|---|---:|---|---|---|---:|
| DiT | 1 | `flash_fwd_kernel` | `(74,1,12)` | `(128,1,1)` | 43.531 ms |
| DiT | 2 | `elementwise_kernel` | `(14040,1,1)` | `(128,1,1)` | 34.666 ms |
| DiT | 3 | `elementwise_kernel` | `(28080,1,1)` | `(128,1,1)` | 14.084 ms |
| VAE | 1 | 主 XMMA alignc4 | `(1,12480,1)` | `(128,1,1)` | 26.652 ms |
| VAE | 2 | 主 XMMA alignc4 | `(2,3120,1)` | `(128,1,1)` | 22.338 ms |
| VAE | 3 | `elementwise_kernel` | `(599040,1,1)` | `(128,1,1)` | 12.584 ms |

说明：这里按“具体 kernel signature 的绝对膨胀量”排序，而不是按 family 聚合或相对减速排序。FA 相对只慢5.36%，但 baseline 很长，所以仍是 DiT 单个 signature 中绝对贡献最大的 kernel。

以下命令假设：

```text
容器：cjh-Fastvideo-cu128
工作目录：/FastVideo/green+pytorch_demo
物理 GPU：1
DiT SM：112
VAE SM：58
```

进入容器：

```bash
docker exec -it cjh-Fastvideo-cu128 bash
cd /FastVideo/green+pytorch_demo
```

采集期间应确保物理 GPU 1 没有其他任务运行。

## 2. 必要的 NVTX 插桩

### 2.1 标记 DiT 的计时 baseline，而不是 warmup

DiT baseline 的前三次调用是 warmup。如果简单地在整个 `benchmark_stream()` 外包 `DiT_eager_alone`，NCU 的 `--launch-count 1` 会优先命中 warmup。建议给 `benchmark_stream()` 增加一个可选的 NVTX 参数，只包围正式计时循环。

将函数签名改为：

```python
def benchmark_stream(
    fn,
    stream,
    warmup_iters,
    profile_iters,
    profile_nvtx_range=None,
):
```

在 warmup 完成并执行 `stream.synchronize()` 后、创建计时 event 前加入：

```python
if profile_nvtx_range is not None:
    torch.cuda.nvtx.range_push(profile_nvtx_range)
```

在 `end.synchronize()` 后、`return` 前加入：

```python
if profile_nvtx_range is not None:
    torch.cuda.nvtx.range_pop()
```

即函数尾部应类似：

```python
with torch.cuda.stream(stream):
    for _ in range(warmup_iters):
        fn()

stream.synchronize()

if profile_nvtx_range is not None:
    torch.cuda.nvtx.range_push(profile_nvtx_range)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

with torch.cuda.stream(stream):
    start.record(stream)
    for _ in range(profile_iters):
        fn()
    end.record(stream)

end.synchronize()

if profile_nvtx_range is not None:
    torch.cuda.nvtx.range_pop()

return start.elapsed_time(end) / profile_iters
```

DiT baseline 调用改为：

```python
dit_time = benchmark_stream(
    run_dit_chunk,
    dit_stream,
    args.warmup_iters,
    args.profile_iters,
    profile_nvtx_range="DiT_eager_alone",
)
```

不要再在整个 DiT `benchmark_stream()` 调用外额外包同名 range，否则会产生嵌套的重复名称。

### 2.2 VAE Graph alone

VAE Graph alone 需要保留现有插桩：

```python
torch.cuda.nvtx.range_push("VAE_graph_alone")

vae_graph_time = benchmark_stream(
    replay_vae_graph,
    vae_stream,
    0,
    args.profile_iters,
)

torch.cuda.nvtx.range_pop()
```

`replay_vae_graph()` 内需要保留：

```python
torch.cuda.nvtx.range_push("VAE_graph_replay_host")
vae_graph.replay()
torch.cuda.nvtx.range_pop()
```

### 2.3 Colocated iteration

共置循环需要保留以下层次：

```text
Colocated_iteration
├── VAE_graph_replay_host
└── DiT_after_VAE_graph_replay
    └── DiT_chunk
```

对应代码框架为：

```python
torch.cuda.nvtx.range_push("Colocated_iteration")

with torch.cuda.stream(vae_stream):
    replay_vae_graph()

torch.cuda.nvtx.range_push("DiT_after_VAE_graph_replay")
with torch.cuda.stream(dit_stream):
    run_dit_chunk()
torch.cuda.nvtx.range_pop()

# 原有 event synchronize 和计时代码

torch.cuda.nvtx.range_pop()
```

## 3. 公共 NCU 参数

先在同一个 shell 中定义公共参数和 VAE主 XMMA 的匹配表达式：

```bash
NCU_COMMON=(
    --replay-mode application
    --app-replay-mode relaxed
    --check-exit-code no
    --graph-profiling node
    --nvtx
    --kernel-name-base function
    --launch-count 1
    --section WarpStateStats
    --section SchedulerStats
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --section LaunchStats
    --clock-control none
    --cache-control none
    --force-overwrite
)

VAE_MAIN_XMMA='regex:.*nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel__5x_cudnn.*'
```

选择 `application replay` 是为了让每个计数器 pass 都重新运行完整程序，尽量保持 VAE与DiT的共置关系。默认 `kernel replay` 会脱离原始执行关系重复单个 kernel，不适合作为共置竞争的主要证据。

保留 `--clock-control none` 和 `--cache-control none` 是为了维持实际共置条件。NCU 会提示时钟和缓存未受控，因此应使用相同参数重复 alone/colocated，并主要比较 stall 构成；真实时间膨胀仍以 Nsys/CUDA Event 为准。

## 4. DiT Top-3 采集命令

DiT的 `--launch-skip` 已根据 Nsys 中首个正式计时 chunk 的实际顺序计算。alone 与 colocated 的执行顺序一致。

### 4.1 DiT #1：FlashAttention

目标：

```text
flash_fwd_kernel
grid=(74,1,12), block=(128,1,1)
matching-name launch-skip=0
```

Alone：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "DiT_eager_alone/DiT_chunk/" \
    --kernel-name flash_fwd_kernel \
    --launch-skip 0 \
    -o ncu_dit_01_flash_alone \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

Colocated：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "Colocated_iteration/DiT_after_VAE_graph_replay/DiT_chunk/" \
    --kernel-name flash_fwd_kernel \
    --launch-skip 0 \
    -o ncu_dit_01_flash_colocated \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

### 4.2 DiT #2：Elementwise，grid `(14040,1,1)`

目标：

```text
elementwise_kernel
grid=(14040,1,1), block=(128,1,1)
matching-name launch-skip=18
```

Alone：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "DiT_eager_alone/DiT_chunk/" \
    --kernel-name elementwise_kernel \
    --launch-skip 18 \
    -o ncu_dit_02_elementwise_14040_alone \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

Colocated：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "Colocated_iteration/DiT_after_VAE_graph_replay/DiT_chunk/" \
    --kernel-name elementwise_kernel \
    --launch-skip 18 \
    -o ncu_dit_02_elementwise_14040_colocated \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

### 4.3 DiT #3：Elementwise，grid `(28080,1,1)`

目标：

```text
elementwise_kernel
grid=(28080,1,1), block=(128,1,1)
matching-name launch-skip=20
```

Alone：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "DiT_eager_alone/DiT_chunk/" \
    --kernel-name elementwise_kernel \
    --launch-skip 20 \
    -o ncu_dit_03_elementwise_28080_alone \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

Colocated：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "Colocated_iteration/DiT_after_VAE_graph_replay/DiT_chunk/" \
    --kernel-name elementwise_kernel \
    --launch-skip 20 \
    -o ncu_dit_03_elementwise_28080_colocated \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

## 5. VAE Top-3 采集命令

VAE 的两个主 XMMA signature 使用相同 kernel name，但 grid 不同，因此通过匹配该完整 XMMA family 后的 `--launch-skip` 区分。第三个目标是单独匹配 `elementwise_kernel` 后的第192次调用，即 `--launch-skip 191`。

### 5.1 VAE #1：主 XMMA，grid `(1,12480,1)`

目标：

```text
main XMMA alignc4
grid=(1,12480,1), block=(128,1,1)
matching-family launch-skip=14
```

Alone：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "VAE_graph_alone/VAE_graph_replay_host/" \
    --kernel-name "$VAE_MAIN_XMMA" \
    --launch-skip 14 \
    -o ncu_vae_01_xmma_1x12480_alone \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

Colocated：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "Colocated_iteration/VAE_graph_replay_host/" \
    --kernel-name "$VAE_MAIN_XMMA" \
    --launch-skip 14 \
    -o ncu_vae_01_xmma_1x12480_colocated \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

### 5.2 VAE #2：主 XMMA，grid `(2,3120,1)`

目标：

```text
main XMMA alignc4
grid=(2,3120,1), block=(128,1,1)
matching-family launch-skip=7
```

Alone：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "VAE_graph_alone/VAE_graph_replay_host/" \
    --kernel-name "$VAE_MAIN_XMMA" \
    --launch-skip 7 \
    -o ncu_vae_02_xmma_2x3120_alone \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

Colocated：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "Colocated_iteration/VAE_graph_replay_host/" \
    --kernel-name "$VAE_MAIN_XMMA" \
    --launch-skip 7 \
    -o ncu_vae_02_xmma_2x3120_colocated \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

### 5.3 VAE #3：Elementwise，grid `(599040,1,1)`

目标：

```text
elementwise_kernel
grid=(599040,1,1), block=(128,1,1)
matching-name launch-skip=191
```

Alone：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "VAE_graph_alone/VAE_graph_replay_host/" \
    --kernel-name elementwise_kernel \
    --launch-skip 191 \
    -o ncu_vae_03_elementwise_599040_alone \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

Colocated：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "Colocated_iteration/VAE_graph_replay_host/" \
    --kernel-name elementwise_kernel \
    --launch-skip 191 \
    -o ncu_vae_03_elementwise_599040_colocated \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --warmup-iters 3 \
        --profile-iters 1
```

## 6. 每次采集后的目标核验

`--launch-skip` 依赖当前脚本的 kernel 顺序。只要模型、输入形状和 forward 路径不变，alone 与 colocated 中的索引一致；如果修改了模型实现、feature cache、算子融合或输入形状，必须重新计算。

每跑完一条命令，先检查报告是否存在：

```bash
ls -lh ncu_*.ncu-rep
```

然后打印详情并确认：

```bash
ncu --import ncu_dit_02_elementwise_14040_alone.ncu-rep --page details
```

至少确认以下字段：

```text
Kernel Name
NVTX Push/Pop Stack
Grid Size / Grid Dimensions
Block Size
# SMs
Uses Green Context
```

alone 与 colocated 必须命中相同：

```text
kernel name
grid
block
SM count
```

如果 grid 不符合本文章节中的目标值，应停止使用该组结果，而不是继续比较。最常见原因是脚本变化导致 `--launch-skip` 失效。

## 7. 结果对照重点

每一对 alone/colocated 报告重点比较：

| 指标 | 解释方向 |
|---|---|
| `Duration` | NCU采样下的目标 kernel 时间，仅作辅助 |
| `SM Frequency` | 判断共置降频是否参与膨胀 |
| `Long Scoreboard` | 长延迟数据依赖，常与 L2 miss/DRAM访问有关 |
| `Short Scoreboard` | 较短的 shared memory/MIO依赖 |
| `LG Throttle` | local/global memory 指令队列压力 |
| `MIO Throttle` | MIO管线拥塞 |
| `Math Pipe Throttle` | 数学执行管线拥塞 |
| `Not Selected` | eligible warp 之间的调度竞争 |
| `Eligible Warps Per Scheduler` | 可用于隐藏延迟的 warp 数量 |
| `L1/TEX Hit Rate` | L1/TEX局部性变化 |
| `L2 Hit Rate` | DiT/VAE共享 L2 干扰 |
| `DRAM Throughput` | 是否转化为更多 HBM流量或带宽压力 |

对于 FA 和主 XMMA，应特别注意“相对膨胀较小但绝对贡献较大”的情况。NCU 分析的是单个 invocation 的微架构行为，而 Nsys 的绝对贡献还取决于调用次数和 baseline 累计时间。

对于 elementwise，应重点检查：

```text
L2 Hit Rate 是否下降
Long Scoreboard 是否上升
LG/MIO Throttle 是否上升
Eligible Warps 是否下降
SM Frequency 是否下降
```

如果 elementwise 共置后 L2 hit rate 下降且 Long Scoreboard 明显上升，才形成“共享 L2/访存延迟导致其30%以上膨胀”的直接证据。如果 stall 基本不变但所有 kernel 近似按同一比例变慢，则应优先考虑降频或更全局的执行资源影响。

## 8. 建议执行顺序

为了尽快获得最有区分度的结果，建议按以下顺序运行：

1. DiT `elementwise_kernel (14040,1,1)` alone/colocated；
2. DiT FA alone/colocated；
3. VAE主 XMMA `(1,12480,1)` alone/colocated；
4. VAE `elementwise_kernel (599040,1,1)` alone/colocated；
5. DiT `elementwise_kernel (28080,1,1)` alone/colocated；
6. VAE主 XMMA `(2,3120,1)` alone/colocated。

这个顺序同时覆盖：高相对膨胀的 elementwise、低相对但高绝对贡献的 FA/XMMA，以及 DiT/VAE 两侧的对照。
