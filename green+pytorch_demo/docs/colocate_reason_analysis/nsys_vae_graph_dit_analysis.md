# CUDA Graph 共置下 DiT/VAE Kernel 膨胀分析

## 1. 分析目标

本文重新分析 VAE CUDA Graph 与 eager DiT 共置时的 kernel 时间膨胀，回答以下问题：

1. DiT 和 VAE 的 kernel active time 分别增长了多少；
2. 哪些 kernel 对绝对增长时间贡献最大；
3. 哪些 kernel 的相对减速比例最高；
4. 此前只分析 VAE 的一个 XMMA kernel、只优先考虑 DiT FlashAttention（FA）有哪些局限。

分析所用报告为：

```text
green+pytorch_demo/dit_vae_cuda_graph_node_gpu_metrics.nsys-rep
```

该报告使用 CUDA Graph node 级追踪，并包含三次 alone 计时迭代和三次 colocated 迭代。

## 2. 统计方法

### 2.1 DiT

脚本的 DiT baseline 包含三次 warmup 和三次计时迭代。因此：

- 丢弃前三个 `DiT_chunk`；
- 使用随后三个 `DiT_chunk` 作为 DiT alone；
- 使用三个 `DiT_after_VAE_graph_replay/DiT_chunk` 作为 colocated；
- 只统计 DiT Green Context stream 上的 kernel。

最后一条非常重要。共置阶段 DiT 的 CPU NVTX 时间窗与 VAE 的 GPU 执行重叠，如果只按时间范围统计，会把 VAE kernel 错算进 DiT，使 DiT active time 虚假地变成约 2.25 倍。加入 stream 限制后，DiT 的 kernel signature 数量在两组中均为 42，调用次数也一致。

### 2.2 VAE CUDA Graph

VAE Graph 的 host NVTX range 只覆盖 `graph.replay()` 的 CPU 提交，不能覆盖异步执行的全部 GPU节点。因此没有用 host range 判断 VAE kernel 归属，而是按 `graphNodeId` 配对。

每个 VAE graph node 在报告中稳定出现七次：

1. 第一次：capture 后的单独 warmup replay；
2. 第二至第四次：三次 graph-alone；
3. 第五至第七次：三次与 DiT 共置的 replay。

报告中的 1661 个 VAE graph node 全部完成七次配对，没有缺失节点。

### 2.3 Kernel 身份与排序方法

聚合时用以下字段区分 kernel signature：

```text
short kernel name + grid dimensions + block dimensions
```

每组结果均除以三，得到每个 chunk/replay 的平均累计时间。排序指标为：

```text
绝对膨胀贡献 = colocated 累计 kernel 时间 - alone 累计 kernel 时间
```

相对膨胀则为：

```text
相对膨胀比例 = colocated / alone - 1
```

绝对贡献和相对比例必须同时看：一个很长的 kernel 即使只变慢 5%，也可能贡献大量绝对时间；一个很短的 kernel 即使变慢 50%，对整体 wall time 的贡献仍可能很小。

## 3. 总体结果

| 模块 | Alone kernel active | Colocated kernel active | 增加 | 相对膨胀 |
|---|---:|---:|---:|---:|
| DiT | 1533.267 ms | 1689.257 ms | **155.991 ms** | **10.17%** |
| VAE Graph | 1715.529 ms | 1851.151 ms | **135.622 ms** | **7.91%** |

两组对照中的主要 kernel 调用次数一致。因此这里的增长不是共置时多执行了一批 kernel，而是同一批 kernel 的执行时间增长。

该结果也说明：VAE 使用 CUDA Graph 后，之前由逐算子 host 提交和 driver 锁争用造成的巨大非重叠空洞已明显减小；剩余开销主要表现为 GPU kernel active duration 增长。

## 4. DiT Kernel 膨胀排序

### 4.1 按 kernel family 聚合

| 排名 | Kernel family | Alone | Colocated | 绝对增加 | 相对膨胀 |
|---:|---|---:|---:|---:|---:|
| 1 | `elementwise_kernel` | 144.858 ms | 193.696 ms | **48.839 ms** | **33.71%** |
| 2 | `flash_fwd_kernel` | 812.840 ms | 856.371 ms | **43.531 ms** | **5.36%** |
| 3 | `Kernel2` | 469.720 ms | 496.346 ms | **26.626 ms** | **5.67%** |
| 4 | `vectorized_elementwise_kernel` | 51.547 ms | 68.485 ms | **16.938 ms** | **32.86%** |
| 5 | `unrolled_elementwise_kernel` | 38.810 ms | 53.018 ms | **14.209 ms** | **36.61%** |
| 6 | `vectorized_layer_norm_kernel` | 7.509 ms | 10.856 ms | **3.347 ms** | **44.57%** |
| 7 | `CatArrayBatchedCopy_vectorized` | 4.355 ms | 6.132 ms | **1.776 ms** | **40.79%** |
| 8 | `reduce_kernel` | 3.257 ms | 3.900 ms | 0.642 ms | 19.72% |

普通、vectorized 和 unrolled 三类 elementwise kernel 合计增加：

```text
48.839 + 16.938 + 14.209 = 79.986 ms
```

约占 DiT 全部 kernel active 膨胀：

```text
79.986 / 155.991 = 51.3%
```

因此，从 family 总量看，elementwise 是 DiT kernel 膨胀的主体。

### 4.2 FA：相对膨胀小，但绝对贡献高

FA 的结果是：

```text
Alone:       812.840 ms
Colocated:   856.371 ms
增加:         43.531 ms
相对膨胀:       5.36%
```

FA 只变慢约 5.36%，远低于 elementwise 常见的 30% 至 45%；但 FA baseline 本身长达约 813 ms，所以仍贡献 43.53 ms，占 DiT 总 kernel 膨胀的约 27.9%。

因此不能只看相对比例而认为 FA 不重要，也不能只看绝对值而认为 FA 是最敏感的 kernel。更准确的结论是：

> Elementwise 对共置更加敏感，是相对减速和合计绝对贡献的主体；FA 的敏感度较低，但由于基数很大，仍是不可忽略的第二大绝对贡献者。

### 4.3 DiT 中绝对贡献最大的具体 signature

| Kernel | Grid | Block | 调用/迭代 | Alone | Colocated | 增加 | 相对膨胀 |
|---|---|---|---:|---:|---:|---:|---:|
| `flash_fwd_kernel` | `(74,1,12)` | `(128,1,1)` | 300 | 812.840 ms | 856.371 ms | **43.531 ms** | 5.36% |
| `elementwise_kernel` | `(14040,1,1)` | `(128,1,1)` | 3615 | 107.989 ms | 142.655 ms | **34.666 ms** | **32.10%** |
| `elementwise_kernel` | `(28080,1,1)` | `(128,1,1)` | 905 | 36.658 ms | 50.742 ms | **14.084 ms** | **38.42%** |
| `unrolled_elementwise_kernel` | `(14040,1,1)` | `(128,1,1)` | 1805 | 37.988 ms | 52.006 ms | **14.018 ms** | **36.90%** |
| `vectorized_elementwise_kernel` | `(7020,1,1)` | `(128,1,1)` | 1955 | 36.921 ms | 49.760 ms | **12.839 ms** | **34.78%** |
| `Kernel2` | `(152,3,1)` | `(128,1,1)` | 900 | 162.594 ms | 172.879 ms | **10.285 ms** | 6.33% |
| `Kernel2` | `(152,2,1)` | `(256,1,1)` | 150 | 179.143 ms | 187.687 ms | **8.545 ms** | 4.77% |
| `Kernel2` | `(296,5,1)` | `(256,1,1)` | 150 | 127.538 ms | 135.276 ms | **7.738 ms** | 6.07% |
| `vectorized_layer_norm_kernel` | `(4680,1,1)` | `(32,4,1)` | 455 | 7.509 ms | 10.856 ms | 3.347 ms | **44.57%** |

## 5. VAE CUDA Graph Kernel 膨胀排序

### 5.1 按 kernel family 聚合

| 排名 | Kernel family | Alone | Colocated | 绝对增加 | 相对膨胀 |
|---:|---|---:|---:|---:|---:|
| 1 | 主 `xmma_fprop_implicit_gemm...alignc4` | 1181.235 ms | 1236.316 ms | **55.081 ms** | 4.66% |
| 2 | `elementwise_kernel` | 239.137 ms | 273.667 ms | **34.530 ms** | **14.44%** |
| 3 | `vectorized_elementwise_kernel` | 104.226 ms | 124.804 ms | **20.578 ms** | **19.74%** |
| 4 | `nchwToNhwcKernel` | 58.075 ms | 71.078 ms | **13.003 ms** | **22.39%** |
| 5 | `implicit_convolveNd_sgemm` | 30.785 ms | 35.319 ms | 4.533 ms | 14.73% |
| 6 | `reduce_kernel` | 11.530 ms | 13.302 ms | 1.773 ms | 15.38% |
| 7 | `indexed_wo_smem` XMMA | 20.930 ms | 22.694 ms | 1.764 ms | 8.43% |
| 8 | 另一种 XMMA | 39.160 ms | 40.768 ms | 1.607 ms | 4.10% |
| 9 | `upsample_nearest2d_out_frame` | 10.214 ms | 11.468 ms | 1.254 ms | 12.28% |
| 10 | `fmha_cutlassF_f32_aligned_32x128_gmem_sm80` | 18.275 ms | 19.516 ms | 1.241 ms | 6.79% |

VAE 的135.622 ms总 kernel 膨胀可粗略分解为：

| 类型 | 绝对增加 | 占总膨胀 |
|---|---:|---:|
| 主 XMMA卷积 | 55.081 ms | 40.6% |
| 普通及 vectorized elementwise | 55.107 ms | 40.6% |
| `nchwToNhwcKernel` | 13.003 ms | 9.6% |
| 其他 | 12.431 ms | 9.2% |

大型 XMMA卷积和 elementwise 两部分贡献几乎相同。前者相对减速只有约 4% 至 5%，但 baseline 极大；后者相对减速约 14% 至 20%，表现出更高的共置敏感性。

### 5.2 VAE 中绝对贡献最大的具体 signature

| Kernel | Grid | Block | 调用/迭代 | Alone | Colocated | 增加 | 相对膨胀 |
|---|---|---|---:|---:|---:|---:|---:|
| 主 XMMA alignc4 | `(1,12480,1)` | `(128,1,1)` | 21 | 511.240 ms | 537.891 ms | **26.652 ms** | 5.21% |
| 主 XMMA alignc4 | `(2,3120,1)` | `(128,1,1)` | 21 | 503.280 ms | 525.618 ms | **22.338 ms** | 4.44% |
| `elementwise_kernel` | `(599040,1,1)` | `(128,1,1)` | 87 | 93.029 ms | 105.613 ms | **12.584 ms** | **13.53%** |
| `vectorized_elementwise_kernel` | `(149760,1,1)` | `(128,1,1)` | 69 | 56.730 ms | 66.009 ms | **9.280 ms** | **16.36%** |
| `elementwise_kernel` | `(299520,1,1)` | `(128,1,1)` | 120 | 63.960 ms | 71.518 ms | **7.558 ms** | **11.82%** |
| 主 XMMA alignc4 | `(3,390,1)` | `(128,1,1)` | 18 | 160.103 ms | 165.849 ms | 5.746 ms | 3.59% |
| `nchwToNhwcKernel` | `(75373,3,1)` | `(256,1,1)` | 21 | 30.567 ms | 36.071 ms | 5.504 ms | **18.01%** |
| `elementwise_kernel` | `(898560,1,1)` | `(128,1,1)` | 21 | 34.456 ms | 39.842 ms | 5.386 ms | **15.63%** |

## 6. 对此前 VAE NCU 结论的修正

此前 NCU 使用以下过滤条件：

```text
regex:.*xmma_fprop_implicit_gemm.*
launch-count: 1
```

最终命中的是：

```text
sm80_xmma_fprop_implicit_gemm_indexed_wo_smem...
```

该 kernel 的 alone/colocated NCU 对照显示：

- L2 hit rate 从 97.72% 降至 94.69%；
- Long Scoreboard 没有增长；
- 单 kernel duration 基本不变；
- 主要 stall 仍为 Math Pipe Throttle。

这个结果本身有效，但现在从完整 Nsys 聚合可知，该 `indexed_wo_smem` family 只增加约 1.764 ms，仅占 VAE 总 kernel 膨胀约 1.3%。因此该 NCU 结论只能表述为：

> 这个特定 `indexed_wo_smem` 卷积虽受到一定 L2扰动，但该扰动没有进入其关键执行路径。

它不能扩展为“VAE所有主卷积都不是访存问题”。真正贡献55.081 ms的主 XMMA 是：

```text
sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_
nhwckrsc_nchw_tilesize128x128x16_stage4_
warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel__5x_cudnn
```

后续需要重新对该主 XMMA 的主要形状进行 NCU alone/colocated 对照。

## 7. 当前结论

1. CUDA Graph 后，DiT 和 VAE 的主要 kernel 调用次数没有变化，剩余共置开销表现为同一批 kernel 的 active duration 增长。
2. DiT kernel active time 增加155.991 ms（10.17%），VAE增加135.622 ms（7.91%）。
3. DiT 的 elementwise 类合计贡献约51.3%的绝对膨胀，相对减速普遍达到30%至45%，是最敏感的一组 kernel。
4. DiT FA 相对只慢5.36%，但因 baseline 达812.840 ms，仍增加43.531 ms，是第二大绝对贡献，不能忽略。
5. VAE主 XMMA卷积和两类 elementwise 各贡献约55.1 ms，分别约占 VAE 总膨胀40.6%。
6. 大型计算 kernel 的典型模式是“相对变化小、绝对贡献大”；elementwise/layout kernel 的典型模式是“单次短、调用多、相对变化大”。
7. 此模式与共享 L2、内存延迟、共享互连、功耗/频率变化对低算术强度 kernel 更敏感的假设相容，但 Nsys 只能显示时间结果，不能单独证明具体微架构原因。

## 8. 下一步 NCU 优先级

应按绝对贡献和相对敏感度共同选取 NCU 样本，而不是只选择最容易识别的 kernel。

### DiT

1. `elementwise_kernel`，grid `(14040,1,1)`：增加34.666 ms，慢32.10%；
2. `flash_fwd_kernel`，grid `(74,1,12)`：增加43.531 ms，慢5.36%；
3. `elementwise_kernel`，grid `(28080,1,1)`：增加14.084 ms，慢38.42%；
4. `unrolled_elementwise_kernel`，grid `(14040,1,1)`：增加14.018 ms，慢36.90%；
5. `vectorized_elementwise_kernel`，grid `(7020,1,1)`：增加12.839 ms，慢34.78%。

### VAE

1. 主 XMMA alignc4，主要两个 grid 形状：合计增加48.990 ms；
2. `elementwise_kernel`，grid `(599040,1,1)`：增加12.584 ms，慢13.53%；
3. `vectorized_elementwise_kernel`，grid `(149760,1,1)`：增加9.280 ms，慢16.36%；
4. `nchwToNhwcKernel`，grid `(75373,3,1)`：增加5.504 ms，慢18.01%。

NCU 中重点比较：

```text
Duration
SM Frequency
Long Scoreboard
Short Scoreboard
LG/MIO Throttle
Math Pipe Throttle
Eligible Warps Per Scheduler
L1/TEX Hit Rate
L2 Hit Rate
DRAM Throughput
```

其中 Nsys/CUDA Event 用于确认真实时间膨胀，NCU 用于解释特定 kernel 的 stall 和资源原因；不能用 NCU application replay 下的整轮 wall time 替代 Nsys 时间。
