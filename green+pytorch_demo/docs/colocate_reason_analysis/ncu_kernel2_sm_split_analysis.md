# Kernel2 在 112/114 SM 台阶处的 NCU 分析

## 1. 分析目标

Nsys 已确认，DiT 从 112 SM 增加到 114 SM 时约 90 ms 的累计收益主要来自
两个 CUTLASS `Kernel2` signature。本文使用 Nsight Compute 继续判断其
duration 突变来自：

- CTA wave 或尾波变化；
- occupancy、register 或 shared-memory residency 变化；
- scheduler stall、缓存或显存效率变化；
- Green Context 分区在 TPC/GPC 拓扑上的离散映射变化。

本文使用以下报告：

```text
green+pytorch_demo/ncu_kernel2_a_sm112.ncu-rep
green+pytorch_demo/ncu_kernel2_a_sm114.ncu-rep
green+pytorch_demo/ncu_kernel2_b_sm112.ncu-rep
green+pytorch_demo/ncu_kernel2_b_sm114.ncu-rep

green+pytorch_demo/ncu_kernel2_b_sweep/kernel2_b_sm110.ncu-rep
green+pytorch_demo/ncu_kernel2_b_sweep/kernel2_b_sm112.ncu-rep
green+pytorch_demo/ncu_kernel2_b_sweep/kernel2_b_sm114.ncu-rep
green+pytorch_demo/ncu_kernel2_b_sweep/kernel2_b_sm116.ncu-rep
```

所有采集均限定在 `DiT_eager_alone/DiT_chunk/`，并启用
`--ignore-sm-coscheduling`，所以比较的是相同 DiT 工作量在不同实际 Green
Context SM 数量下的行为。

## 2. 目标 kernel

### 2.1 Kernel A

```text
CUTLASS specialization:
cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8

grid                         = (152, 3, 1)
Grid Size                    = 456 CTA
block                        = (128, 1, 1)
registers/thread             = 230
dynamic shared memory/block = 81.92 KB
```

### 2.2 Kernel B

```text
CUTLASS specialization:
cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8

grid                         = (152, 2, 1)
Grid Size                    = 304 CTA
block                        = (256, 1, 1)
registers/thread             = 222
dynamic shared memory/block = 73.73 KB
```

## 3. Kernel A：456 CTA 的普通尾波阈值

### 3.1 112/114 对比

| 指标 | 112 SM | 114 SM | 变化 |
|---|---:|---:|---:|
| Duration | 182.78 us | 149.31 us | -18.31% |
| Elapsed Cycles | 531,920 | 433,968 | -18.42% |
| SM Frequency | 2.91 GHz | 2.90 GHz | 基本不变 |
| SM Active Cycles | 433,735 | 426,061 | -1.77% |
| Achieved Occupancy | 8.33% | 8.33% | 不变 |
| L2 Hit Rate | 98.88% | 98.91% | 不变 |
| Warp Cycles/Instruction | 10.52 | 10.52 | 不变 |

NCU 给出的 residency 限制为：

```text
Block Limit Registers  = 2
Block Limit Shared Mem = 1
Block Limit Warps      = 12
```

因此 shared memory 是主限制，每个 SM 同时只能驻留一个该 kernel 的 CTA。
分区内所需 CTA wave 为：

```text
112 SM: ceil(456 / 112) = 5 waves
114 SM: ceil(456 / 114) = 4 waves
```

112 SM 的第五波只有：

```text
456 - 112 * 4 = 8 CTA
```

而 114 SM 恰好满足：

```text
456 = 114 * 4
```

理想时间比为：

```text
4 / 5 = 0.800
```

NCU 实测为：

```text
149.31 / 182.78 = 0.817
```

因此 Kernel A 的台阶可以由 114 SM 消除一个严重欠填充的尾波完整解释。
这也修正了早期用“每 SM 同时驻留 4 CTA”推导 448/456 的错误：NCU 已证明
该 kernel 实际受 shared memory 限制，只能同时驻留 1 CTA/SM。

## 4. Kernel B：57 TPC 处的离散阈值

### 4.1 110–116 SM sweep

| DiT SM | TPC | Duration | Elapsed Cycles | SM Active Cycles |
|---:|---:|---:|---:|---:|
| 110 | 55 | 约 1.19 ms | 3,491,490 | 2,411,421 |
| 112 | 56 | 约 1.20 ms | 3,491,535 | 2,369,063 |
| 114 | 57 | 801.44 us | 2,333,162 | 2,326,470 |
| 116 | 58 | 801.28 us | 2,333,844 | 2,286,881 |

结果形成两个非常稳定的平台：

```text
110 SM ~= 112 SM
114 SM ~= 116 SM
```

112 到 114 SM 的 elapsed-cycle 比为：

```text
2,333,162 / 3,491,535 = 0.6683
```

非常接近 `2/3`。这排除了“每增加 2 SM 都带来连续加速”的解释，证明分区
在 `114 SM / 57 TPC` 处跨过了一个离散调度阈值。

### 4.2 单 CTA 执行特征没有变化

| 指标 | 110 SM | 112 SM | 114 SM | 116 SM |
|---|---:|---:|---:|---:|
| SM Frequency | 2.92 GHz | 2.91 GHz | 2.91 GHz | 2.91 GHz |
| Achieved Occupancy | 16.67% | 16.66% | 16.67% | 16.67% |
| One or More Eligible | 8.70% | 8.70% | 8.70% | 8.71% |
| Warp Cycles/Instruction | 22.98 | 22.97 | 22.97 | 22.97 |
| L2 Hit Rate | 90.41% | 90.46% | 90.18% | 90.14% |

四种配置中的 launch 资源也完全相同：

```text
Grid Size                    = 304 CTA
Block Size                   = 256 threads
Registers Per Thread         = 222
Dynamic Shared Memory/Block  = 73.73 KB
Block Limit Registers        = 1
Block Limit Shared Mem       = 1
Block Limit Warps            = 6
```

因此突变不是以下因素造成的：

- GPU 时钟变化；
- occupancy 或 residency 改善；
- register/shared-memory 使用量变化；
- warp scheduler 或单 CTA stall 行为改善；
- L2 命中率改善。

114 SM 后显示更高的 memory/compute throughput，是相同工作在更短 wall time
内完成的结果，不能反过来解释为访存带宽导致加速。

### 4.3 Active cycles 平滑，但 makespan 突变

`SM Active Cycles` 随 SM 数增加平滑下降，每增加 2 SM 约下降 1.7%；但
`Elapsed Cycles` 在 112 到 114 SM 时突然减少约 33%。这说明单个 CTA 完成
相同计算所需的活跃工作量没有突变，减少的是整体 makespan 中的离散调度
阶段或尾部。

这是“对齐或排布阈值”的典型特征。

### 4.4 为什么普通 CTA wave 模型解释不了 Kernel B

NCU 显示该 kernel 同样只能驻留 1 CTA/SM。普通平坦模型给出：

```text
ceil(304 / 110) = 3
ceil(304 / 112) = 3
ceil(304 / 114) = 3
ceil(304 / 116) = 3
```

四种分区都应需要 3 个 CTA wave，但实测 makespan 却呈现稳定的
`3 -> 2` 比例。因此 Kernel B 消失的“第三份时间”不是普通意义上的全局
CTA wave。

当前最合理的候选是二维 CUTLASS grid：

```text
grid = (152, 2, 1)
```

经过 threadblock swizzle/rasterization 后，与 Green Context 的 TPC/GPC
资源集合形成离散映射。可能涉及：

- 不同 grid 维度或边界 tile 的工作量不均匀；
- CUTLASS tile 的 raster 顺序；
- CTA 在 TPC/GPC 间的分发与尾部排列；
- `56 -> 57 TPC` 时 Green Context 分区拓扑发生有利变化。

这些是与现有证据一致的候选机制，但 NCU 当前聚合指标还不能区分它们，不能
把其中任何一项写成已经证实的根因。

## 5. `Waves Per SM` 在 Green Context 下的归一化问题

NCU 对 Kernel A 报告：

```text
Waves Per SM = 2.68
```

对 Kernel B 报告：

```text
Waves Per SM = 1.79
```

它们分别严格对应：

```text
456 / 170 = 2.682
304 / 170 = 1.788
```

说明 NCU 的该字段仍使用整张 GPU 的 170 个物理 SM 归一化，而不是报告中
同时显示的 Green Context `# SMs = 110/112/114/116`。

因此在 Green Context 实验中，不能直接使用 `Waves Per SM` 判断实际 wave
数量。对于可使用平坦模型的 kernel，应根据以下信息手工计算：

```text
实际 Grid Size
Green Context # SMs
Block Limit Registers
Block Limit Shared Mem
Block Limit Warps
```

## 6. 与 Nsys 累计结果的对应

NCU 的单次 duration 与 Nsys 聚合结果一致：

- Kernel A 从约 181 us 降到约 147–149 us，接近 `5 -> 4`；
- Kernel B 从约 1.19 ms 降到约 0.80 ms，接近 `3 -> 2`；
- 两者的调用次数没有变化；
- 两个 signature 合计解释了 112 到 114 SM 时约 90 ms 的绝大多数累计
  kernel-time 收益。

NCU application replay 会扰动绝对时间，因此累计端到端时间仍以 Nsys 和
CUDA Event 为准；本轮 NCU 的价值在于定位 residency、occupancy、stall 和
memory 指标是否发生结构性变化。

## 7. 最终结论

1. Kernel A 的台阶已经解释清楚：456 CTA、1 CTA/SM；112 SM 需要 5 波，
   114 SM 恰好 4 波，消除了仅含 8 CTA 的尾波。
2. Kernel B 在 110/112 和 114/116 形成两个稳定平台，证明阈值严格出现在
   `114 SM / 57 TPC` 附近，而非连续 SM scaling。
3. Kernel B 的普通 `ceil(CTA/SM)` wave 数在四种配置下都是 3，因此其
   `3 -> 2` duration 比例来自更复杂的 tile/swizzle 与硬件拓扑映射，而不是
   简单全局 CTA wave。
4. 两个 kernel 的 occupancy、资源限制、scheduler 特征、L2 hit rate 和
   时钟均未发生能够解释台阶的变化。
5. 当前证据将主因限定为 CTA 整体调度、对齐或尾部 makespan，而不是单 CTA
   内部计算或访存效率。
6. NCU 的 `Waves Per SM` 使用全卡 170 SM 归一化，不适用于直接判断 Green
   Context 分区内的 wave 数。
