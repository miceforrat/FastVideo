# 110/112/114 SM 性能台阶的 Nsys 定位

## 1. 分析目标

在启用

```text
CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING
```

后，DiT 分区可以按 2 SM 粒度调整。完整扫描显示 DiT 在
`110 -> 112` 和 `112 -> 114` 时各出现一次约 6% 的非线性加速。
本文使用以下三个报告定位两次台阶分别由哪些 kernel 产生：

```text
green+pytorch_demo/sm_split_nsys/dit_sm110.nsys-rep
green+pytorch_demo/sm_split_nsys/dit_sm112.nsys-rep
green+pytorch_demo/sm_split_nsys/dit_sm114.nsys-rep
```

## 2. 统计口径

- 只统计 DiT 所在的 Green Context stream；
- 排除 warmup；
- 对每份报告分别统计 3 个 DiT-alone chunk 和 3 个 colocated chunk；
- 使用 `kernel name + grid + block` 作为精确 signature 聚合；
- 三种配置均观察到 42 个 DiT kernel signature，调用次数保持一致；
- 因而下文比较的是相同工作量在不同 SM 分区下的执行时间，不是调用次数变化。

`kernel active time` 是目标 DiT stream 上各 kernel duration 之和，不包含
CPU launch gap，也不等同于端到端 wall time。

## 3. 总体结果

### 3.1 DiT alone

| DiT SM | Kernel active time |
|---:|---:|
| 110 | 1626.538 ms |
| 112 | 1529.851 ms |
| 114 | 1438.339 ms |

因此：

| 跨度 | Active time 减少 |
|---|---:|
| 110 -> 112 | 96.687 ms |
| 112 -> 114 | 91.512 ms |

### 3.2 DiT colocated with VAE Graph

| DiT SM | Kernel active time |
|---:|---:|
| 110 | 1786.511 ms |
| 112 | 1675.726 ms |
| 114 | 1570.357 ms |

因此：

| 跨度 | Active time 减少 |
|---|---:|
| 110 -> 112 | 110.785 ms |
| 112 -> 114 | 105.369 ms |

两次台阶在 alone 和 colocated 中都存在，说明其根源是 DiT kernel 在离散
SM 容量阈值处的执行效率变化，而不是只有共置时才产生的新工作。

## 4. 第一次台阶：110 -> 112 几乎完全来自 FlashAttention

DiT-alone 中，FlashAttention signature 的结果为：

```text
grid  = (74, 1, 12)
block = (128, 1, 1)
calls = 300
```

| DiT SM | FA 累计时间 | 平均每次调用 |
|---:|---:|---:|
| 110 | 908.6053 ms | 3028.684 us |
| 112 | 811.4463 ms | 2704.821 us |

FA 累计减少 `97.1591 ms`，而全部 DiT kernel active time 减少
`96.687 ms`。差异说明其余 kernel 的变化合计略微抵消了 FA 的收益；就第一
次台阶而言，FA 可以解释约 100% 的净加速。

这个 FA kernel 每次调用共有：

```text
74 * 12 = 888 blocks
```

若按每个 SM 在该 kernel 中有效推进一个 block 的简化视角：

```text
888 / 110 = 8.073  -> 至少需要第 9 个调度阶段
888 / 112 = 7.929  -> 可以在 8 个阶段内完成
```

理想时间比为：

```text
8 / 9 = 0.8889
```

实际平均调用时间比为：

```text
2704.821 / 3028.684 = 0.8931
```

两者非常接近。这强烈支持：`112 SM` 恰好使 888-block 的 FA kernel
消除一个尾部执行阶段，从而产生第一次性能台阶。

在 colocated 区间中，FA 从 110 到 112 SM 减少 `100.146 ms`，占全部
`110.785 ms` 收益的 `90.4%`。因此共置不会改变第一次台阶的主导 kernel。

## 5. 第二次台阶：112 -> 114 由两个 Kernel2 signature 主导

第二次台阶并非来自 FA。DiT-alone 中，FA 时间为：

```text
112 SM: 811.4463 ms
114 SM: 811.4705 ms
```

二者基本相同。真正发生变化的是 Nsys 中显示为 `Kernel2` 的 kernel family：

```text
112 SM: 469.6203 ms
114 SM: 379.4320 ms
减少:    90.1884 ms
```

该 family 解释了全部 `91.512 ms` active-time 收益的 `98.6%`。

其中贡献最大的两个精确 signature 为：

### 5.1 Kernel2，grid=(152,2,1)，block=(256,1,1)

```text
calls = 150
```

| DiT SM | 累计时间 | 平均每次调用 |
|---:|---:|---:|
| 112 | 178.8915 ms | 1192.610 us |
| 114 | 119.5555 ms | 797.037 us |

累计减少 `59.3359 ms`，平均调用时间比为 `0.6683`，接近 `2/3`。

### 5.2 Kernel2，grid=(152,3,1)，block=(128,1,1)

```text
calls = 900
```

| DiT SM | 累计时间 | 平均每次调用 |
|---:|---:|---:|
| 112 | 162.6239 ms | 180.693 us |
| 114 | 131.9954 ms | 146.662 us |

累计减少 `30.6285 ms`，平均调用时间比为 `0.8117`，接近 `4/5`。

两个 signature 合计减少 `89.9644 ms`，可解释第二次台阶全部
`91.512 ms` 收益的 `98.3%`。

在 colocated 区间中，这两个 signature 分别减少约 `60.368 ms` 和
`30.962 ms`；整个 Kernel2 family 减少 `92.390 ms`，占全部
`105.369 ms` 收益的 `87.7%`。其余收益主要来自 FA 和 elementwise，量级
明显更小。

## 6. 能证明什么，暂时不能证明什么

### Nsys 直接证明的事实

1. `110 -> 112` 的 DiT 台阶由 888-block 的 FA signature 主导；
2. FA 的实测时间比与消除第 9 个阶段后的 `8/9` 非常接近；
3. `112 -> 114` 的台阶与 FA 无关，主要来自两个固定的 Kernel2
   signature；
4. 第二次台阶中两个 Kernel2 的调用数没有变化，变化来自单次 kernel
   duration；
5. 同样的主导项在 alone 和 colocated 中均可见。

### 基于结果的强推断

两个 Kernel2 的时间比分别接近 `2/3` 和 `4/5`，符合跨过离散执行波次或
调度阶段阈值的特征。因此 `114 SM` 很可能使这两个 kernel 各减少一个有效
执行阶段。

### Nsys 尚不能单独证明的细节

不能再沿用简单的：

```text
112 * 4 = 448
114 * 4 = 456
```

来声称 `grid=(152,3,1)` 恰好从尾波变为铺满。`grid.y` 的含义、CTA 映射、
每 SM 实际 residency，以及 Kernel2 内部的 cuBLAS/CUTLASS 调度策略不能只
从 Nsys timeline 确定。若要解释精确的内部容量公式，需要继续采集对应
signature 的 NCU `LaunchStats`、occupancy 和 scheduler 指标，必要时结合
kernel 来源或反汇编。

## 7. 最终结论

DiT 的两个台阶不是一个统一的“112/114 SM 更快”现象，而是两个不同
kernel family 分别跨过容量阈值：

```text
110 -> 112: FlashAttention，888 blocks，约减少 97 ms
112 -> 114: 两个 Kernel2 signature，合计约减少 90 ms
```

这也解释了为什么增加 2 SM 的收益高度不连续：只有当特定高频 kernel
越过离散 wave/调度阶段边界时才会出现明显收益。

从端到端共置效果看，`112/58` 仍是当前扫描中的最优点。虽然给 DiT 增加到
114 SM 能再次消除约 90 ms 的 DiT kernel 时间，但此时关键路径已经转移到
VAE；VAE 从 58 SM 减少到 56 SM 后变慢更多，使 parallel wall time 从约
`1862.63 ms` 上升到 `1901.53 ms`。
