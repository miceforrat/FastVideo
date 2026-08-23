# DiT/VAE 共置关键 Kernel 的 NCU 结果分析

## 1. 分析目标与报告

本文分析 `ncu_profile_cmd.md` 中规划的六组关键 kernel。每个目标分别采集 alone 和 colocated，共十二份 Nsight Compute 报告：

```text
ncu_dit_01_flash_alone.ncu-rep
ncu_dit_01_flash_colocated.ncu-rep
ncu_dit_02_elementwise_14040_alone.ncu-rep
ncu_dit_02_elementwise_14040_colocated.ncu-rep
ncu_dit_03_elementwise_28080_alone.ncu-rep
ncu_dit_03_elementwise_28080_colocated.ncu-rep
ncu_vae_01_xmma_1x12480_alone.ncu-rep
ncu_vae_01_xmma_1x12480_colocated.ncu-rep
ncu_vae_02_xmma_2x3120_alone.ncu-rep
ncu_vae_02_xmma_2x3120_colocated.ncu-rep
ncu_vae_03_elementwise_599040_alone.ncu-rep
ncu_vae_03_elementwise_599040_colocated.ncu-rep
```

采集公共条件为：

```text
DiT Green Context: 112 SM
VAE Green Context: 58 SM
replay mode: application
CUDA Graph profiling: node
clock control: none
cache control: none
```

NCU 用于分析单个 kernel invocation 的微架构行为。真实的累计时间与 wall time 仍以 Nsys/CUDA Event 为准。

## 2. 目标核验

十二份报告均只包含一个有效 kernel 数据行。每对 alone/colocated 均命中相同的：

```text
kernel name
grid dimensions
block dimensions
registers per thread
Green Context SM count
```

目标没有因 `--launch-skip` 失效而抓错。

| 目标 | Grid | Block | SM | 核验结果 |
|---|---|---|---:|---|
| DiT FA | `(74,1,12)` | `(128,1,1)` | 112 | 一致 |
| DiT elementwise | `(14040,1,1)` | `(128,1,1)` | 112 | 一致 |
| DiT elementwise | `(28080,1,1)` | `(128,1,1)` | 112 | 一致 |
| VAE主 XMMA | `(1,12480,1)` | `(128,1,1)` | 58 | 一致 |
| VAE主 XMMA | `(2,3120,1)` | `(128,1,1)` | 58 | 一致 |
| VAE elementwise | `(599040,1,1)` | `(128,1,1)` | 58 | 一致 |

## 3. 总览

| Kernel | NCU单次 Duration变化 | L2 hit变化 | Long Scoreboard变化 | 当前判断 |
|---|---:|---:|---:|---|
| DiT FA | +0.36% | -0.11 pp | +5.6%，但绝对值极小 | 本次调用基本不受影响 |
| DiT elementwise `(14040)` | **+58.65%** | **-48.84 pp** | **+128.0%** | 强烈的 L2/DRAM竞争 |
| DiT elementwise `(28080)` | **+7.85%** | **-12.07 pp** | **+10.1%** | 明显访存竞争，并伴随降频 |
| VAE XMMA `(1,12480)` | +0.04% | 基本不变 | 基本不变 | 本次调用不受影响 |
| VAE XMMA `(2,3120)` | +0.06% | 基本不变 | 基本不变 | 本次调用不受影响 |
| VAE elementwise `(599040)` | -0.06% | 基本不变 | 基本不变 | 本次调用不受影响 |

最重要的新证据是：

> VAE 共置会显著破坏部分 DiT elementwise kernel 的 L2局部性，使请求转向 DRAM，增加 Long Scoreboard，并减少 eligible warp。这是当前最直接的“SM之外共享存储资源竞争”证据。

## 4. DiT Elementwise `(14040,1,1)`

### 4.1 Kernel信息

```text
operation: BF16 add
grid: (14040,1,1)
block: (128,1,1)
registers/thread: 22
SM count: 112
```

### 4.2 时间与存储层级

| 指标 | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| Duration | 18.336 | 29.088 | **+58.65%** |
| SM Frequency | 2.8997 GHz | 2.8998 GHz | 基本不变 |
| L1/TEX Hit Rate | 57.34% | 58.16% | +0.82 pp |
| L2 Hit Rate | **99.90%** | **51.06%** | **-48.84 pp** |
| DRAM Throughput | 1.64% | **56.65%** | **+55.01 pp** |
| Memory Throughput | 29.88% | 56.65% | +26.77 pp |

该 kernel 的证据链非常清晰：

```text
VAE同时运行
  → DiT elementwise 的 L2 hit 从99.9%降到51.1%
  → 大量请求转向DRAM
  → DRAM利用率从1.6%升到56.7%
  → 数据依赖等待增长
  → kernel慢58.6%
```

### 4.3 Warp stall与调度器

| 指标 | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| Warp Cycles Per Issued Instruction | 18.10 | 31.06 | **+71.6%** |
| Long Scoreboard | 11.41 | 26.01 | **+128.0%** |
| Eligible Warps/Scheduler | 1.428 | 0.553 | **-61.3%** |
| Issue Active | 0.58 | 0.35 | **-39.7%** |
| Active Warps/Scheduler | 10.50 | 10.91 | +3.9% |

该 kernel 并不是因为没有足够 active warp：每个 scheduler 有约10.5至10.9个 active warp。问题是大量 active warp 同时等待长延迟数据，因此 eligible warp 数量显著下降。

SM频率没有下降，所以58.65%的膨胀不能用降频解释。

## 5. DiT Elementwise `(28080,1,1)`

### 5.1 Kernel信息

```text
operation: float copy/conversion
grid: (28080,1,1)
block: (128,1,1)
registers/thread: 20
SM count: 112
```

### 5.2 关键指标

| 指标 | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| Duration | 66.816 | 72.064 | **+7.85%** |
| SM Frequency | 2.9124 GHz | 2.8789 GHz | **-1.15%** |
| L1/TEX Hit Rate | 4.12% | 4.51% | +0.40 pp |
| L2 Hit Rate | 88.37% | 76.31% | **-12.07 pp** |
| DRAM Throughput | 21.53% | 44.74% | **+23.21 pp** |
| Warp Cycles Per Issued Instruction | 89.93 | 97.03 | +7.90% |
| Long Scoreboard | 66.50 | 73.21 | **+10.08%** |
| Eligible Warps/Scheduler | 0.150 | 0.141 | -5.51% |
| Drain | 10.88 | 11.88 | +9.10% |
| MIO Throttle | 0.889 | 1.059 | +19.1% |

这个 kernel 在 alone 时已经明显 memory-bound：Long Scoreboard 高达66.5 cycles。共置后 L2 hit继续下降、DRAM压力增加、Long Scoreboard上升，同时 SM频率下降约1.15%。

因此其7.85%减速更可能由两部分共同造成：

```text
访存延迟增加 + 小幅降频
```

## 6. DiT FlashAttention

### 6.1 Kernel信息

```text
kernel: flash_fwd_kernel
grid: (74,1,12)
block: (128,1,1)
registers/thread: 174
SM count: 112
```

### 6.2 关键指标

| 指标 | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| Duration | 5.3023 | 5.3214 | +0.36% |
| SM Frequency | 2.9255 GHz | 2.9148 GHz | -0.37% |
| L2 Hit Rate | 98.53% | 98.42% | -0.11 pp |
| DRAM Throughput | 2.33% | 2.51% | +0.18 pp |
| Warp Cycles Per Issued Instruction | 14.516 | 14.517 | 基本不变 |
| Long Scoreboard | 0.0540 | 0.0570 | 绝对值仍很小 |
| Math Pipe Throttle | 7.781 | 7.779 | 基本不变 |
| Wait | 3.248 | 3.248 | 基本不变 |

本次 FA invocation 主要受 Math Pipe Throttle、Wait和Barrier影响，而不是 global-memory Long Scoreboard。共置后各项基本相同，0.36%的时间增长也接近0.37%的频率下降。

但是 Nsys 对300次 FA 的累计结果为：

```text
812.840 ms → 856.371 ms，+5.36%
```

因此当前结果只能表述为：

> `launch-skip=0` 命中的第一次 FA 调用没有明显受到 VAE影响。

它不能证明300次 FA都不受影响。后续调用可能处于更强的并发区间。

## 7. VAE 主 XMMA `(1,12480,1)`

```text
registers/thread: 242
SM count: 58
waves/SM: 73.41
```

| 指标 | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| Duration | 17.2617 | 17.2682 | +0.04% |
| SM Frequency | 2.9101 GHz | 2.9082 GHz | -0.07% |
| L2 Hit Rate | 91.0279% | 91.0277% | 基本不变 |
| DRAM Throughput | 6.0666% | 6.0618% | 基本不变 |
| Warp Cycles Per Issued Instruction | 9.3791 | 9.3785 | 基本不变 |
| Math Pipe Throttle | 4.4472 | 4.4473 | 基本不变 |
| Wait | 3.7515 | 3.7515 | 基本不变 |

当前采集的调用没有可见的共置影响。

但 Nsys 对该 signature 的21次调用累计结果为：

```text
511.240 ms → 537.891 ms，+5.21%
```

## 8. VAE 主 XMMA `(2,3120,1)`

```text
registers/thread: 242
SM count: 58
waves/SM: 36.71
```

| 指标 | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| Duration | 16.8712 | 16.8806 | +0.06% |
| SM Frequency | 2.9094 GHz | 2.9077 GHz | -0.06% |
| L2 Hit Rate | 95.3694% | 95.3643% | 基本不变 |
| DRAM Throughput | 3.1134% | 3.1073% | 基本不变 |
| Warp Cycles Per Issued Instruction | 9.8822 | 9.8818 | 基本不变 |
| Math Pipe Throttle | 4.7950 | 4.7952 | 基本不变 |
| Wait | 3.9788 | 3.9788 | 基本不变 |

当前采集的调用同样没有可见影响。

Nsys 对该 signature 的21次调用累计结果为：

```text
503.280 ms → 525.618 ms，+4.44%
```

## 9. VAE Elementwise `(599040,1,1)`

### 9.1 Kernel信息

```text
operation: FP32 add
grid: (599040,1,1)
block: (128,1,1)
registers/thread: 22
SM count: 58
```

### 9.2 关键指标

| 指标 | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| Duration | 1.0453 | 1.0446 | -0.06% |
| SM Frequency | 2.8993 GHz | 2.9053 GHz | +0.21% |
| L1/TEX Hit Rate | 54.7810% | 54.7912% | 基本不变 |
| L2 Hit Rate | 50.0143% | 50.0160% | 基本不变 |
| DRAM Throughput | 66.0888% | 66.1405% | 基本不变 |
| Warp Cycles Per Issued Instruction | 22.5042 | 22.5041 | 基本不变 |
| Long Scoreboard | 17.2874 | 17.2648 | 基本不变 |
| Eligible Warps/Scheduler | 0.7741 | 0.7737 | 基本不变 |

该调用自身明显 memory-bound，但 alone/colocated 几乎完全相同。

Nsys 对该 signature 的87次调用累计结果为：

```text
93.029 ms → 105.613 ms，+13.53%
```

因此当前这一份 NCU 报告不能代表该 signature 的全部87次调用。

## 10. 为什么 VAE单次 NCU 与累计 Nsys不同

当前每种 signature 都使用：

```text
--launch-count 1
```

因此 NCU 只采集由 `--launch-skip` 选中的一次调用，而 Nsys统计的是同一 signature 在整个 chunk/graph replay 中的累计时间。

VAE Graph 的执行顺序为：

```text
CPU提交 VAE graph replay
  → VAE Graph前部节点立即开始运行
CPU随后开始逐算子提交 eager DiT
  → DiT负载逐渐填入另一 Green Context
```

所以 VAE Graph前部节点可能在 DiT形成充分 GPU负载前执行，受到的干扰较弱；Graph中后部节点才更可能与 DiT充分重叠。

类似地，DiT的第一次 FA和第一次指定 elementwise也不一定代表该 signature 在所有 denoise step、block和KV update位置上的平均行为。

从本轮结果可以看到这种异质性：

| Signature | Nsys累计膨胀 | 当前NCU单次膨胀 |
|---|---:|---:|
| DiT FA | +5.36% | +0.36% |
| DiT elementwise `(14040)` | +32.10% | +58.65% |
| DiT elementwise `(28080)` | +38.42% | +7.85% |
| VAE XMMA `(1,12480)` | +5.21% | +0.04% |
| VAE XMMA `(2,3120)` | +4.44% | +0.06% |
| VAE elementwise `(599040)` | +13.53% | -0.06% |

这些差异不表示 NCU 与 Nsys互相矛盾，而是说明同名、同形状 kernel 在不同时间位置受到的共置干扰不同。

## 11. 当前能够确认的结论

1. 至少部分 DiT elementwise 的减速直接来自共享存储层级竞争。
2. 最强样本中，L2 hit从99.9%降到51.1%，DRAM throughput从1.6%升到56.7%，Long Scoreboard增长128%，kernel慢58.6%。
3. 第二种 DiT elementwise 同样出现 L2 hit下降、DRAM流量增长和 Long Scoreboard增长，并伴随约1.15%的降频。
4. CUDA Graph消除了 VAE逐算子 host/driver提交锁争用，但不会隔离 L2、DRAM、内存互连、功耗和频率等共享GPU资源。
5. 当前采集的第一次 FA调用没有明显存储竞争，不能解释其全部300次调用累计增加43.53 ms的原因。
6. 当前采集的三个 VAE调用均没有明显变化，但不能据此推翻 Nsys中 VAE signature 的累计膨胀。
7. 当前证据更符合“共置影响具有明显时间位置和 kernel类型依赖”，而不是所有 kernel按固定比例均匀变慢。

## 12. 后续如果继续分析

后续不应再只选择每种 signature 的第一次调用。应从 Nsys 中对同一 signature 的逐调用膨胀进行排序，再分别选择：

```text
Graph/chunk前部调用
中位数调用
膨胀最大的调用
Graph/chunk后部调用
```

然后用 NCU比较相同位置的 alone/colocated，才能进一步回答：

1. FA的累计43.53 ms主要来自哪些调用；
2. VAE主 XMMA的4%至5%累计膨胀发生在 Graph什么位置；
3. VAE elementwise的13.53%累计膨胀是否同样来自 L2/DRAM竞争；
4. 共置干扰是否随 DiT提交进度和 GPU overlap强度变化。

在此之前，最稳妥的阶段性表述是：

> CUDA Graph解决了主要的 host提交锁争用，但剩余 GPU侧膨胀具有明显 kernel类型与时间位置依赖。其中部分 DiT elementwise 已获得直接的 L2命中率下降、DRAM压力上升和 Long Scoreboard增长证据；FA与当前采集的 VAE前部样本则没有表现出相同程度的单次膨胀。
