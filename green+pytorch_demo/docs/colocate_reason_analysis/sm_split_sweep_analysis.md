# Ignore-SM-Coscheduling 下的 SM Split 扫描分析

## 1. 实验背景

Green Context 的 SM split 已加入：

```cpp
CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING
```

在当前170-SM GPU上，该 flag 将实际分配粒度从默认的8 SM降低到2 SM。随后固定总 SM数为170，对 DiT SM从104扫描到120，VAE使用剩余 SM。所有配置都启用 `ignore_sm_coscheduling=true`。

## 2. 完整结果

| DiT/VAE SM | DiT alone | VAE Graph alone | DiT colocated | VAE colocated | DiT slowdown | Parallel wall |
|---|---:|---:|---:|---:|---:|---:|
| 104 / 66 | 1729.26 ms | 1543.32 ms | 1904.25 ms | 1658.15 ms | 10.12% | 1904.67 ms |
| 106 / 64 | 1731.28 ms | 1594.18 ms | 1903.68 ms | 1707.43 ms | 9.96% | 1904.12 ms |
| 108 / 62 | 1718.76 ms | 1631.43 ms | 1879.84 ms | 1749.45 ms | 9.37% | 1880.11 ms |
| 110 / 60 | 1718.35 ms | 1684.19 ms | 1882.35 ms | 1802.66 ms | 9.54% | 1882.75 ms |
| **112 / 58** | **1615.80 ms** | 1733.83 ms | **1763.71 ms** | 1862.55 ms | 9.15% | **1862.63 ms** |
| 114 / 56 | 1525.95 ms | 1781.22 ms | 1662.84 ms | 1901.38 ms | 8.97% | 1901.53 ms |
| 116 / 54 | 1524.42 ms | 1838.62 ms | 1649.29 ms | 1957.30 ms | 8.19% | 1957.43 ms |
| 118 / 52 | 1512.93 ms | 1906.23 ms | 1637.51 ms | 2019.08 ms | 8.23% | 2019.24 ms |
| 120 / 50 | 1508.36 ms | 1968.53 ms | 1626.24 ms | 2083.15 ms | 7.82% | 2083.30 ms |

当前最优端到端配置是：

```text
DiT=112 SM
VAE=58 SM
parallel wall=1862.63 ms
```

## 3. DiT：性能平台和台阶

DiT alone可分为三个区域：

```text
104–110 SM：约1718–1731 ms
112 SM：   1616 ms
114–120 SM：约1508–1526 ms
```

| SM变化 | DiT alone变化 |
|---|---:|
| 104 → 106 | +0.12% |
| 106 → 108 | -0.72% |
| 108 → 110 | -0.02% |
| **110 → 112** | **-5.97%** |
| **112 → 114** | **-5.56%** |
| 114 → 116 | -0.10% |
| 116 → 118 | -0.75% |
| 118 → 120 | -0.30% |

新增2个 SM在多数位置几乎没有收益，但跨过112和114时，DiT时间突然下降约6%。

### 3.1 不能只用8-SM自然对齐解释

- 104也是8的倍数，却与106–110处于同一慢速平台；
- 114不是8的倍数，却比112再快5.56%；
- 120是8的倍数，但相比118只快0.30%。

Ignore-coscheduling改变分区的硬件层级和对称性，仍可能影响性能，但不能单独解释两个特定台阶。

### 3.2 固定 grid 的 wave/尾波阈值

固定 grid kernel 的时间不一定随 SM数连续缩放：

```text
waves ≈ grid blocks / 同时可驻留的总blocks
```

新增2个 SM若仍不足以容纳最后一组 blocks，kernel会继续需要相同数量的执行波次；跨过容量阈值后，最后一波会突然缩小或消失。这与104–110平台、112和114两次台阶相符。

### 3.3 114台阶的候选 kernel

此前 Nsys 中有一个高频 DiT `Kernel2`：

```text
grid=(152,3,1)
total blocks=456
block=(128,1,1)
calls/chunk=900
```

如果每个 SM可同时驻留4个 block：

```text
112 SM × 4 = 448 blocks
114 SM × 4 = 456 blocks
```

则112 SM时剩余8个 block进入尾波，114 SM时恰好铺满。该 kernel每个 chunk调用900次，短尾波也可能累计成明显时间。这是待 Nsys验证的具体候选，而非已经证明的结论。

## 4. VAE：随 SM数量平滑缩放

| VAE SM | Graph alone | Colocated | 绝对膨胀 | 相对膨胀 |
|---:|---:|---:|---:|---:|
| 66 | 1543.32 ms | 1658.15 ms | +114.83 ms | 7.44% |
| 64 | 1594.18 ms | 1707.43 ms | +113.25 ms | 7.10% |
| 62 | 1631.43 ms | 1749.45 ms | +118.02 ms | 7.23% |
| 60 | 1684.19 ms | 1802.66 ms | +118.47 ms | 7.03% |
| 58 | 1733.83 ms | 1862.55 ms | +128.71 ms | 7.42% |
| 56 | 1781.22 ms | 1901.38 ms | +120.16 ms | 6.75% |
| 54 | 1838.62 ms | 1957.30 ms | +118.68 ms | 6.46% |
| 52 | 1906.23 ms | 2019.08 ms | +112.85 ms | 5.92% |
| 50 | 1968.53 ms | 2083.15 ms | +114.61 ms | 5.82% |

VAE没有 DiT式时间台阶，Graph-alone时间基本随 SM数量单调、平滑变化。

### 4.1 为什么 VAE更平滑

VAE主要 kernel 的 grid和 waves都很大：

```text
主 XMMA grid=(1,12480,1)：约73.41 waves/SM
主 XMMA grid=(2,3120,1)：约36.71 waves/SM
Elementwise grid=(599040,1,1)：约293.65 waves/SM
```

几十到数百个 waves使增减2个 SM只会轻微改变总波次数和尾波比例。DiT中固定小 grid、高频、少 wave kernel更容易跨过离散容量阈值。

### 4.2 VAE绝对共置损失近似固定

除112/58略高外，VAE共置后的绝对增加大致稳定在113–120 ms，没有随 baseline或 SM数量同比例增长。

相对 slowdown从7.44%降至5.82%，主要有两种稀释效应：

1. VAE SM减少后 baseline变长，约115 ms固定损失的占比自然下降；
2. 高 DiT-SM配置中 DiT更早结束，VAE后部脱离 DiT单独执行。

例如120/50：

```text
DiT colocated=1626 ms
VAE colocated=2083 ms
```

DiT结束后，VAE还有约457 ms不再与 DiT重叠。因此相对 slowdown下降不能直接解释为共享竞争减弱。

## 5. 为什么112/58最优

110/60时 DiT决定关键路径：

```text
DiT=1882.35 ms
VAE=1802.66 ms
wall=1882.75 ms
```

112/58时关键路径切换到 VAE：

```text
DiT=1763.71 ms
VAE=1862.55 ms
wall=1862.63 ms
```

114/56以后，DiT已不在关键路径，继续给 DiT增加 SM只会让 VAE变慢。理论平衡点位于110/60和112/58之间，但实际分配粒度为2 SM，没有111/59组合，所以112/58成为离散最优。

## 6. 当前结论

1. Ignore-SM-coscheduling在当前 GPU上提供2-SM粒度分区。
2. VAE随50–66 SM平滑缩放，未出现明显性能台阶。
3. VAE共置绝对损失基本稳定在约115–120 ms。
4. DiT在112和114 SM处出现不能由线性缩放解释的性能台阶。
5. 完整扫描不支持只用8-SM自然对齐解释台阶。
6. 固定 grid kernel 的 wave/尾波容量阈值是当前最具体的候选原因。
7. `Kernel2 grid=(152,3,1)` 是114台阶的强候选，但需要 Nsys验证。
8. 当前最优为112/58，wall为1862.63 ms。

## 7. 后续 Nsys采集命令

重点采集：

```text
110/60：第一性能平台
112/58：第一次台阶后
114/56：第二次台阶后
```

目标是 kernel timeline和累计时间，不需要 GPU metrics或 CPU sampling。

进入容器：

```bash
docker exec -it cjh-Fastvideo-cu128 bash
cd /FastVideo/green+pytorch_demo
mkdir -p sm_split_nsys
```

### 7.1 110/60

```bash
CUDA_VISIBLE_DEVICES=1 nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-graph-trace=node \
    --sample=none \
    --force-overwrite=true \
    -o sm_split_nsys/dit_sm110 \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 110 \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 3
```

### 7.2 112/58

```bash
CUDA_VISIBLE_DEVICES=1 nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-graph-trace=node \
    --sample=none \
    --force-overwrite=true \
    -o sm_split_nsys/dit_sm112 \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 3
```

### 7.3 114/56

```bash
CUDA_VISIBLE_DEVICES=1 nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-graph-trace=node \
    --sample=none \
    --force-overwrite=true \
    -o sm_split_nsys/dit_sm114 \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 114 \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 3
```

### 7.4 一次性循环

```bash
mkdir -p sm_split_nsys

for dit_sms in 110 112 114; do
    CUDA_VISIBLE_DEVICES=1 nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-graph-trace=node \
        --sample=none \
        --force-overwrite=true \
        -o "sm_split_nsys/dit_sm${dit_sms}" \
        python vae_dit_testbench_cuda_graph.py \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --warmup-iters 3 \
            --profile-iters 3
done
```

生成：

```text
sm_split_nsys/dit_sm110.nsys-rep
sm_split_nsys/dit_sm112.nsys-rep
sm_split_nsys/dit_sm114.nsys-rep
```

## 8. Nsys后续分析方法

1. 丢弃三次 warmup，只统计三次正式计时的 `DiT_chunk`；
2. 限定 DiT Green Context stream，避免把 VAE Graph节点算入 DiT；
3. 按 `short kernel name + grid + block` 聚合；
4. 计算每个 signature 的 calls/chunk、total time/chunk和average duration/call；
5. 分别按 `110 total - 112 total` 和 `112 total - 114 total` 排序。

优先检查：

```text
Kernel2 grid=(152,3,1), block=(128,1,1)
flash_fwd_kernel grid=(74,1,12)
elementwise_kernel grid=(14040,1,1)
elementwise_kernel grid=(28080,1,1)
unrolled/vectorized elementwise
vectorized_layer_norm_kernel
```

如果 `Kernel2 (152,3,1)` 在112→114时单次或累计时间大幅下降、调用次数仍为900，就支持尾波阈值推断。

如果许多完全不同 grid 的 kernel都按相同比例下降，则应转向检查 GPU频率、SM物理分区拓扑、GPC/TPC分布和 ignore-coscheduling带来的层级差异。
