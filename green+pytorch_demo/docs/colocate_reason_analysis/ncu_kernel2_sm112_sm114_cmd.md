# 112/114 SM 下两个 Kernel2 的 NCU 采集命令

## 1. 采集目的

Nsys 已确认，DiT 从 112 SM 增加到 114 SM 后，约 90 ms 的主要收益来自
两个 CUTLASS `Kernel2` signature。NCU 采集用于继续判断收益是否来自：

- CTA wave 或尾波减少；
- occupancy、寄存器或 shared memory 限制变化；
- scheduler stall、L2/DRAM 行为变化。

本实验只比较 `DiT_eager_alone`，不引入 VAE 共置竞争。原因是此次目标是
解释 112/114 SM 本身的性能台阶；Nsys 已经证明相同台阶在 alone 和
colocated 中都存在。

## 2. 两个目标 kernel

### Kernel A

```text
Nsys short name: Kernel2
grid:  (152, 3, 1)
block: (128, 1, 1)
registers/thread: 230
dynamic shared memory: 81920 bytes
```

完整 demangled name：

```text
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(T1::Params)
```

### Kernel B

```text
Nsys short name: Kernel2
grid:  (152, 2, 1)
block: (256, 1, 1)
registers/thread: 222
dynamic shared memory: 73728 bytes
```

完整 demangled name：

```text
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)
```

## 3. 前置条件

进入容器：

```bash
docker exec -it cjh-Fastvideo-cu128 bash
cd /FastVideo/green+pytorch_demo
```

确保物理 GPU 1 没有其他任务运行：

```bash
nvidia-smi
```

脚本必须已有以下正式计时 NVTX 层次：

```text
DiT_eager_alone/DiT_chunk/
```

本轮必须传入 `--ignore-sm-coscheduling`。否则 114 SM 可能按默认粒度被
round 到其他实际 SM 数量，实验就不再是 112 对 114。

## 4. 推荐方案：运行两次，同时采两个 kernel

### 4.1 公共参数

先在当前 shell 中定义：

```bash
NCU_COMMON=(
    --replay-mode application
    --app-replay-mode relaxed
    --check-exit-code no
    --nvtx
    --nvtx-include "DiT_eager_alone/DiT_chunk/"
    --kernel-name-base function
    --launch-skip 2
    --launch-count 2
    --section WarpStateStats
    --section SchedulerStats
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --section LaunchStats
    --section Occupancy
    --clock-control none
    --cache-control none
    --force-overwrite
)

KERNEL2_REGEX='regex:.*cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_(64x256_32x4|128x256_32x3)_tn_align8.*'
```

这里使用组合正则，同时匹配两个完整 CUTLASS specialization，而不是模糊
匹配所有 `Kernel2`。

根据 112-SM Nsys 报告，目标匹配序列在一个正式 DiT chunk 的开头为：

```text
match 0: Kernel A, grid=(152,3,1)
match 1: Kernel A, grid=(152,3,1)
match 2: Kernel A, grid=(152,3,1)
match 3: Kernel B, grid=(152,2,1)
```

因此：

```text
--launch-skip 2 --launch-count 2
```

会采集一次 Kernel A 和紧随其后的一次 Kernel B。这样每个 SM 配置只需启动
一次 NCU。

### 4.2 112 SM

```bash
 112 SM，Kernel A：

  CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
      --launch-skip 2 \
      -o ncu_kernel2_a_sm112 \
      python vae_dit_testbench_cuda_graph.py \
          --dit-sms 112 \
          --ignore-sm-coscheduling \
          --warmup-iters 3 \
          --profile-iters 1

  112 SM，Kernel B：

  CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
      --launch-skip 9 \
      -o ncu_kernel2_b_sm112 \
      python vae_dit_testbench_cuda_graph.py \
          --dit-sms 112 \
          --ignore-sm-coscheduling \
          --warmup-iters 3 \
          --profile-iters 1
```

### 4.3 114 SM

```bash
  114 SM 同理：

  CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
      --launch-skip 2 \
      -o ncu_kernel2_a_sm114 \
      python vae_dit_testbench_cuda_graph.py \
          --dit-sms 114 \
          --ignore-sm-coscheduling \
          --warmup-iters 3 \
          --profile-iters 1

  CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
      --launch-skip 9 \
      -o ncu_kernel2_b_sm114 \
      python vae_dit_testbench_cuda_graph.py \
          --dit-sms 114 \
          --ignore-sm-coscheduling \
          --warmup-iters 3 \
          --profile-iters 
```

预期生成：

```text
ncu_kernel2_sm112.ncu-rep
ncu_kernel2_sm114.ncu-rep
```

## 5. 采集后立即检查是否命中正确

运行：

```bash
ncu --import ncu_kernel2_sm112.ncu-rep --page details \
    --print-summary per-kernel

ncu --import ncu_kernel2_sm114.ncu-rep --page details \
    --print-summary per-kernel
```

每份报告应包含两个 profile result，并分别能看到：

```text
64x256_32x4 ... grid 152 x 3 x 1 ... block 128 x 1 x 1
128x256_32x3 ... grid 152 x 2 x 1 ... block 256 x 1 x 1
```

如果报告只有 Kernel A，或命中了两个 Kernel A，说明当前运行中的匹配顺序
与 Nsys 报告不同。此时不要用错误结果比较，改用下一节的四次独立采集方案。

## 6. 备用方案：四次独立采集

该方案不依赖两个 kernel 的相对出现顺序，最稳妥，但需要运行四次程序。

先定义不带 launch skip/count 的参数：

```bash
NCU_COMMON_SINGLE=(
    --replay-mode application
    --app-replay-mode relaxed
    --check-exit-code no
    --nvtx
    --nvtx-include "DiT_eager_alone/DiT_chunk/"
    --kernel-name-base function
    --launch-count 1
    --section WarpStateStats
    --section SchedulerStats
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --section LaunchStats
    --section Occupancy
    --clock-control none
    --cache-control none
    --force-overwrite
)

KERNEL_A='regex:.*cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8.*'
KERNEL_B='regex:.*cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8.*'
```

### 6.1 Kernel A，112 SM

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON_SINGLE[@]}" \
    --kernel-name "${KERNEL_A}" \
    -o ncu_kernel2_a_sm112 \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 1
```

### 6.2 Kernel A，114 SM

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON_SINGLE[@]}" \
    --kernel-name "${KERNEL_A}" \
    -o ncu_kernel2_a_sm114 \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 114 \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 1
```

### 6.3 Kernel B，112 SM

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON_SINGLE[@]}" \
    --kernel-name "${KERNEL_B}" \
    -o ncu_kernel2_b_sm112 \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 112 \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 1
```

### 6.4 Kernel B，114 SM

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON_SINGLE[@]}" \
    --kernel-name "${KERNEL_B}" \
    -o ncu_kernel2_b_sm114 \
    python vae_dit_testbench_cuda_graph.py \
        --dit-sms 114 \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 1
```

## 7. 后续分析重点

拿到报告后，对 Kernel A 和 Kernel B 分别比较 112/114 SM，重点查看：

1. `Duration`：确认 NCU 中仍复现约 `4/5` 和 `2/3` 的时间比；
2. `Waves Per SM` 与 `Achieved Occupancy`：判断是否跨过 wave 边界；
3. `Block Limit Registers`、`Block Limit Shared Mem`、
   `Block Limit Warps`：判断实际 residency 的限制来源；
4. `Eligible Warps Per Scheduler`、`Issued Warp Per Scheduler`：判断调度供给；
5. `Warp Stall Reasons`：尤其是 `Long Scoreboard`、`Not Selected`、
   `Wait`；
6. L1/L2 hit rate、DRAM throughput：排除主要由缓存或显存行为变化造成的
   加速；
7. `SM Busy`、`Compute Throughput`：判断两个配置是否都能稳定填满各自分区。

注意：NCU 的多 pass replay 会扰动绝对执行时间，因此最终时间差仍以 Nsys
为准；NCU 用于解释 occupancy、wave、stall 和 memory 指标为何变化。
