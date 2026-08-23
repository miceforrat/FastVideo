# Kernel2 离散性能台阶的后续定位方案

## 1. 已知现象与目标

目标 Kernel B：

```text
cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8
grid=(152,2,1), block=(256,1,1), Grid Size=304 CTA
```

已有 NCU sweep：

| DiT SM | TPC | Duration | Elapsed cycles |
|---:|---:|---:|---:|
| 110 | 55 | 约 1.19 ms | 3,491,490 |
| 112 | 56 | 约 1.20 ms | 3,491,535 |
| 114 | 57 | 801.44 us | 2,333,162 |
| 116 | 58 | 801.28 us | 2,333,844 |

结果形成 `110 ~= 112` 和 `114 ~= 116` 两个平台，阈值位于
`114 SM / 57 TPC`。但该 kernel 只能驻留 1 CTA/SM，而
`ceil(304 / SM_count)` 在四种配置下都为 3，普通平坦 CTA wave 模型解释
不了约 `3 -> 2` 的 duration 变化。

后续要区分：

1. ignore-coscheduling 改变了 SM/TPC/GPC 层级；
2. 相同 SM 数下具体物理资源子集不同；
3. CUTLASS CTA swizzle、tile rasterization 或边界对齐；
4. 前序 kernel、缓存或输入地址状态；
5. NCU 平均值隐藏了 per-SM 负载不均衡。

官方参考：

- [CUDA Green Context Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
- [CUDA Programming Guide: Green Contexts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)
- [cuBLASLt algorithm attributes](https://docs.nvidia.com/cuda/cublas/index.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/pdf/NsightComputeCli.pdf)

## 2. 推荐顺序

```text
Step 1  相同 SM 数：default split 对比 ignore split
Step 2  相同 SM 数：正常 group 对比 remainder
Step 3  采集 per-instance SM 指标
Step 4  确定实际 M/N/K 和 cuBLASLt algorithm 配置
Step 5  构造独立 GEMM microbenchmark
Step 6  改变矩阵形状，观察阈值是否随 grid 移动
```

Step 1 当前即可执行。Step 2–6 需要先增加测试开关或独立 testbench；本文会
写出最小实现要求和实现后的命令，不把尚不存在的参数伪装成当前可运行参数。

## 3. 公共环境与 NCU 参数

```bash
docker exec -it cjh-Fastvideo-cu128 bash
cd /FastVideo/green+pytorch_demo
nvidia-smi
```

定义 Kernel B 公共参数：

```bash
NCU_KERNEL_B=(
    --replay-mode application
    --app-replay-mode relaxed
    --check-exit-code no
    --nvtx
    --nvtx-include "DiT_eager_alone/DiT_chunk/"
    --kernel-name-base function
    --kernel-name Kernel2
    --launch-skip 9
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
```

`--launch-skip 9` 已由 Nsys runtime correlation 验证：目标是首个正式
`DiT_chunk` 内第 10 个名为 `Kernel2` 的 launch。

## 4. Step 1：default split 对比 ignore split

### 4.1 目的

112 和 120 都是默认 8-SM 对齐下可获得的大小。固定 SM 数，仅改变是否传入
`CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`，可以隔离该 flag 的
影响。官方文档说明它会降低对齐要求并把 SM 独立于层级处理，代价是失去部分
coscheduling/cluster 能力。

### 4.2 一次性命令

```bash
mkdir -p ncu_kernel2_b_split_flag

for dit_sms in 112 120; do
    CUDA_VISIBLE_DEVICES=1 ncu "${NCU_KERNEL_B[@]}" \
        -o "ncu_kernel2_b_split_flag/sm${dit_sms}_default" \
        python vae_dit_testbench_cuda_graph.py \
            --dit-sms "${dit_sms}" \
            --warmup-iters 3 \
            --profile-iters 1

    CUDA_VISIBLE_DEVICES=1 ncu "${NCU_KERNEL_B[@]}" \
        -o "ncu_kernel2_b_split_flag/sm${dit_sms}_ignore" \
        python vae_dit_testbench_cuda_graph.py \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --warmup-iters 3 \
            --profile-iters 1
done
```

批量导出：

```bash
for dit_sms in 112 120; do
    for split_mode in default ignore; do
        ncu \
            --import "ncu_kernel2_b_split_flag/sm${dit_sms}_${split_mode}.ncu-rep" \
            --page details \
            --print-summary per-kernel \
            > "ncu_kernel2_b_split_flag/sm${dit_sms}_${split_mode}.txt"
    done
done
```

快速检查：

```bash
for report in ncu_kernel2_b_split_flag/*.txt; do
    echo "===== ${report} ====="
    grep -E \
        "Kernel2<|Grid Size|Block Size|# SMs|# TPCs|Duration|SM Active Cycles|Achieved Occupancy" \
        "${report}"
done
```

必须确认日志的 `actual_dit` 是 112/120，且报告命中
`grid=(152,2,1), block=(256,1,1)`。

### 4.3 判定

```text
相同112 SM，default明显快于ignore：
    coscheduling或SM/TPC/GPC层级布局参与。

相同112 SM，两者都约1.20 ms：
    ignore flag本身不是充分原因，继续Step 2和Step 6。

112相同而120不同，或反之：
    分区大小与硬件层级存在组合效应。
```

## 5. Step 2：交换正常 group 与 remainder

### 5.1 当前缺少的开关

当前 C++ 固定把 requested group 给 DiT、remainder 给 VAE。需要新增：

```text
GreenContext(..., dit_from_remainder=False)
--dit-from-remainder
```

启用 ignore split 后：

```text
目标DiT=112：先请求58 SM group，把112 SM remainder给DiT
目标DiT=114：先请求56 SM group，把114 SM remainder给DiT
```

日志必须打印 requested group、remainder、DiT 使用哪一侧、actual_dit 和
actual_vae。官方文档指出 remainder 不具备正常 group 的相同性能/功能保证，
所以这只用于根因诊断，不作为生产方案。

### 5.2 实现后的命令

以下命令在 `--dit-from-remainder` 实现前不可执行：

```bash
mkdir -p ncu_kernel2_b_resource_role

for dit_sms in 112 114; do
    CUDA_VISIBLE_DEVICES=1 ncu "${NCU_KERNEL_B[@]}" \
        -o "ncu_kernel2_b_resource_role/sm${dit_sms}_group" \
        python vae_dit_testbench_cuda_graph.py \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --warmup-iters 3 \
            --profile-iters 1

    CUDA_VISIBLE_DEVICES=1 ncu "${NCU_KERNEL_B[@]}" \
        -o "ncu_kernel2_b_resource_role/sm${dit_sms}_remainder" \
        python vae_dit_testbench_cuda_graph.py \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --dit-from-remainder \
            --warmup-iters 3 \
            --profile-iters 1
done
```

判定：同 SM 数下 group/remainder 不同，说明具体物理子集或资源角色参与；
若相同，则更像由 SM/TPC 数量与 CUTLASS grid 共同决定。

## 6. Step 3：per-instance SM 指标

### 6.1 查询本机 metric

先查询再采集，避免假设当前 CC 12.0/NCU 版本支持某个 suffix：

```bash
ncu --query-metrics-mode all > ncu_metrics_all.txt

grep -E \
    "^sm__cycles_active|^smsp__cycles_active|^sm__warps_active|^smsp__warps_active" \
    ncu_metrics_all.txt \
    > ncu_sm_instance_metrics.txt
```

### 6.2 采集模板

如果查询确认以下名字可用：

```text
sm__cycles_active.avg
smsp__cycles_active.avg
sm__warps_active.avg
```

则执行：

```bash
mkdir -p ncu_kernel2_b_instances

for dit_sms in 112 114; do
    CUDA_VISIBLE_DEVICES=1 ncu \
        --replay-mode application \
        --app-replay-mode relaxed \
        --check-exit-code no \
        --nvtx \
        --nvtx-include "DiT_eager_alone/DiT_chunk/" \
        --kernel-name-base function \
        --kernel-name Kernel2 \
        --launch-skip 9 \
        --launch-count 1 \
        --metrics sm__cycles_active.avg,smsp__cycles_active.avg,sm__warps_active.avg \
        --print-metric-instances values \
        --clock-control none \
        --cache-control none \
        --force-overwrite \
        -o "ncu_kernel2_b_instances/sm${dit_sms}" \
        python vae_dit_testbench_cuda_graph.py \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --warmup-iters 3 \
            --profile-iters 1
done
```

导出：

```bash
for dit_sms in 112 114; do
    ncu \
        --import "ncu_kernel2_b_instances/sm${dit_sms}.ncu-rep" \
        --page raw \
        --print-metric-instances values \
        > "ncu_kernel2_b_instances/sm${dit_sms}.txt"
done
```

若 112 中少数实例明显更长、114 更均匀，则直接支持 CTA/TPC 尾部不均衡。
若实例分布也相同，则转向 CUTLASS tile 顺序或 profiler 不可见的驱动调度。

NCU 已出现 `Waves Per SM` 按全卡 170 SM 归一化的问题，因此必须先核对
instance 数量和层级，不能直接把实例编号当作 Green Context 内连续 SM ID。

## 7. Step 4：确定 M/N/K 与算法配置

需要获得：

```text
M/N/K、layout、leading dimensions、batch count、epilogue、algorithm ID
CTA swizzling、split-K、reduction scheme、custom option、inner shape ID
```

先尝试 cuBLASLt logger：

```bash
mkdir -p cublaslt_logs

CUBLASLT_LOG_LEVEL=5 \
CUBLASLT_LOG_FILE=/FastVideo/green+pytorch_demo/cublaslt_logs/dit_sm112.log \
CUDA_VISIBLE_DEVICES=1 \
python vae_dit_testbench_cuda_graph.py \
    --dit-sms 112 \
    --ignore-sm-coscheduling \
    --warmup-iters 1 \
    --profile-iters 1
```

检查日志：

```bash
ls -lh cublaslt_logs/dit_sm112.log

grep -Ei \
    "algo|swizz|splitk|reduction|matmul|152" \
    cublaslt_logs/dit_sm112.log \
    | head -n 200
```

如果 logger 信息不足，需要实现实验用 cuBLASLt wrapper，通过
`cublasLtMatmulAlgoConfigGetAttribute` 查询：

```text
CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
CUBLASLT_ALGO_CONFIG_SPLITK_NUM
CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME
CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID
```

wrapper 当前尚未实现。

## 8. Step 5：独立 GEMM microbenchmark

建议新增：

```text
green+pytorch_demo/kernel2_green_context_testbench.py
```

要求：复现真实 M/N/K、layout、BF16、epilogue 和算法；使用现有 Green
Context stream；支持 `--dit-sms`、`--ignore-sm-coscheduling`；预分配输入
输出；提供 `Kernel2_target` NVTX range；打印实际 grid/block。

实现后的基础 sweep：

```bash
mkdir -p kernel2_micro_logs

for dit_sms in 110 112 114 116; do
    CUDA_VISIBLE_DEVICES=1 \
    python kernel2_green_context_testbench.py \
        --dit-sms "${dit_sms}" \
        --ignore-sm-coscheduling \
        --warmup-iters 20 \
        --profile-iters 100 \
        > "kernel2_micro_logs/sm${dit_sms}.log" 2>&1
done
```

若独立 GEMM 仍在 114 SM 跳变，根因位于 GEMM tile/algorithm 与 Green
Context 调度层；若不再跳变，则前序 kernel、cache、地址或完整模型环境参与。

## 9. Step 6：shape/grid sweep

独立 testbench 需要支持 `--m --n --k`，并打印实际 grid。确定 M 到 grid.x
的映射后，选择能产生：

```text
grid.x = 148, 150, 152, 154, 156
```

的真实 M 值，再运行：

```bash
mkdir -p kernel2_shape_sweep

for grid_x in 148 150 152 154 156; do
    m_value="REPLACE_WITH_M_FOR_GRID_X_${grid_x}"

    for dit_sms in 108 110 112 114 116 118; do
        CUDA_VISIBLE_DEVICES=1 \
        python kernel2_green_context_testbench.py \
            --m "${m_value}" \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --warmup-iters 20 \
            --profile-iters 100 \
            > "kernel2_shape_sweep/grid${grid_x}_sm${dit_sms}.log" 2>&1
    done
done
```

含 `REPLACE_WITH...` 的模板不能原样执行。必须先生成明确的
`grid_x -> M` 对照表。

判定：

```text
阈值随grid.x移动：CUTLASS tile/swizzle/rasterization主导。
不同grid.x都固定在57 TPC：Green Context TPC/GPC拓扑主导。
阈值同时依赖grid.x和TPC：二者耦合。
```

## 10. 暂不优先的方向

暂不优先 PC sampling 或更细 SASS stall。现有 110–116 报告中的 occupancy、
warp cycles/instruction、eligible warps、L2 hit rate 和频率基本一致，问题更像
整体 CTA 排布和 makespan，而不是某条指令在 114 SM 后突然变快。

## 11. 结论判据

| 观察 | 支持的结论 |
|---|---|
| 同 SM 下 default/ignore 不同 | coscheduling 或层级资源布局参与 |
| 同 SM 下 group/remainder 不同 | 具体物理 SM 子集或资源角色参与 |
| 独立 GEMM 复现 | 根因位于 GEMM/调度层，不依赖完整 DiT |
| 阈值随 grid.x 移动 | CUTLASS tile/swizzle 主导 |
| 阈值固定在 57 TPC | Green Context TPC/GPC 拓扑主导 |
| 112 per-instance 长尾、114 均匀 | 直接证明负载分发/尾部不均衡 |

即使无法看到闭源驱动的 CTA 分发实现，完成这些正交实验后，也可以把原因从
笼统的“SM 数变化”收敛到 CUTLASS 映射、Green Context 拓扑或二者耦合，
形成可复现、可佐证的工程结论。
