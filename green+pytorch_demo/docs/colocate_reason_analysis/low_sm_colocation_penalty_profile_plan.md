# 低 DiT SM 下共置膨胀增大的定位方案

## 1. 目标与当前证据

定义 DiT 的共置膨胀：

```text
Delta_DiT = DiT_colocated - DiT_alone
```

细粒度扫描中，7 个 chunk 都表现为 DiT SM 越少，`Delta_DiT` 越大：

| Chunk | 低/高 DiT SM | 低 SM Delta | 高 SM Delta |
|---:|---:|---:|---:|
| 0 | 88 / 104 | 193.7 ms | 139.8 ms |
| 1 | 80 / 96 | 244.1 ms | 187.9 ms |
| 2 | 88 / 104 | 219.5 ms | 160.3 ms |
| 3 | 96 / 112 | 194.3 ms | 132.9 ms |
| 4 | 96 / 112 | 187.0 ms | 141.5 ms |
| 5 | 104 / 120 | 170.8 ms | 114.6 ms |
| 6 | 104 / 120 | 167.3 ms | 121.2 ms |

接下来需要回答两个问题：

1. 低 SM 多出的约 45--61 ms 由哪些 kernel signature 贡献；
2. 主导 kernel 是因为 CTA wave 增多，还是因为 L2/HBM 竞争和 warp stall
   增加。

其中 chunk 0 必须使用 cold VAE，chunk 1--6 使用 steady-state VAE。

## 2. 实验控制要求

进入容器并确认 GPU 空闲：

```bash
docker exec -it cjh-Fastvideo-cu128 bash
cd /FastVideo/green+pytorch_demo
nvidia-smi -i 1
```

所有细粒度分区都要保留：

```text
--ignore-sm-coscheduling
```

否则请求的 SM 数可能重新受默认共调度粒度约束。采集期间不要让其他进程使用
物理 GPU 1。

## 3. 第一阶段：采集 7 个 chunk 的低/高 SM Nsys

每份报告内部同时包含 DiT alone 和 DiT+VAE Graph colocated，因此可以在同一
进程、同一模型状态下比较，减少跨运行波动。为控制报告大小，正式迭代设为 1。

### 3.1 一次运行全部 14 份报告

```bash
cd /FastVideo/green+pytorch_demo

mkdir -p nsys_low_high_sm

LOW_SMS=(88 80 88 96 96 104 104)
HIGH_SMS=(104 96 104 112 112 120 120)

for chunk_idx in 0 1 2 3 4 5 6; do
    vae_args=()
    if [[ "${chunk_idx}" -eq 0 ]]; then
        vae_args+=(--vae-first-chunk)
    fi

    for endpoint in low high; do
        if [[ "${endpoint}" == low ]]; then
            dit_sms="${LOW_SMS[${chunk_idx}]}"
        else
            dit_sms="${HIGH_SMS[${chunk_idx}]}"
        fi

        echo "chunk=${chunk_idx} endpoint=${endpoint} dit_sms=${dit_sms}"

        CUDA_VISIBLE_DEVICES=1 nsys profile \
            --trace=cuda,nvtx,osrt,cudnn,cublas \
            --cuda-graph-trace=node \
            --sample=none \
            --force-overwrite=true \
            -o "nsys_low_high_sm/chunk${chunk_idx}_${endpoint}_sm${dit_sms}" \
            python vae_dit_testbench_cuda_graph.py \
                --chunk-idx "${chunk_idx}" \
                --dit-sms "${dit_sms}" \
                --ignore-sm-coscheduling \
                --warmup-iters 3 \
                --profile-iters 1 \
                "${vae_args[@]}"
    done
done
```

预期生成 14 个 `.nsys-rep`。检查：

```bash
find nsys_low_high_sm -maxdepth 1 -name '*.nsys-rep' -printf '%f\n' | sort
```

### 3.2 生成命令行摘要

先查看当前 Nsys 版本支持的 report 名称：

```bash
nsys stats --help-reports | grep -E 'nvtx_gpu_proj_sum|cuda_gpu_kern_sum'
```

批量输出 NVTX GPU 投影和全局 kernel 摘要：

```bash
mkdir -p nsys_low_high_sm/stats

for rep in nsys_low_high_sm/*.nsys-rep; do
    base="$(basename "${rep}" .nsys-rep)"

    nsys stats \
        --report nvtx_gpu_proj_sum \
        --report cuda_gpu_kern_sum \
        --format csv \
        --force-export=true \
        "${rep}" \
        > "nsys_low_high_sm/stats/${base}.csv"
done
```

`cuda_gpu_kern_sum` 是整份报告的摘要，会混入 warmup、alone 和 colocated，
不能直接拿它计算共置膨胀。它主要用于检查 kernel family 和调用次数。
`nvtx_gpu_proj_sum` 用来确认以下范围存在且边界正常：

```text
DiT_eager_alone/DiT_chunk
Colocated_iteration/DiT_after_VAE_graph_replay/DiT_chunk
Colocated_iteration/VAE_graph_replay_host
```

## 4. 在 Nsys UI 中获得 kernel 贡献 Top-N

对每份报告分别做下面两次选择：

1. 展开目标 DiT Green Context stream；
2. 定位 `DiT_eager_alone -> DiT_chunk`，只选择正式计时对应的 kernel；
3. 在 Events View 中按 `Kernel Name + Grid + Block` 聚合或导出；
4. 再定位
   `Colocated_iteration -> DiT_after_VAE_graph_replay -> DiT_chunk`，做同样导出；
5. 不要选中 warmup、VAE graph alone 或 VAE stream。

对每个精确 signature 计算：

```text
contribution_k = colocated_total_duration_k - alone_total_duration_k
```

然后按 `contribution_k` 从大到小排序。需要同时保留：

```text
chunk
endpoint
kernel name
grid
block
calls
alone total duration
colocated total duration
absolute contribution
slowdown ratio
```

低、高 SM 报告的工作量应该一致。若同一 signature 的 calls 不一致，先检查
是否误选了 warmup 或多个 iteration，不要继续比较 duration。

### 4.1 需要计算的核心差分

对同一 chunk：

```text
low_penalty_k  = low_colocated_k  - low_alone_k
high_penalty_k = high_colocated_k - high_alone_k

explained_drop_k = low_penalty_k - high_penalty_k
```

按 `explained_drop_k` 排序，Top-3 就是“为什么低 SM 的共置影响更大”的直接
贡献者。这里不能只按 slowdown 百分比排序；FA 可能百分比不大，但 baseline
很长，因此绝对贡献仍可能最大。

最后检查：

```text
sum(explained_drop_k)
```

是否接近 CUDA Event 测得的：

```text
Delta_DiT_low - Delta_DiT_high
```

两者的剩余差额主要对应 kernel launch gap、CPU submission gap 和 stream
空闲间隙，而不是 kernel active time。

## 5. 第二阶段：对 Top-3 kernel 做 NCU alone/colocated 对照

不建议一开始对 7 个 chunk 的所有 kernel 全量跑 NCU。先根据 Nsys 找到每个
chunk 的 `explained_drop` Top-3；若多个 chunk 的 Top-3 signature 相同，优先
选择一个代表 chunk，再扩展到其他 chunk。

### 5.1 公共 NCU 参数

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
```

使用 application replay 是为了尽量维持原始共置执行关系。NCU 会显著扰动
运行时间，因此 NCU 中的 duration 不作为真实端到端耗时证据。

### 5.2 单个 endpoint 的采集模板

以下变量需要根据 Nsys Top-3 的结果填写：

```bash
chunk_idx=0
dit_sms=88
endpoint=low

# 示例：flash_fwd_kernel；其他 kernel 可改为正则表达式。
kernel_filter='flash_fwd_kernel'

# 在对应 NVTX 范围内，该名称第几个 launch；从 0 开始。
launch_skip=0

vae_args=()
if [[ "${chunk_idx}" -eq 0 ]]; then
    vae_args+=(--vae-first-chunk)
fi

mkdir -p ncu_low_high_sm
```

Alone：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include "DiT_eager_alone/DiT_chunk/" \
    --kernel-name "${kernel_filter}" \
    --launch-skip "${launch_skip}" \
    -o "ncu_low_high_sm/chunk${chunk_idx}_${endpoint}_sm${dit_sms}_alone" \
    python vae_dit_testbench_cuda_graph.py \
        --chunk-idx "${chunk_idx}" \
        --dit-sms "${dit_sms}" \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 1 \
        "${vae_args[@]}"
```

Colocated：

```bash
CUDA_VISIBLE_DEVICES=1 ncu "${NCU_COMMON[@]}" \
    --nvtx-include \
        "Colocated_iteration/DiT_after_VAE_graph_replay/DiT_chunk/" \
    --kernel-name "${kernel_filter}" \
    --launch-skip "${launch_skip}" \
    -o "ncu_low_high_sm/chunk${chunk_idx}_${endpoint}_sm${dit_sms}_colocated" \
    python vae_dit_testbench_cuda_graph.py \
        --chunk-idx "${chunk_idx}" \
        --dit-sms "${dit_sms}" \
        --ignore-sm-coscheduling \
        --warmup-iters 3 \
        --profile-iters 1 \
        "${vae_args[@]}"
```

对高 SM endpoint 修改 `dit_sms` 和 `endpoint`，再运行相同的 alone/colocated
命令。每个 kernel 最终应得到四份报告：

```text
low alone
low colocated
high alone
high colocated
```

如果提示 `No kernels were profiled`，先移除 `--nvtx-include`，让 NCU 输出
Available Kernels，确认 `--kernel-name`；然后恢复 NVTX 过滤，并核对范围末尾
的 `/` 和 `launch_skip`。同名 kernel 有多个 grid 时，需要从 Nsys 中确认目标
signature 在该 NVTX 范围内的顺序，使用 `--launch-skip` 精确命中。

## 6. NCU 指标判读

优先比较同一 endpoint 的 colocated 与 alone：

### 6.1 支持共享访存竞争的证据

```text
DRAM Throughput / DRAM utilization 上升或接近饱和
L2 hit rate 下降，或 L2 sectors/bytes 明显增加
Long Scoreboard stalls 上升
Memory Throttle stalls 上升
每条指令平均等待时间增加
```

随后比较 low 与 high：如果这些差异在 low SM 下更大，就支持“VAE 的
L2/HBM 压力在低 DiT SM 下累计得更严重”。

### 6.2 支持 CTA wave/尾波的证据

```text
grid、block、寄存器和 shared memory 不变
理论/实际 occupancy 基本不变
SM 数跨过某个阈值时 kernel duration 阶跃变化
colocated 与 alone 都在相同 SM 阈值出现阶跃
```

这种情况说明主要是离散 wave 数变化，而不是共置独有的访存退化。

### 6.3 两者同时存在

最可能的实际情况是：

```text
低 SM -> 更多 CTA wave
每个 wave 又受到 VAE 的共享缓存/显存流量影响
最终 stall 在更多 wave 上累计
```

因此不要只看单个百分比指标，应同时报告 kernel 调用次数、单次 duration、
累计 absolute contribution、grid/block 和 warp stall 构成。

## 7. 建议执行顺序

1. 先完成 14 份 Nsys 报告；
2. 对每个 chunk 计算 `explained_drop_k` 并取 Top-3；
3. 检查 Top-3 是否在多个 chunk 中重复；
4. 先对一个代表 chunk 的 Top-3 做四象限 NCU；
5. 只有当不同 chunk 表现不一致时，再扩展 NCU；
6. 将 Nsys 的 absolute contribution 作为主结论，NCU stall/访存指标作为机理
   证据。

