# 新 GPU 上验证 VAE CUDA Graph 与完全最优分区：完整工作流

## 1. 目标与实验边界

本文是一份可独立使用的迁移和实验手册，用于把当前 FastVideo DiT/VAE 共置
工作临时迁移到一张新 GPU，并验证：

```text
真实请求
  + chunkwise DiT/VAE overlap
  + CUDA Green Context SM 隔离
  + 每个 chunk pair 的最优 SM 分区
  + VAE CUDA Graph replay
```

本轮最核心的问题是：

> 在首请求的 warmup/capture 成本被摊销后，绑定到各自最优 Green Context 的
> VAE CUDA Graph，能否稳定降低相对 full-SM sequential 的真实请求延迟，并且
> 显存成本在新 GPU 上可接受？

本文默认：

- 容器名或镜像环境为 `cjh-Fastvideo-cu128`；
- 容器内仓库位于 `/FastVideo`；
- Python 位于 `/opt/venv/bin/python`；
- 模型为 `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers`；
- 请求为 T2V、81 frames、7 个 causal chunks；
- 不修改原 FastVideo 默认 pipeline，而是使用新增 pipeline/stage；
- 所有命令除“宿主机检查”外都在容器中执行。

## 2. 当前方案与已知结论

### 2.1 为什么需要 VAE CUDA Graph

此前真实 pipeline 和 NSYS 实验表明，eager VAE 会产生大量 Host 侧 CUDA、cuDNN、
allocator 和 Driver 调用。尤其在 VAE 切换到新的 Green Context ExternalStream
时，可能出现：

```text
cudaMalloc/cudaFree
cudaEventQuery/cudaEventSynchronize
cuDNN plan/workspace 准备
大量逐 kernel launch
```

这些调用会阻塞共享 Host 提交线程，使后续 DiT kernel 无法及时提交，最终形成
GPU timeline 空洞。CUDA Green Context 只能隔离 SM，不能隔离 Host、Driver、L2、
HBM 或 allocator。

VAE CUDA Graph 将大量 eager launch 压缩为一次 `graph.replay()`，主要作用是：

1. 减少 Host 提交量；
2. 减少 VAE 与 DiT 在 Driver launch 路径上的互相干扰；
3. 缩短 VAE replay 到 DiT eager submit 之间的延迟；
4. 让 GPU 上的 DiT/VAE overlap 更接近离线 testbench。

Graph 不会消除 GPU 侧 L2/HBM 和执行资源竞争，因此 colocated kernel 仍可能膨胀。

### 2.2 为什么“完全最优分区”需要多张 graph

CUDA Graph node 绑定 capture 时的 stream/context。过去的 node-level NSYS 已证明：

```text
Python current stream 切换到新的 VAE GC
    !=
已捕获 graph 的 node 自动迁移到新的 VAE GC
```

因此一张在 70-SM VAE GC 上捕获的 warm graph，不能通过外层
`with torch.cuda.stream(58_sm_stream)` 变成 58-SM graph。

严格动态最优方案需要为不同 capture Green Context 保存不同 graph。当前 V2
实现更进一步，为 feature-cache chain 中的每个 VAE chunk 保存一张 graph，共
7 张串联 graph。

### 2.3 当前 170-SM 卡的逐 chunk 配置

当前 V2 stage 中的配置为：

| 窗口 | DiT | VAE | VAE 路径 |
|---|---:|---:|---|
| `DiT0` | full | - | - |
| `DiT1 || VAE0` | 100 | 70 | cold graph 0 |
| `DiT2 || VAE1` | 94 | 76 | warm graph 1 |
| `DiT3 || VAE2` | 100 | 70 | warm graph 2 |
| `DiT4 || VAE3` | 106 | 64 | warm graph 3 |
| `DiT5 || VAE4` | 112 | 58 | warm graph 4 |
| `DiT6 || VAE5` | 112 | 58 | warm graph 5 |
| `VAE6` | - | full | warm graph 6 |

对应代码：

```text
fastvideo/pipelines/stages/dynamic_green_context_cuda_graph_v2_stage.py
fastvideo/pipelines/basic/wan/
  wan_causal_dmd_dynamic_green_context_cuda_graph_v2_pipeline.py
```

这套具体数字仅对原 170-SM 实验环境有效。新卡硬件变化后必须重新确认。

### 2.4 双图方案不是严格最优方案

`bucketed_green_context_cuda_graph_stage.py` 保存 cold/warm 两张 graph，显存成本
较低，但 graph 仍绑定到其 capture GC。它可以作为显存折中对照，不应被标记为
“完全最优动态分区”。

## 3. 建议迁移的文件集合

严格 V2 主线的最小代码链为：

```text
fastvideo/green_context/
fastvideo/pipelines/stages/pingpong_stage.py
fastvideo/pipelines/stages/dynamic_green_context_cuda_graph_v2_stage.py
fastvideo/pipelines/basic/wan/
  wan_causal_dmd_dynamic_green_context_cuda_graph_v2_pipeline.py
examples/inference/basic/
  basic_self_forcing_causal_dynamic_green_context_cuda_graph_v2.py
  benchmark_self_forcing_causal_vae_graph_optimal_gc_ab.py
```

分区扫描链为：

```text
green+pytorch_demo/vae_dit_real_chunk_colocation_testbench.py
green+pytorch_demo/run_real_chunk_sm_sweep_8.sh
green+pytorch_demo/run_real_chunk_best_sm_fine_sweep.sh
```

统一 A/B runner 支持：

```text
--variant sequential
--variant v2-optimal-graph
```

两种 variant 使用相同 prompt、shape、seed、生成入口和统计代码，但必须分别在
新的 Python 进程中运行，避免模型、Graph private pool 和 allocator 状态串组。

## 4. 宿主机与目标 GPU 检查

### 4.1 获取硬件和 Driver 信息

在宿主机执行：

```bash
nvidia-smi -L
nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap \
  --format=csv
```

查看是否存在其他计算进程：

```bash
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv
```

正式跑 A/B 和 profile 时必须使用空闲卡。其他进程即使只间歇运行，也可能改变
时钟、显存余量、L2/HBM 压力和测量方差。

### 4.2 宿主机 GPU ID 与容器逻辑 ID

若容器只映射宿主机物理 GPU 3：

```text
宿主机 physical GPU 3 -> 容器 cuda:0
```

此时容器内使用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

如果容器暴露全部 GPU，容器编号通常才与宿主机编号一致。不要根据旧机器命令
猜测，进入容器后重新执行：

```bash
nvidia-smi -L
/opt/venv/bin/python - <<'PY'
import torch

print("device_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    print({
        "logical_index": index,
        "name": prop.name,
        "sms": prop.multi_processor_count,
        "memory_gib": prop.total_memory / 1024**3,
        "compute_capability": f"{prop.major}.{prop.minor}",
    })
PY
```

### 4.3 容器要求

复用已有容器时检查：

- `/FastVideo` 是目标 checkout；
- 模型/Hugging Face cache 已挂载；
- 目标 GPU 可见；
- `/flash-attention` 指向需要的本地 FA 源码；
- NSYS/NCU 可执行文件存在。

若创建新容器，通常需要：

```text
--gpus device=<physical-id>
--ipc=host
--cap-add SYS_ADMIN
--security-opt seccomp=unconfined   # 仅在机器策略需要时
-v <repo>:/FastVideo
-v <model-cache>:<container-cache>
-v <fa-source>:/flash-attention     # 使用本地 FA 时
```

不同机器的宿主机目录和 Docker 启动规范不同，因此这里不写死完整 `docker run`。

## 5. 容器内环境复现

### 5.1 保存软件版本

```bash
cd /FastVideo

git rev-parse HEAD
git status --short
nvidia-smi
nvcc --version
nsys --version
ncu --version

/opt/venv/bin/python - <<'PY'
import torch

prop = torch.cuda.get_device_properties(0)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", prop.name)
print("SMs:", prop.multi_processor_count)
print("memory GiB:", prop.total_memory / 1024**3)
print("compute capability:", prop.major, prop.minor)
PY
```

将输出和实验日志放在一起。最终报告必须能追溯到硬件、Driver、CUDA、PyTorch、
cuDNN、FA 和 FastVideo commit。

### 5.2 安装 FastVideo

```bash
cd /FastVideo
/opt/venv/bin/python -m pip install -e .
```

若容器中已经是 editable install，且只改了 Python 文件，可以跳过。切换到新的
checkout、Python 环境或容器后应重新执行。

### 5.3 编译自带 Green Context 扩展

canonical 实现位于：

```text
/FastVideo/fastvideo/green_context
```

不要使用 `green+pytorch_demo/` 下的旧重复实现，也不要从旧机器复制编译后的
`_greenctx*.so`。

容器内执行：

```bash
cd /FastVideo/fastvideo/green_context
rm -rf build
rm -f _greenctx*.so
/opt/venv/bin/python setup.py build_ext --inplace
```

这里的删除范围仅限 GC extension 自己的构建目录和二进制产物。

验证 import：

```bash
cd /FastVideo
/opt/venv/bin/python -c \
  'from fastvideo.green_context import GreenContextPairPool; print(GreenContextPairPool)'
```

验证真实资源切分：

```bash
cd /FastVideo
CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python - <<'PY'
import torch
from fastvideo.green_context import GreenContextPairPool

total = torch.cuda.get_device_properties(0).multi_processor_count
requested = total // 2
requested -= requested % 2

pool = GreenContextPairPool(
    [requested],
    device=0,
    ignore_sm_coscheduling=True,
)
pair = pool[requested]
print({
    "requested_dit_sms": requested,
    "actual_dit_sms": pair.actual_dit_sms,
    "actual_vae_sms": pair.actual_vae_sms,
    "total_sms": pair.total_sms,
})
pool.synchronize()
PY
```

必须满足：

```text
actual_dit_sms == requested_dit_sms
actual_dit_sms + actual_vae_sms == total_sms
```

`ignore_sm_coscheduling=True` 对应
`CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`，是当前 2-SM 粒度扫描的基础。

若编译时找不到 Green Context API 或该 flag，说明 toolkit/header 版本不满足；
若运行时报 Driver 不支持，说明宿主机 Driver 不满足。不能用旧 `.so` 绕过。

### 5.4 自定义 FlashAttention

若新卡实验需要复现当前修改后的 FA，先记录源码：

```bash
cd /flash-attention
git rev-parse HEAD
git status --short
rg -n "64.*64|kBlockM|kBlockN|cuFuncSetAttribute" \
  csrc/flash_attn/src/flash_fwd_launch_template.h \
  csrc/flash_attn 2>/dev/null
```

安装：

```bash
cd /flash-attention
MAX_JOBS=8 /opt/venv/bin/python -m pip install \
  -v --no-build-isolation --no-cache-dir .
```

验证运行时加载的二进制：

```bash
cd /FastVideo
/opt/venv/bin/python - <<'PY'
import flash_attn
import flash_attn_2_cuda

print("flash_attn:", flash_attn.__file__)
print("flash_attn_2_cuda:", flash_attn_2_cuda.__file__)
PY
```

源码改成 64x64 不代表 Python 一定加载了对应 `.so`。必须保存这两个路径，并在
NSYS 中用 FA grid/kernel 形状做第二次确认。

### 5.5 环境变量

基础组先保持简单：

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
```

默认先不设置：

```text
PYTORCH_CUDA_ALLOC_CONF
```

若多图 capture 因 allocator fragmentation 失败，再独立运行：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

default allocator 与 expandable-segments 的数据不能混在同一统计组。后者可能
改善碎片，但不能消除 CUDA Graph 静态地址和 private pool 的真实显存占用。

## 6. 判断是否需要重扫分区

执行：

```bash
/opt/venv/bin/python - <<'PY'
import torch

p = torch.cuda.get_device_properties(0)
print({
    "name": p.name,
    "cc": (p.major, p.minor),
    "sms": p.multi_processor_count,
    "memory_gib": p.total_memory / 1024**3,
})
PY
```

判断规则：

1. 同一 GPU 型号、架构和 170 SM，仅显存容量/空闲量更高：可以先复用旧配置，
   但建议在旧最优点附近做 step=2 局部复核；
2. GPU 型号、compute capability 或 SM 总数改变：必须重跑 step=8 和 step=2；
3. FA tile、cuDNN、precision、分辨率、frame 数或 chunk size 改变：至少做局部
   复核；
4. 非 170-SM 卡不能直接运行写死 100/94/106/112 的 stage；
5. 新卡的“等比例 SM”也不保证最优，因为 CTA wave quantization 与 kernel grid
   是离散关系。

## 7. 真实负载下扫描完全最优分区

### 7.1 Chunk 配对语义

扫描使用：

```text
green+pytorch_demo/vae_dit_real_chunk_colocation_testbench.py
```

它按真实 pipeline 配对：

```text
DiT0 || 上一个请求 VAE6
DiT1 || 当前请求 VAE0 cold
DiT2 || 当前请求 VAE1 warm
DiT3 || 当前请求 VAE2 warm
DiT4 || 当前请求 VAE3 warm
DiT5 || 当前请求 VAE4 warm
DiT6 || 当前请求 VAE5 warm
```

它包含真实 DiT 多 noise steps、remask/writeback、真实 VAE streaming decode、
post-quant conv 和 feature cache 路径，同时打印 full-SM sequential 对照。

当前 V2 单请求 pipeline 的边界仍为：

```text
DiT0 full
VAE0..5 与 DiT1..6 配对
VAE6 full
```

因此 chunk0 扫描用于未来跨请求 overlap，不写入当前 V2 的
`DIT_SMS_BY_CHUNK`。

### 7.2 170-SM 卡的 step=8 coarse sweep

当前脚本扫描 DiT SM 72--128，step=8：

```bash
cd /FastVideo/green+pytorch_demo

GPU_ID=0 \
PYTHON_BIN=/opt/venv/bin/python \
WARMUP_ITERS=3 \
PROFILE_ITERS=5 \
OUTPUT_DIR=/FastVideo/green+pytorch_demo/logs/new_gpu_graph/coarse_step8 \
bash run_real_chunk_sm_sweep_8.sh
```

非 170-SM 卡必须先重新设计范围，不能直接使用该命令中的旧上下界。

### 7.3 step=2 fine sweep

从 coarse 日志中为 chunk0--6 选择 7 个最优中点，填入：

```text
green+pytorch_demo/run_real_chunk_best_sm_fine_sweep.sh
```

的 `BEST_DIT_SMS`。每个 chunk 扫 `[best-8, best+8]`，step=2：

```bash
cd /FastVideo/green+pytorch_demo

GPU_ID=0 \
PYTHON_BIN=/opt/venv/bin/python \
WARMUP_ITERS=3 \
PROFILE_ITERS=5 \
OUTPUT_DIR=/FastVideo/green+pytorch_demo/logs/new_gpu_graph/fine_step2 \
bash run_real_chunk_best_sm_fine_sweep.sh
```

最优点主要按 5 次 `parallel wall` 的均值选择，同时检查：

- 原始 5 次数据和标准差；
- `DiT with VAE graph`；
- `VAE graph colocated`；
- full-SM sequential total；
- 是否存在突变点或 wave quantization cliff；
- 是否有 OOM、illegal access 或其他进程干扰。

### 7.4 更新 V2 配置

更新：

```text
fastvideo/pipelines/stages/dynamic_green_context_cuda_graph_v2_stage.py
```

中的：

```python
DIT_SMS_BY_CHUNK = {
    1: ...,  # VAE0 cold
    2: ...,  # VAE1 warm
    3: ...,  # VAE2 warm
    4: ...,  # VAE3 warm
    5: ...,  # VAE4 warm
    6: ...,  # VAE5 warm
}
```

启动时 `GreenContextPairPool` 会验证 requested SM 与 actual SM 是否一致，以及
DiT/VAE 两个资源集合是否覆盖全部 SM。

## 8. 正式 A/B 测试

### 8.1 统一 Runner

使用：

```text
examples/inference/basic/
  benchmark_self_forcing_causal_vae_graph_optimal_gc_ab.py
```

默认：

```text
request 0：lazy initialization + VAE warmup/capture，不计入均值
request 1--5：steady replay，计入统计
```

程序打印每个请求的数据，并自动统计 mean、median、sample standard deviation、
min、max 和 peak memory。

### 8.2 A 组：stock full-SM sequential

```bash
cd /FastVideo
mkdir -p green+pytorch_demo/logs/new_gpu_graph

CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python \
  examples/inference/basic/benchmark_self_forcing_causal_vae_graph_optimal_gc_ab.py \
  --variant sequential \
  --num-requests 6 \
  --discard-requests 1 \
  2>&1 | tee green+pytorch_demo/logs/new_gpu_graph/ab_sequential_6req.log
```

### 8.3 C 组：VAE 七图 + 完全最优 GC

必须使用新的独立 Python 进程：

```bash
cd /FastVideo

CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python \
  examples/inference/basic/benchmark_self_forcing_causal_vae_graph_optimal_gc_ab.py \
  --variant v2-optimal-graph \
  --num-requests 6 \
  --discard-requests 1 \
  2>&1 | tee green+pytorch_demo/logs/new_gpu_graph/ab_v2_7graph_6req.log
```

首次正确性检查可以加：

```bash
--save-last-video
```

正式计时不保存视频，避免编码和文件 I/O 混入。

### 8.4 推荐补充组

为了拆分收益来源，建议后续增加：

| 组 | 方案 | 作用 |
|---|---|---|
| A | stock full-SM sequential | 端到端基线 |
| B | eager VAE + dynamic optimal GC | 单独测 overlap/分区 |
| C | 7 Graph + dynamic optimal GC | 本轮目标 |
| D | cold/warm 双图 | 显存折中 |

当前统一 runner 已实现 A/C。B/D 可以后续增加 variant，不能用不同 prompt 或不同
请求数的历史 example 直接拼表。

### 8.5 A/B 固定变量

所有组必须一致：

- 同一 GPU 且无其他进程；
- 同一 FastVideo commit；
- 同一 FA `.so`；
- 同一模型 checkpoint；
- 同一 prompt、seed、分辨率、81 frames 和 chunk size；
- 同一 precision、autocast、tiling、offload；
- 同一 allocator 环境变量；
- 同一请求数和 discard 数；
- 无 profiler 的性能结果与 NSYS/NCU 结果分开。

### 8.6 结果计算

记录：

| variant | req1 | req2 | req3 | req4 | req5 | mean | median | std | peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential | | | | | | | | | |
| V2 7-graph optimal | | | | | | | | | |

计算：

```text
speedup = sequential_mean / graph_mean
latency_reduction = (sequential_mean - graph_mean) / sequential_mean
```

不要只比较最快请求，也不要把 request0 capture 时间混入 steady mean。

## 9. CUDA Graph 显存验收

当前约 31.36-GiB 卡的既有现象：

- cold/warm 双图曾产生约 7.7 GiB allocated delta 和约 9.5 GiB reserved delta；
- Graph private pool、模型、KV cache、VAE feature cache、静态输入输出同时常驻；
- 多图版本可能在 capture 或首次 replay 时 OOM；
- `expandable_segments=True` 只能缓解部分碎片，不能消除真实常驻量。

新卡记录四个时间点：

```text
模型加载后
VAE 独立 warmup 后
7 graph capture 后
steady request 峰值
```

通过条件：

1. 7 张 graph 全部 capture 完成；
2. capture 后仍有运行时 headroom；
3. request1--5 无 OOM/illegal access；
4. request1--5 不重复 capture；
5. 最后一个保存的视频正常；
6. request shape 改变时实现会拒绝错误复用。

当前 graph signature 包括 latent shape、dtype、device、VAE precision、autocast 和
VAE tiling。分辨率、frame 数、chunk size 或这些设置变化时需要新的 graph。

## 10. NSYS Profile

### 10.1 为什么必须使用 node trace

普通 CUDA Graph trace 可以看到 replay，但不能充分证明 graph 内各节点属于哪个
Green Context。此次关键证据是：

```text
VAE graph node 的 stream / greenContextId
是否与对应 chunk pair 的 capture GC 一致
```

因此使用 `--cuda-graph-trace=node`。

### 10.2 采集命令

先完成无 profiler A/B，再执行：

```bash
cd /FastVideo
mkdir -p green+pytorch_demo/profiles/new_gpu_graph

CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o green+pytorch_demo/profiles/new_gpu_graph/v2_optimal_gc_node \
  /opt/venv/bin/python \
    examples/inference/basic/benchmark_self_forcing_causal_vae_graph_optimal_gc_ab.py \
    --variant v2-optimal-graph \
    --num-requests 2 \
    --discard-requests 1
```

NSYS 只跑 2 个请求即可：第一个 capture，第二个提供 steady timeline。不要为了
profile 重复 6 次，避免报告过大和额外显存压力。

GPU metrics 参数随 NSYS 版本变化，先检查：

```bash
nsys profile --gpu-metrics-devices=help
```

不要使用已被当前版本拒绝的旧参数 `--gpu-metrics-device=0`。

### 10.3 UI 中看哪个时间段

跳过模型加载、VAE warmup、capture 和 request0。定位 request1 的：

```text
DynamicGC_DiT_chunk_0
DynamicGC_DiT_chunk_1 ... DynamicGC_DiT_chunk_6
DynamicGC_VAE_static_input_copy_chunk_0 ... chunk_6
DynamicGC_VAE_graph_replay_chunk_0 ... chunk_6
```

逐窗口检查：

1. `VAE0 graph || DiT1`；
2. `VAE1 graph || DiT2`；
3. `VAE2 graph || DiT3`；
4. `VAE3 graph || DiT4`；
5. `VAE4 graph || DiT5`；
6. `VAE5 graph || DiT6`；
7. 最后的 full-SM `VAE6 graph`。

每个窗口检查：

- graph node 是否在正确 stream/`greenContextId`；
- DiT 与 VAE 实际 overlap 区间；
- input copy 是否发生在 replay 前；
- pair 边界是否出现 GPU 空洞；
- Host graph replay API 是否保持很短；
- 是否仍出现长 `cudaMalloc/cudaFree/cudaEventSynchronize`；
- kernel active time 是否因 L2/HBM 竞争膨胀。

### 10.4 导出 SQLite

```bash
cd /FastVideo
nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output green+pytorch_demo/profiles/new_gpu_graph/v2_optimal_gc_node.sqlite \
  green+pytorch_demo/profiles/new_gpu_graph/v2_optimal_gc_node.nsys-rep
```

`.nsys-rep` 和 `.sqlite` 不提交 Git。

## 11. NCU 流程

NCU 用来解释具体 kernel 的膨胀，不用于端到端 latency。顺序应是：

```text
无 profiler A/B
  -> NSYS 判断 Host gap、overlap、graph binding
  -> 找到 active-time 绝对增长最大的 kernel family
  -> NCU alone/colocated 成对采集
```

已有重点：

- VAE XMMA convolution；
- VAE elementwise/layout conversion；
- DiT FlashAttention；
- DiT `Kernel2`；
- DiT elementwise。

具体 NCU section 和过滤命令参考：

```text
green+pytorch_demo/docs/colocate_reason_analysis/ncu_profile_cmd.md
```

NCU application replay 会严重放大运行时间，也可能改变 allocator/Graph 行为。
一次只采明确的 kernel 和少量 launch，不用 NCU 时间计算端到端 speedup。

## 12. 日志与产物管理

建议按 GPU 和日期分目录：

```text
green+pytorch_demo/logs/new_gpu_graph/<gpu>_<date>/
green+pytorch_demo/profiles/new_gpu_graph/<gpu>_<date>/
```

文件名包含：

```text
variant
split/schedule
request count
allocator mode
FA tile/version
```

例如：

```text
sequential_full_6req_default_allocator.log
v2_7graph_optimal_6req_default_allocator.log
v2_7graph_optimal_2req_node.nsys-rep
```

以下产物不进入 Git：

```text
*.log
*.nsys-rep
*.ncu-rep
*.sqlite
*.so
build/
video_samples*/
模型权重和 Hugging Face cache
```

## 13. 当前 Git 整理与提交建议

当前工作区包含主线实现、早期 demo、多个控制实验、失败方案、workspace 尝试、
分析文档和二进制 profile。禁止使用：

```bash
git add .
```

也不要用 `git reset --hard` 或 `git clean -fd` 清理，未提交文件应视为仍有价值的
实验资产。

### 13.1 Commit 1：Green Context 基础设施

建议 commit message：

```text
[feat]: add reusable CUDA Green Context runtime
```

精确加入：

```bash
cd /FastVideo

git add \
  fastvideo/green_context/__init__.py \
  fastvideo/green_context/pool.py \
  fastvideo/green_context/binding.cpp \
  fastvideo/green_context/green_context.cpp \
  fastvideo/green_context/green_context.h \
  fastvideo/green_context/setup.py \
  fastvideo/green_context/README.md
```

不要 add `_greenctx*.so`。当前通用 `.gitignore` 会忽略 `build/`，但仍应逐文件
add，避免本地二进制混入。

### 13.2 Commit 2：真实请求级 V2 Graph pipeline

建议 commit message：

```text
[feat]: add chunkwise VAE CUDA Graph colocation pipeline
```

加入：

```bash
git add \
  fastvideo/pipelines/stages/pingpong_stage.py \
  fastvideo/pipelines/stages/dynamic_green_context_cuda_graph_v2_stage.py \
  fastvideo/pipelines/basic/wan/wan_causal_dmd_dynamic_green_context_cuda_graph_v2_pipeline.py \
  examples/inference/basic/basic_self_forcing_causal_dynamic_green_context_cuda_graph_v2.py \
  examples/inference/basic/benchmark_self_forcing_causal_vae_graph_optimal_gc_ab.py \
  examples/inference/basic/basic_self_forcing_causal_sequential_steady_state.py
```

V2 继承 `PingPongDenoisingDecodingStage` 的真实 causal DiT 辅助逻辑，因此
`pingpong_stage.py` 是依赖，不是无关历史文件。

### 13.3 Commit 3：分区扫描与完整文档

建议 commit message：

```text
[misc]: add reproducible DiT VAE colocation workflow
```

加入：

```bash
git add \
  green+pytorch_demo/vae_dit_real_chunk_colocation_testbench.py \
  green+pytorch_demo/run_real_chunk_sm_sweep_8.sh \
  green+pytorch_demo/run_real_chunk_best_sm_fine_sweep.sh \
  green+pytorch_demo/docs/colocate_reason_analysis/vae_dit_colocation_complete_exploration_summary.md \
  green+pytorch_demo/docs/colocate_reason_analysis/new_gpu_vae_cuda_graph_optimal_split_complete_workflow.md
```

如果 `analyze_real_chunk_eager_sweep.py` 已确认能解析 graph testbench 日志，再加入；
否则不要因名称相似顺手提交。

### 13.4 本轮暂不提交的 tracked 修改

以下修改属于 `dynamic_gc_pair_timing` 插桩，不是严格 V2 Graph 运行依赖：

```text
fastvideo/configs/sample/base.py
fastvideo/entrypoints/video_generator.py
fastvideo/pipelines/pipeline_batch_info.py
```

以下属于 VAE allocator/static workspace 探索，也不是 V2 必需依赖：

```text
fastvideo/models/vaes/wanvae.py
fastvideo/models/vaes/causal_conv_workspace.py
fastvideo/models/vaes/causal_conv_static_workspace.py
```

它们应进入独立 commit，或继续留在工作区。

### 13.5 本轮暂不提交的历史方案

```text
fixed_green_context_dual_cuda_graph_v4/v5/v6*
dual_thread_*
single_pair_*
six_identical_pairs_*
fine_grained_serial_workspace_*
fine_grained_serial_static_workspace_*
greem_demo/
green_demo2/
green+pytorch_demo/logs/
green+pytorch_demo/profiles/
```

这些内容可以后续统一移动到 `experiments/archive/`，形成独立的 research-artifact
commit；不要与新卡验证主线混在一起。

### 13.6 每个 commit 的检查

```bash
git status --short
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
git diff --cached
```

Python 静态检查：

```bash
cd /FastVideo

/opt/venv/bin/python -m py_compile \
  fastvideo/green_context/pool.py \
  fastvideo/pipelines/stages/pingpong_stage.py \
  fastvideo/pipelines/stages/dynamic_green_context_cuda_graph_v2_stage.py \
  fastvideo/pipelines/basic/wan/wan_causal_dmd_dynamic_green_context_cuda_graph_v2_pipeline.py \
  examples/inference/basic/basic_self_forcing_causal_dynamic_green_context_cuda_graph_v2.py \
  examples/inference/basic/benchmark_self_forcing_causal_vae_graph_optimal_gc_ab.py
```

GC import：

```bash
/opt/venv/bin/python -c \
  'from fastvideo.green_context import GreenContextPairPool; print(GreenContextPairPool)'
```

最后在目标 GPU 上分别跑 sequential 和 V2 smoke/steady test。

## 14. 推荐实际执行顺序

```text
1. 整理并提交 GC 基础设施
2. 提交 V2 pipeline 与统一 A/B runner
3. 提交 testbench、扫描脚本和本完整文档
4. 新卡 checkout 同一 commit
5. 记录硬件、Driver、CUDA、PyTorch、cuDNN、FA
6. 在新容器重新编译 GC extension
7. 运行 GC smoke test，确认 actual SM split
8. 确认运行时加载的 FA .so
9. 判断是否可复用旧分区
10. 必要时跑 step=8 coarse sweep
11. 填写中点并跑 step=2 fine sweep
12. 更新 V2 DIT_SMS_BY_CHUNK
13. 无 profiler 跑 sequential 6 requests
14. 新进程跑 V2 6 requests
15. 保存一次最终视频验证正确性
16. 检查 graph memory 和稳态 headroom
17. NSYS node trace 跑 capture + 1 steady request
18. 核对 graph stream/greenContextId 和 overlap
19. 必要时对关键 kernel 做 NCU alone/colocated
20. 只提交文字结论，不提交二进制报告
```

## 15. 最终验收标准

只有同时满足以下条件，才能认为新卡上的方案验证成功：

1. GC extension 在新环境重新编译并正确切分 SM；
2. 最优分区来自目标硬件，或有充分依据证明可复用；
3. 7 张串联 VAE graph capture 完成；
4. graph capture 后仍有安全显存余量；
5. 连续 5 个 steady requests 无 OOM、illegal access 和重复 capture；
6. V2 steady mean 优于 full-SM sequential steady mean；
7. NSYS 证明各 graph node 位于预期 Green Context；
8. DiT/VAE overlap 明显，pair 边界无异常长空洞；
9. 输出视频与原 pipeline 在可接受质量范围内一致；
10. 结果可由明确 commit、环境版本、命令和日志复现。

若只满足“程序能运行”，但 graph node 仍绑定错误 GC，或者收益只存在于包含
capture 的单次请求中，均不能算作完全最优分区方案成功。
