# DiT/VAE 共置完整探索总结：执行、锁、显存、CUDA Graph 与卷积路径

## 1. 文档目的

本文汇总当前 FastVideo AR diffusion 场景中 DiT 与 VAE 共置的完整探索，统一回答：

1. 最初观察到了什么性能问题；
2. 哪些现象已经被 NSYS/NCU 或严格 A/B 实验证实；
3. 双线程、单线程、细粒度提交、FA 修改、CUDA Graph 和静态 workspace
   分别解决了什么，又留下了什么问题；
4. 为什么同样具有访存压力的 attention 没有表现出 VAE 式的 allocator/显存问题；
5. 为什么多 Green Context pair 和动态 SM 配置难以与整段 VAE CUDA Graph
   同时使用；
6. 卷积/DNN 共置相关工作提供了哪些可借鉴的方向；
7. 当前是否真的无解，以及下一阶段应优先验证什么。

本文是总览入口。详细数字、命令和单轮实验仍以同目录下的专项文档为准。

## 2. 场景和目标

FastVideo 的真实请求包含以下主要流水线阶段：

```text
InputValidationStage
TextEncodingStage
ConditioningStage
LatentPreparationStage
CausalDMDDenoisingStage
DecodingStage
```

其中主要 GPU 时间集中在：

- Causal DMD denoising，即 DiT；
- chunkwise VAE decoding。

共置目标是在一张 170 SM GPU 上利用 CUDA Green Context，把 DiT 和上一 chunk
的 VAE decoder 放到互斥 SM 集合中并行执行：

```text
DiT chunk i  ||  VAE chunk i-1
```

离线扫描得到的一组代表性配置为：

| 执行阶段 | DiT SM | VAE SM |
|---|---:|---:|
| DiT0 | full | - |
| DiT1--4 + VAE0--3 | 100 | 70 |
| DiT5--6 + VAE4--5 | 112 | 58 |
| VAE6 | - | full |

理想状态要求同时满足：

1. GPU 上有足够的 DiT/VAE overlap；
2. Host 能持续提交 kernel，不出现长空洞；
3. chunk 边界可以切换 100:70 和 112:58；
4. 显存不超过约 31.36 GiB 的实际设备容量；
5. feature cache、cold/warm VAE 语义和输出正确。

## 3. 最初现象

### 3.1 单独限制 SM 可能提升吞吐，共置却可能变慢

最初实验表明，单独执行 DiT 或 VAE 时，适当减少其可使用的 SM 数量，有时会因
调度、wave quantization 或 kernel 映射变化而获得更好的吞吐。但 DiT 与 VAE
放在一起后，端到端时间反而可能增加。

同时观察到明显的不对称性：

- DiT 受到 VAE 的影响较大；
- VAE 受到 DiT 的影响相对较小；
- 两个 VAE 共置时相互影响也比较明显；
- 两个 DiT 的表现与两个 VAE 不同。

这说明 SM 数量不是唯一变量，共享 L2、HBM、运行库、allocator 和 Host 提交路径
都可能参与。

### 3.2 总时间中存在大量额外非重叠空隙

早期 NSYS 时间轴显示，端到端损失不能只用“同一批 kernel 慢了多少”解释。
在一些实验中，GPU 上存在显著的无 kernel 间隙：前一批工作已经结束，下一批
工作却没有及时提交或开始。

因此研究逐步分成两个问题：

1. **Host/Driver/allocator 问题**：为什么 GPU 会断粮；
2. **GPU 资源竞争问题**：已经提交并执行的 kernel 为什么变慢。

这两个问题必须分开统计，不能把全部时间增长统一称为“访存竞争”。

## 4. Host 提交和 CUDA Driver 路径

### 4.1 eager kernel launch 的调用模型

一次 eager GPU 算子大致经过：

```text
Python model forward
  -> PyTorch dispatcher/backend
  -> cuDNN/cuBLAS/FlashAttention
  -> CUDA Runtime/Driver API
  -> CUDA stream/device queue
  -> GPU kernel execution
```

GPU kernel 本身是异步的，但 Host 仍需逐个完成算子准备、函数属性处理、参数打包、
依赖记录和 launch。任何长时间 Host 阻塞都会让后续 GPU 工作无法及时进入队列。

### 4.2 FlashAttention 的 `cuFuncSetAttribute`

早期双 Host 线程实验发现，DiT FlashAttention 在 launch 前会设置 kernel 的动态
shared-memory 属性，进入 `cuFuncSetAttribute`/相关 Driver 路径。NSYS 中还观察到
`pthread_rwlock_*`。

当前能确认的是：

- `libcuda.so` 内部实现不开源，不能看到锁保护的具体对象；
- FA 的函数属性设置是 Driver 写路径的重要候选；
- VAE 不会像 FA 一样为每个主卷积频繁执行完全相同的 attribute 设置；
- 双线程同时进入 CUDA/Driver 会放大竞争；
- CUDA Graph replay 减少逐 kernel launch 后，该类阻塞明显减弱。

但不能把所有秒级 gap 都归因于可见的 `pthread_rwlock_wrlock`。在后续六 pair
报告中，可见写锁累计只有约 2.82 ms，远小于约 2 秒请求差异。

### 4.3 FA call-once 修改

对 FA 本地实现做 call-once/减少重复 attribute 设置后，不带 graph 的 testbench
有一定改善，说明该路径确实贡献开销。但真实 pipeline 仍然不够快，因为还存在：

- VAE allocator 跨 stream 同步；
- 多 pair identity 切换；
- feature cache 和大 activation；
- L2/HBM 竞争；
- pipeline 自身调度空洞。

因此 FA 修改是必要的局部优化，但不是 VAE 共置问题的完整解。

## 5. 多 Green Context pair 的问题

### 5.1 创建 pair 不是主要成本，切换 identity 才是

相同 106:64 配置的严格控制实验：

| 方案 | 稳态时间 |
|---|---:|
| 预创建两个 pair，但始终使用 pair1 | 约 13.8 s |
| 请求中间只切换一次 | 约 14.36 s |
| pair1/pair2 按 chunk 反复轮换 | 约 16.12 s |

另一次 A--B--A：

| 方案 | 均值 | 标准差 |
|---|---:|---:|
| 单 pair A1 | 13.5814 s | 0.0713 s |
| 六个相同 split 的不同 pair | 15.4683 s | 0.9606 s |
| 切回单 pair A2 | 13.5119 s | 0.0565 s |

六 pair 比 A2 慢约 1.9563 s/request，且方差明显增加；峰值显存只增加约
48.4 MiB。因此不是“创建了更多 Green Context 就直接消耗很多时间/显存”，
而是运行期在不同 stream identity 间迁移工作导致。

### 5.2 双线程不是唯一原因

双线程但固定复用一个 pair 的均值约 13.9083 s，标准差约 0.1986 s。由此可将
两类开销初步分离：

- 双 Host 线程相对单线程：约 0.3--0.4 s；
- 多 pair/stream identity：约 1.7--2.0 s，并带来更大抖动。

### 5.3 慢请求表现为 DiT step 间断粮

快请求的 DiT6 noise-step gap 常为 8--9 ms；慢请求出现过：

```text
429.3 ms
449.1 ms
756.6 ms
```

这不是“所有 DiT kernel 固定慢一点”，而是下一步提交出现数百毫秒空洞。

## 6. VAE allocator/stream 是当前最明确的 Host 根因

### 6.1 因果链

当前单 Host 线程、细粒度 eager、多 pair pipeline 的主要证据链为：

```text
VAE 切换到新的 Green Context ExternalStream
  -> 卷积、padding、upsample、skip/cache 申请大量临时 tensor
  -> allocator 发现旧 block 仍与其他 stream 关联
  -> cudaEventQuery/cudaEventSynchronize/cudaFree/cudaMalloc
  -> 唯一 Python 提交线程阻塞
  -> 后续 DiT kernel 无法及时提交
  -> GPU timeline 出现空洞
  -> 观察上表现为 DiT GPU span 和 wall time 增长
```

### 6.2 新 stream 的 VAE frame0 最明显

single-switch 报告中，chunk4 切换 VAE stream 后：

| 指标 | 未切换 | 切换后 |
|---|---:|---:|
| VAE frame0 Host range | 约 12.5 ms | 约 483.1 ms |

483.1 ms 内主要包括：

- 58 次 `cudaFree`：约 344.3 ms；
- 31 次 `cudaMalloc`：约 119.6 ms；
- kernel launch API：约 3.3 ms。

同 chunk 后续 frame1/frame2 立即恢复到约 12--13 ms。这说明问题是“新 stream
第一次触碰 allocator 状态”，而不是每个卷积都持续慢 400 ms。

round-robin 中还观察到多次约 258--274 ms 的 `cudaEventSynchronize`。

### 6.3 为什么 DiT 看上去是主要受害者

VAE 调用阻塞的是共享的 Host 提交线程；DiT 的下一批 kernel 因而不能按时发射。
所以 DiT span 增长不代表 DiT 自己的 kernel 都变慢了相同比例。它同时包含：

1. 已发射 kernel 的执行膨胀；
2. VAE Host 调用阻塞造成的提交空洞。

## 7. 为什么 fake VAE 很难复现

构造过 compute-heavy 和 read-heavy fake VAE，但都很难达到真实 VAE 对 DiT 的
影响程度。原因是简单 fake kernel 通常只模拟了 GPU 上的算力或读带宽，没有
模拟：

- 大量不同尺寸 tensor 的申请/释放；
- cuDNN execution plan/workspace；
- `cat`、`pad`、layout conversion；
- 上采样后快速增大的 feature map；
- causal feature cache 的替换和拼接；
- 多 stream allocator record/event；
- 数千次混合 CUDA/Driver 调用。

所以“read-heavy fake 不拖慢 DiT”不能推出真实 VAE 不是访存/allocator 问题。

## 8. GPU kernel active-time 膨胀

VAE CUDA Graph 消除大部分 Host gap 后，可以更干净地观察 GPU 资源竞争。

| 模块 | Alone active | Colocated active | 增加 | 比例 |
|---|---:|---:|---:|---:|
| DiT | 1533.267 ms | 1689.257 ms | 155.991 ms | 10.17% |
| VAE Graph | 1715.529 ms | 1851.151 ms | 135.622 ms | 7.91% |

调用次数一致，因此是同一批 kernel 执行时间增长。

### 8.1 DiT 贡献排序

| family | 绝对增加 | 相对增加 |
|---|---:|---:|
| elementwise | 48.839 ms | 33.71% |
| FlashAttention | 43.531 ms | 5.36% |
| Kernel2 | 26.626 ms | 5.67% |
| vectorized elementwise | 16.938 ms | 32.86% |
| unrolled elementwise | 14.209 ms | 36.61% |

普通、vectorized、unrolled elementwise 合计贡献约 79.986 ms，即 DiT 膨胀的
51.3%。FA 相对只慢 5.36%，但 baseline 达 812.840 ms，因此绝对贡献仍很大。

### 8.2 VAE 贡献排序

| family | 绝对增加 | 相对增加 |
|---|---:|---:|
| 主 XMMA convolution | 55.081 ms | 4.66% |
| elementwise | 34.530 ms | 14.44% |
| vectorized elementwise | 20.578 ms | 19.74% |
| NCHW->NHWC | 13.003 ms | 22.39% |

主卷积和两类 elementwise 各贡献约 40.6% 的 VAE 总膨胀：大型卷积相对变化小、
但基数大；elementwise/layout 单次短、相对更敏感。

### 8.3 NCU 的边界

早期 NCU 命中的 `indexed_wo_smem` XMMA 只占 VAE 总膨胀约 1.3%。它的 L2 hit
rate 从 97.72% 降到 94.69%，但 duration 基本不变、主要 stall 仍是 Math Pipe
Throttle。这只能说明该特定卷积没有因 L2 变化进入关键路径，不能代表所有 VAE
卷积。

真正绝对贡献最大的 `alignc4` 主 XMMA 仍值得继续做 NCU alone/colocated 对照。

## 9. Attention 和卷积都“访存”，为什么表现不同

“访存密集”至少包含两种不同含义：

1. kernel 执行时消耗 HBM/L2 带宽；
2. 框架运行时频繁申请、释放和跨 stream 回收 global-memory tensor。

FlashAttention 通过 tiling 和 online softmax 避免完整物化 N x N attention matrix，
大量临时状态位于 register/shared memory。其 global-memory 生命周期更接近：

```text
固定 Q/K/V/output + 少量辅助状态
```

VAE decoder 则要逐层物化完整高分辨率 feature map：

```text
cache/input cat
  -> pad tensor
  -> convolution output
  -> activation/residual
  -> upsample output
  -> 下一层 convolution
```

decoder 后半段 H/W 快速增大，还会出现独立 layout conversion 和 cuDNN workspace。
项目中实际见过单次 884 MiB `F.pad` 和 294 MiB 卷积相关临时申请。

DiT KV cache 通常是固定 storage 加 index 更新；VAE feature cache 仍包含 clone、
替换、cold/warm 差异和短时间维拼接。前者更像固定仓库，后者更像持续生成并替换
新的 feature tensor。

所以 attention 不是没有问题：它有 L2/HBM 竞争、FA attribute/Driver 路径和
执行膨胀；只是没有同时产生 VAE 式的大 activation allocator 生命周期问题。

## 10. CUDA Graph replay 探索

### 10.1 能解决什么

VAE graph replay 将数千次 eager launch 压缩为一次 Host graph launch，因而减少：

- Python/PyTorch 逐算子提交；
- FA/VAE 并发进入 Driver 的机会；
- replay 期间逐算子 allocator 活动；
- 原本的大量非重叠 Host gap。

VAE graph alone 与 eager alone 的 GPU 执行时间通常接近，说明 replay 主要优化
提交路径，而不是让卷积计算本身变快。

### 10.2 为什么显存膨胀

CUDA Graph 要求输入、输出、中间 activation 和 cache 地址在 replay 时保持稳定。
PyTorch 因而为 graph 维护 private pool。graph 外 eager 临时张量又不能直接复用
private pool 中的空闲区域。

NVIDIA 官方列出的典型 OOM 原因与本项目一致：

- static input 必须常驻；
- 不同 graph/pool 的中间 tensor 难以互相复用；
- graph 外 eager 临时分配不能使用 graph private pool；
- global/private pool 碎片；
- 多 stream capture 延迟回收。

参考：<https://docs.nvidia.com/dl-cuda-graph/troubleshooting/memory-issues.html>

### 10.3 实际显存数字

cold/warm 双图运行成功时：

```text
allocated delta = 7728.90 MiB
reserved delta  = 9456.00 MiB
allocated total = 16598.41 MiB
reserved total  = 22518.00 MiB
```

两张不同 warm graph 的实验曾达到约 10.38 GiB private pools，随后连 56 MiB
decoded buffer 都无法分配。三/四图捕获过程中曾遇到 884 MiB `F.pad` OOM。

错误信息中的 7--10 GiB 只是 graph private-pool 部分，不是整张卡总占用；模型、
KV/cache、latents、普通 allocator 和其他 CUDA context 已占用剩余显存。

### 10.4 cold/warm 双图可以跑，但不能动态改变 Green Context

当前可运行的双图按 VAE 语义划分：

- cold graph：VAE0；
- warm graph：VAE1--6。

普通运行结果：

```text
请求 0（含 capture）：16.94 s
请求 1（纯 replay）：13.63 s
峰值显存：约 20613 MiB
```

随后用 `--cuda-graph-trace=node` 采集：

```text
green+pytorch_demo/profiles/bucketed_cuda_graph/
  cold_warm_graph_stream_check.nsys-rep
```

NSYS 结论：

| 阶段 | Python/current stream | Graph node 实际位置 |
|---|---|---|
| VAE0--3 | GC3 / stream146（70 SM） | GC3 / stream146 |
| VAE4--5 | GC5 / stream152（58 SM） | GC3 / stream170 |
| VAE6 | GC0 / stream21（full） | GC3 / stream170 |

输入 copy 正确切换到了 58-SM/full stream，但所有带 `graphNodeId` 的 VAE kernel
始终属于 `greenContextId=3`。stream ID 可能变化为内部 stream170，但 Green
Context 没变。

因此：

> CUDA Graph 可以从另一个 current stream 发起 launch，但 graph node 的 Green
> Context 仍继承捕获时的执行环境，不能靠 Python `with stream` 动态改变 SM 分区。

严格实现当前调度至少需要：

```text
cold-70 graph
warm-70 graph
warm-58 graph
warm-full graph，或者 full eager
```

但 32 GiB 卡无法容纳这些整段 VAE graph；graph 存在时边界 eager 又可能因为
无法复用 private-pool 空间而 OOM。

### 10.5 NSYS 第二请求 OOM 的附加发现

最新 NSYS 的第一请求完整完成，17.69 s；第二请求在 Text Encoder FSDP 展开
参数时申请 3.91 GiB OOM。该错误发生在 DiT/VAE stage 之前，不影响第一请求的
stream 结论，但说明 Text Encoder 生命周期也是可利用的显存杠杆。

## 11. 静态存储探索

### 11.1 Workspace V1：只复用 causal-conv concat

用持久 buffer 替代：

```python
torch.cat([cache_x, x], dim=2)
```

结果：

| 指标 | 原始 | V1 |
|---|---:|---:|
| 稳态均值 | 15.767 s | 15.660 s |
| 标准差 | 0.583 s | 0.372 s |
| 峰值显存 | 17656.7 MiB | 19417.0 MiB |

V1 active buffer 1024 MiB，累计保留旧 buffer 1761 MiB。它减少部分
`cudaEventSynchronize`/`cudaFree`，但最后 VAE6 因剩余显存不足而从 310.4 ms
变成 455.6 ms。

### 11.2 Workspace V2：合并 concat 与 pad

V2 直接在精确尺寸的 padded-input buffer 中写入 cache/current input，并使用
CUDA event 安全回收 retired buffer。

结果：

| 指标 | 原始 | V2 |
|---|---:|---:|
| 稳态均值 | 15.915 s | 15.694 s |
| 中位数 | 16.261 s | 15.741 s |
| 峰值显存 | 17656.7 MiB | 17963.2 MiB |

active 约 883.3 MiB；第二请求后 retired 回收为 0；pipeline peak 只增加约
306.6 MiB。

NSYS：

| 指标 | 原始 | V2 |
|---|---:|---:|
| VAE Host range | 1811.4 ms | 2148.2 ms |
| `cudaEventSynchronize` | 992.7 ms | 1390.8 ms |
| `cudaMalloc` | 100/327.0 ms | 86/349.7 ms |
| `cudaFree` | 183/176.4 ms | 170/130.9 ms |
| VAE kernel 数 | 9826 | 9348 |
| kernel 累计时间 | 9750.6 ms | 9452.5 ms |

V2 减少了 478 个 kernel、约 298.1 ms kernel active time，却增加约 398.1 ms
event synchronization。局部静态 buffer 自己成为长期占用块，而卷积输出、upsample
和 skip activation 仍动态分配，所以没有解决跨 stream 的整体生命周期。

### 11.3 静态存储的结论

Python 层只固定 `cat`、`pad` 或 feature cache 不足以解决结构性问题。真正需要的
是覆盖主要 VAE activation 的 liveness-aware arena：生命周期不重叠的 tensor
映射到同一块显存，并显式管理 cuDNN workspace。

普通 PyTorch `Conv3d` 缺少适合该目标的简单稳定 `out=` 路径，继续推进可能需要：

- cuDNN Backend/Frontend；
- 自定义 CUDA operator；
- TensorRT execution context；
- 编译器级 activation memory planning。

## 12. 卷积/DNN 共置相关工作

### 12.1 共置研究通常同时考虑 SM 和共享内存系统

空间多任务 GPU 研究将 slowdown 分为 SM 数减少造成的 scalability slowdown，
以及 L2/global-memory contention 造成的 interference slowdown：

<https://www.sciencedirect.com/science/article/abs/pii/S0743731519307361>

这支持当前结论：SM 分区不能单独保证 VAE/DiT 无干扰。

### 12.2 Orion：按 operator 的 compute/memory 属性调度

Orion 在单个 operator 粒度判断任务是 compute-bound 还是 memory-bound，再安排
空间共享，而不是让两个完整模型阶段无条件重叠：

<https://fotstrt.github.io/files/2024-orion.pdf>

对当前项目的启示是：大型 XMMA conv、layout/upsample、elementwise 可能需要不同
共置策略，chunk 级单一比例不一定最优。

### 12.3 USHER：合并 operator graph，减少 cache interference

USHER 使用 operator graph merger 降低模型间 GPU cache interference：

<https://www.usenix.org/conference/osdi24/presentation/shubha>

它说明解决共置干扰不一定依赖更细 SM 比例，也可以从两条独立执行图的交错/合并
入手。

### 12.4 GPUlets 和稳定虚拟分区

GPUlets 使用时空共享、batch 与干扰预测选择稳定的虚拟 GPU 配置：

<https://www.usenix.org/conference/atc22/presentation/choi-seungbeom>

这类系统通常让 worker/context/stream 长期固定，不在一个请求内频繁迁移大型
卷积 activation，因此较少暴露当前 allocator-stream identity 问题。

### 12.5 STAO：高风险组合退回时间隔离

STAO 对低干扰组合使用 SM masking，对 L2/HBM 高风险组合切换到时间复用：

<https://www.sciencedirect.com/science/article/pii/S0167739X26003687>

对当前 VAE 的启示是：不应强求所有 VAE 卷积与 DiT overlap。可以只共置低风险
elementwise/小卷积，将大 activation 或高带宽阶段时间隔离。

### 12.6 SIRIUS：动态显存让渡

SIRIUS 指出空间共置中的显存动态共享本身是核心问题，静态显存分区会损失效率，
Host offload/Unified Memory 又可能带来较大传输开销：

<https://www.usenix.org/system/files/atc25-wang-jiali.pdf>

这与当前 graph private pool 挤压 eager 临时显存的现象一致。

### 12.7 cuDNN 原生 graph 和显式 workspace

cuDNN Backend Graph API 允许应用提供固定 workspace 指针，并能创建/更新 native
CUDA graph 的 VariantPack 指针：

<https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html>

TensorRT 也建议每个 captured graph 使用独立 execution context，同时由应用显式
提供并在不重叠执行时共享 activation memory：

<https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html>

这些能力比 PyTorch caching allocator 更接近本项目所需的“固定地址但跨配置复用
物理内存”。

CUDA graph memory nodes 还允许 Driver 根据 GPU ordered lifetime 在图内或图间
进行物理内存别名复用，但这需要更底层的 graph/memory-node 设计，当前 PyTorch
整段 capture 没有自动获得相同效果：

<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>

## 13. 为什么 PD 共置较少报告同类问题

典型 Prefill/Decode 系统通常具有：

- 固定 worker/process/context/stream；
- 显式预分配或分页 KV cache；
- 更规则的 Transformer shape；
- 按 request/batch/token iteration 的较粗调度；
- MPS/MIG 或稳定虚拟分区；
- 自定义 serving allocator。

当前 VAE pipeline 则在一个请求内反复切换 ExternalStream，并运行形状持续变化的
高分辨率卷积 activation。二者的 allocator 生命周期不同。

PD 并非绝对没有 Driver/allocator 问题；只是常见设计弱化了“同一个 allocator
携带大型 activation 在多个 stream identity 间迁移”这一触发条件。

## 14. 已排除或明显削弱的解释

1. **反复创建 Green Context**：pair 在请求前创建并长期保存。
2. **只需增加 warmup**：丢弃三个完整请求后，多 pair 仍慢且抖动。
3. **只是 SM 数量不同**：相同 106:64 的不同 pair 仍出现约 2 秒差异。
4. **只是双 Host 线程**：单 pair 双线程只贡献较小的 0.3--0.4 秒量级。
5. **只是可见 `pthread_rwlock_wrlock`**：部分报告中的写锁累计远小于 wall 差异。
6. **只是 L2 cache**：L2/HBM 能解释 kernel active 膨胀，不能解释 VAE frame0
   中数百毫秒 `cudaEventSynchronize/cudaFree/cudaMalloc`。
7. **只固定一个最大 `cat/pad` 就能解决**：V2 已减少 kernel，却没有减少主要
   allocator event wait。
8. **一张 graph 能跨 Green Context 动态改变 SM**：最新 node-level NSYS 已否定。

## 15. 当前问题是否无解

不是无解，但“PyTorch 整段 VAE graph + 多 Green Context 动态分区 + 32 GiB”这条
直接路径基本走不通。可行方向需要至少牺牲一个约束或进入更底层实现。

### 15.1 方案 A：固定一个保守 VAE Green Context

- cold/warm graph 都捕获在固定 split；
- 请求内不切换 VAE stream identity；
- 重新扫描一个整体最优固定 VAE SM 数；
- 接受 chunk4--6 不是各自孤立最优。

优点是当前代码已经接近可运行，能消除 Host 提交问题。缺点是放弃动态最优分区。

### 15.2 方案 B：只 capture VAE 子图

不 capture 整个 VAE，而是选择：

- 某个 up block；
- 重复 ResNet/conv group；
- launch 数多但 private-pool 增量较小的区域。

然后只为这些子图分别保存 70-SM 和 58-SM graph。预期显存显著低于多套整段图，
同时仍减少大量卷积 launch。这是近期最值得验证的方向。

### 15.3 方案 C：完整 activation arena

对整个 VAE 做 tensor liveness 分析：

1. 记录每个大型 activation 的 shape、dtype、首次/最后使用位置；
2. 让生命周期不重叠的 tensor 共享同一 offset；
3. feature cache 使用固定槽位；
4. 为 cuDNN 固定一个最大 workspace；
5. 融合或避免 `cat/pad/layout` 物化；
6. 固定请求 shape 后反复复用 arena。

这是最根本的 allocator 解法，但工程量最大。

### 15.4 方案 D：cuDNN Backend/TensorRT VAE

VAE resolution、chunk size、dtype 固定时，可考虑独立导出 decoder：

- 固定 cuDNN execution plan/tactic；
- 用户管理 workspace；
- fuse conv/bias/activation/residual；
- 用 execution context 管理 activation memory；
- 在保证不并发时让不同配置共享底层 arena。

卷积 VAE 比具有复杂 cache/control 的完整 DiT 更适合先做这条路径。

### 15.5 方案 E：释放其他阶段显存

Prompt encoding 后，Text Encoder 在 denoise/decode 阶段不再参与计算。可以验证：

- CPU offload Text Encoder；
- prompt embedding cache；
- Text Encoder 放到另一 GPU；
- 将请求按 text-encoding 和 generation 分批调度。

最新 OOM 中 FSDP Text Encoder 重新展开需要约 3.91 GiB，说明这部分容量可能足以
容纳额外的 warm-58 子图/整图。但每请求反复搬运约数 GiB 权重可能增加延迟，
需要批处理或独立设备配合。

### 15.6 方案 F：混合空间/时间调度

- 小卷积、elementwise、cache update 与 DiT 共置；
- 大 XMMA conv、upsample、大 pad 阶段暂时避免 overlap；
- 用离线 NSYS/NCU 建立 operator 风险表；
- 调度目标加入切换成本，而不是只使用孤立 kernel 的最优 SM 点。

这与 Orion/STAO 的研究方向最一致，也可能比强求整个 chunk overlap 更稳。

## 16. 建议的下一阶段顺序

### Step 1：建立可靠基线

1. 保留一个固定 split 的 cold/warm graph baseline；
2. 与单 pair、单 Host、细粒度 eager baseline 做同请求 A/B；
3. 固定 resolution、chunk size、seed、模型和 FA 版本；
4. 分开统计 capture 请求和 steady-state 请求。

### Step 2：测显存可回收空间

在各 stage 边界记录：

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.memory_snapshot()
```

重点测 Text Encoder、DiT KV/cache、VAE graph/private pool 和 decoded output。

### Step 3：做 VAE 子图显存曲线

按 decoder block 逐步扩大 capture 范围，记录：

```text
Host launch 数减少
graph allocated/reserved delta
alone GPU time
colocated GPU time
是否能同时保存 70/58 两套
```

目标不是 graph 化最多，而是最大化：

```text
减少的 Host/Driver 时间 / 新增常驻显存
```

### Step 4：主卷积 NCU

优先分析真正贡献最大的主 XMMA `alignc4`，以及 VAE elementwise/layout：

- duration；
- long/short scoreboard；
- math pipe throttle；
- L1/TEX、L2 hit rate；
- DRAM throughput；
- eligible warps；
- SM frequency。

### Step 5：决定工程路线

- 子图 graph 能容纳两套：继续子图方案；
- 释放 Text Encoder 后能容纳动态 graph：评估请求调度/offload；
- 两者都不行：转向 cuDNN Backend/TensorRT activation arena；
- 工程预算不足：采用固定 split 或混合时间调度。

## 17. 最终结论

当前研究已经把最初笼统的“VAE 可能抢 L2”细化为三个相互关联但不同的问题：

1. **Host/Driver 问题**：双线程 eager 和 FA attribute 路径会产生 Driver 竞争；
2. **allocator/stream 问题**：真实 VAE 在多 ExternalStream 间切换时触发大规模
   event synchronization、malloc/free，阻塞 Host 并让 DiT 断粮；
3. **GPU 共享资源问题**：消除 Host gap 后，DiT/VAE kernel 仍分别膨胀约
   10.17%/7.91%，涉及主卷积、FA、elementwise、layout 和共享 L2/HBM。

CUDA Graph 能有效解决第 1、2 类问题，却把 VAE 的大 activation 生命周期固化为
7--10 GiB private pool，并且 graph node 不能跨 Green Context 动态改变 SM 分区。
局部静态 workspace 能减少若干 kernel/分配，但无法覆盖整个卷积 decoder 的
activation 生命周期，甚至可能因常驻显存增加 allocator 等待。

因此最准确的阶段性判断是：

> 问题并非没有解决办法；但解法不再是增加一个 Python stream、再保存一张整段
> CUDA Graph 或固定一个 cache tensor。下一步需要转向 VAE 子图 capture、显式
> cuDNN workspace、全局 activation arena、释放其他 stage 显存，或按 operator
> 风险进行混合空间/时间调度。

