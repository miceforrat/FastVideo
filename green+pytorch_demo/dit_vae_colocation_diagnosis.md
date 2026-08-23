# DiT/VAE Green Context 共置非重叠间隙诊断

## 结论

主要额外间隙不是 Green Context 分区代码写错，而是
FlashAttention 的 host-side `cudaFuncSetAttribute` 与 NVIDIA driver
全局写锁冲突。

## 根因证据链

最大的 DiT gap 为 20--26 ms，几乎全部被 VAE 的 TF32 cuDNN 3D
convolution 覆盖。

对 gap 后第一个 DiT kernel 做时间关联：

```text
VAE convolution 开始
DiT stream 上一个 kernel 结束
DiT host thread 等待约 20--26 ms
cudaLaunchKernel 被调用
约 5 us 后 FlashAttention kernel 启动
```

说明 FlashAttention kernel 并没有提前提交到 GPU 后等待调度，而是 host
线程迟迟没有完成 launch。

Nsight 的 OS runtime trace 显示，三个 DiT 提交线程分别有：

```text
pthread_rwlock_wrlock:
累计等待 321.6 / 327.7 / 330.1 ms
单次最长约 25.6 ms
```

这与每轮新增约 331 ms 的 DiT kernel gap 几乎完全吻合。

调用栈是：

```text
FlashAttention mha_fwd
  -> run_flash_fwd
  -> cudaFuncSetAttribute
  -> cuFuncSetAttribute
  -> NVIDIA libcuda.so
  -> pthread_rwlock_wrlock
```

对应源码在
`/data/cjh/flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h`：

```cpp
if (smem_size >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
}
kernel<<<grid, ..., smem_size, stream>>>(params);
```

这个设置在每一次 FlashAttention forward 时都会执行，不只是第一次初始化。
DiT 每个 chunk 有大量 attention 调用，于是不断获取 driver 写锁。

VAE 的 cuDNN 大卷积运行时，`cuFuncSetAttribute` 的写锁获取被阻塞；等 VAE
kernel 接近结束，锁才返回，然后 FlashAttention 才被提交。

## Green Context 代码有没有问题

`green+pytorch_demo/green_context.cpp` 的主要流程是合理的：

```text
cuDeviceGetDevResource
cuDevSmResourceSplitByCount
cuDevResourceGenerateDesc
cuGreenCtxCreate
cuGreenCtxStreamCreate
```

而且报告确认：

- DiT 和 VAE 有不同 `greenContextId`。
- 使用不同 stream。
- 两边绝大部分 GPU 时间确实有 kernel overlap。
- 分区并没有退化成同一个普通 CUDA stream。

因此 SM partition 本身是生效的。

C++ 封装有一些工程性问题，但不是这次 330 ms gap 的根因：

- 析构中的 `cuStreamDestroy`、`cuGreenCtxDestroy` 没检查返回值。
- 没有暴露实际分配的 SM 数量供 Python 验证。
- 没检查 `dit_sms` 的有效范围和 SM group 对齐结果。
- 创建了 `CU_GREEN_CTX_DEFAULT_STREAM`，随后又额外创建 stream；略显冗余，
  但不导致当前串行。
- 混合 Driver API stream 与 PyTorch Runtime API 是敏感路径，但 Nsight 已确认
  算子进入了对应 Green Context。

## Python benchmark 的问题

`green+pytorch_demo/vae_dit_testbench.py` 的双线程方式不是根因，但有几处应改进。

### 1. 每轮重新创建两个 Python thread

```python
for _ in range(args.profile_iters):
    t_dit = threading.Thread(...)
    t_vae = threading.Thread(...)
```

这增加噪声。更合理的是创建两个常驻 worker，通过 barrier/queue 驱动。

### 2. 子线程没有显式设置 CUDA device

建议每个 worker 开头调用：

```python
torch.cuda.set_device(0)
```

`testbench_samemodel.py` 已经这样做了。当前只有一张目标 GPU，所以没导致此次
gap，但最好补上。

### 3. Tensor 生命周期没有 `record_stream`

模型输入、输出、cache 从默认 stream 创建，在 Green stream 使用。初始化后的
全局同步保证首次使用正确，但 caching allocator 对临时 tensor 的跨 stream
生命周期可能依赖隐式 event。

不过这里输出还在 Python 引用中，且两个 workload 不共享临时 tensor，因此它
不是 20--26 ms 锁等待的解释。

### 4. CUDA Event 在每轮新线程里创建

不是主要开销，但可以预先创建或使用固定 worker，避免反复触发 per-thread CUDA
状态初始化。

## 为什么表现为 VAE 阻塞 DiT，而不是反过来

VAE 本身没有反复调用 `cudaFuncSetAttribute`。它提交的是较长的 cuDNN
convolution。

DiT 的 FlashAttention 在每次调用前都执行需要 driver 写锁的：

```cpp
cudaFuncSetAttribute(...)
```

因此关系是：

```text
VAE 长 cuDNN kernel / driver activity
          |
          v
driver 写锁暂时不可得
          |
          v
DiT cudaFuncSetAttribute 等待
          |
          v
FlashAttention 尚未提交
          |
          v
DiT Green Context 出现空洞
```

所以即使 SM 已严格分区，host/driver 控制路径仍然没有被 Green Context 隔离。

VAE 不需要经过相同的高频写锁路径，所以不会产生对称的 330 ms 提交空洞。

## 最有价值的验证实验

### 1. 临时切换到 Torch SDPA

FastVideo 支持：

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

例如：

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  -o green+pytorch_demo/dit_vae_sdpa \
  python green+pytorch_demo/vae_dit_testbench.py \
    --dit-sms 112 \
    --warmup-iters 3 \
    --profile-iters 3
```

如果：

- `pthread_rwlock_wrlock` 的约 330 ms 消失；
- DiT gap 从约 397 ms 接近独跑的约 66 ms；
- 只剩 kernel duration 的 memory/L2 slowdown；

就能完全坐实这个根因。

SDPA 本身可能比 FlashAttention 慢，所以重点不是比较绝对 DiT 时间，而是比较：

```text
gap_colocated - gap_alone
```

### 2. 缓存 `cudaFuncSetAttribute`

真正针对 FlashAttention 的修复方向是：

- 每个具体 kernel specialization、device、smem size 只设置一次。
- 后续 forward 直接 launch，不再调用 `cudaFuncSetAttribute`。
- 缓存必须线程安全。

概念上类似：

```cpp
static std::once_flag attr_once;

std::call_once(attr_once, [&] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
});

kernel<<<grid, ..., smem_size, stream>>>(params);
```

但不能简单在函数模板里只放一个全局 flag，因为实际存在多种 kernel
specialization。缓存 key 至少应覆盖：

```text
device
kernel function pointer / specialization
smem_size
attribute type
```

更简单的实验补丁可以先针对当前固定的 head dim、dtype、causal 配置做
`std::once_flag`，验证收益后再泛化。

### 3. CUDA Graph

如果 DiT shape 固定，CUDA Graph replay 也可能绕过每轮 Python dispatch 和重复
attribute 设置。需要在 VAE 未运行时先完成 warmup/capture，再在 DiT Green
stream replay。

不过建议先做 SDPA A/B，因为它无需改源码，能最快验证结论。

## 当前结论

可以把整体 DiT 的 `+28.8%` 暂时拆成：

- 约 `8.8%`：kernel 本身因 L2/DRAM 等共享资源而变慢。
- 约 `20%`：FlashAttention 高频 `cudaFuncSetAttribute` 获取 NVIDIA driver
  写锁，被 VAE cuDNN 路径阻塞，从而产生 host submission gap。

所以当前第一优先级不再是采 L2 counter，而是先跑一次 `TORCH_SDPA` 共置实验。
若约 330 ms 的 `pthread_rwlock_wrlock` 消失，就可以确认代码层主因；之后再单独
研究剩余约 8.8% 的硬件资源竞争。
