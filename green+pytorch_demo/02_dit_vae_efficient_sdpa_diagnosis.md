# 02：DiT/VAE Efficient SDPA 共置诊断

本文承接 `dit_vae_colocation_diagnosis.md`，记录真正关闭 Flash SDP 后的
Nsight Systems 验证结果。

## 实验配置

报告：

```text
green+pytorch_demo/dit_vae_sdpa_real.nsys-rep
```

本轮显式关闭 Flash SDP，只保留 memory-efficient SDPA：

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
```

报告中没有 `flash_fwd_kernel`、`pytorch_flash::mha_fwd` 或
`_scaled_dot_product_flash_attention`，可以确认 Flash SDP 已真正关闭。

实际启动参数为：

```text
--dit-sms 136
--warmup-iters 3
--profile-iters 3
```

RTX 5090 一共有约 170 个 SM，因此本轮大致为：

```text
DiT: 136 SM
VAE: 34 SM
```

这与此前的 112:58 分区不同。本轮内部的 alone/colocated 对照有效，但不能直接
比较跨报告的绝对时间。VAE-alone 从约 1.75 s 变成约 2.82 s，主要是因为 VAE
可用 SM 从约 58 降到约 34，并不是关闭 Flash SDP 影响了 VAE。

## 精确时间结果

DiT measured-alone（chunk 4--6）：

```text
2477.79 / 2477.87 / 2479.81 ms
平均约 2478.49 ms
```

DiT colocated（chunk 7--9）：

```text
3101.74 / 3074.60 / 3083.31 ms
平均约 3086.55 ms
```

因此：

```text
DiT slowdown ≈ 24.5%
```

VAE measured-alone 平均约 2823.06 ms，colocated 平均约 2942.26 ms：

```text
VAE slowdown ≈ 4.2%
```

非对称干扰依然存在。

## DiT busy/gap 分解

| DiT | Alone | Colocated | 变化 |
|---|---:|---:|---:|
| GPU span | 2478.5 ms | 3086.6 ms | +24.5% |
| Kernel busy | 2420.1 ms | 2496.7 ms | +3.2% |
| Kernel gap | 58.4 ms | 589.9 ms | +531.5 ms |

本轮结果比此前更清楚：

- kernel 本身累计只慢约 3.2%；
- 绝大多数端到端退化来自额外约 531.5 ms 的 kernel gap；
- L2/DRAM 引起的 kernel slowdown 仍存在，但在本配置下是次要因素；
- host-side submission lock 是主因。

## Efficient Attention 仍调用 `cudaFuncSetAttribute`

关闭 Flash SDP 后，attention 主 kernel 变为：

```text
fmha_cutlassF_bf16_aligned_64x128_rf_sm80
```

调用栈确认 PyTorch 使用 memory-efficient attention：

```text
at::native::scaled_dot_product_attention
  -> _scaled_dot_product_efficient_attention_cuda
  -> _efficient_attention_forward
  -> cudaFuncSetAttribute
  -> cuFuncSetAttribute
  -> NVIDIA libcuda.so
  -> pthread_rwlock_wrlock
```

因此，虽然 Flash 已关闭，PyTorch efficient attention 仍会在 launch 前执行
`cudaFuncSetAttribute`，进入 NVIDIA driver 的全局写锁路径。

共置期间三个 DiT 线程的数据：

```text
pthread_rwlock_wrlock:
146 次
累计 1698.84 ms / 3 轮
单次最长 44.04 ms
```

换算到每轮：

```text
1698.84 / 3 ≈ 566.3 ms/chunk
```

而每轮 DiT gap 增量为：

```text
589.9 - 58.4 ≈ 531.5 ms/chunk
```

两者非常接近。锁等待略大于最终新增 gap，是因为部分锁等待与已有 kernel 执行或
原始间隙重叠，不会全部线性反映在 GPU span 上。

完整证据链为：

```text
VAE 长 cuDNN convolution
        |
        v
DiT efficient attention 调用 cudaFuncSetAttribute
        |
        v
cuFuncSetAttribute 等待 libcuda 全局写锁
        |
        v
attention kernel 没有及时提交
        |
        v
DiT Green Context 出现约 0.53 s 额外空洞
```

## DiT kernel 本身的退化

| DiT kernel | 共置 slowdown |
|---|---:|
| Efficient attention 主 kernel | +1.8% |
| CUTLASS GEMM | +1.3% |
| Elementwise | +14.0% |
| Vectorized elementwise | +16.7% |
| Unrolled elementwise | +22.3% |
| LayerNorm | +13.1% |
| Cat/copy | +24.5% |

memory 类 kernel 仍比计算 kernel 更敏感，说明 L2/DRAM 竞争依然存在；但由于
它们占比有限，整体 kernel busy time 只增加约 3.2%。

本轮可以把 DiT 的约 24.5% slowdown 粗略分成：

```text
约 3.2%：kernel 自身变慢
约 21%：host-side driver lock 导致的 submission gap
```

## 三组 Attention 实验比较

| Backend | Attention 实现 | `cudaFuncSetAttribute` | 共置锁等待 |
|---|---|---:|---:|
| External FA | 外部 FlashAttention | 有 | 约 330 ms/chunk |
| Torch SDPA 默认 | PyTorch Flash SDP | 有 | 约 332 ms/chunk |
| Torch efficient SDPA | CUTLASS efficient attention | 有 | 约 566 ms/chunk |

因此可以排除“只是外部 FlashAttention 实现有问题”。更准确的根因是：

> 当前这些 optimized attention 实现都会在 forward launch 路径中重复执行
> `cudaFuncSetAttribute`。当另一个 Green Context 中有长时间 cuDNN
> convolution 执行时，`cuFuncSetAttribute` 会在 NVIDIA driver 的全局写锁上
> 等待。Green Context 隔离了 SM，却没有隔离这个 host/driver 控制路径。

## 下一步

### 1. 小规模强制 Math SDPA

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

Math backend 更可能不调用 `cudaFuncSetAttribute`，但可能显著变慢、生成完整
attention matrix，甚至 OOM。应先减小序列长度或只执行一个 DiT forward，验证
调用栈和 gap，不建议直接运行完整 benchmark。

核心问题是：

```text
没有 cudaFuncSetAttribute 后，
VAE 运行期间 DiT 是否还出现对应的 submission gap？
```

### 2. CUDA Graph

如果 DiT shape 固定，可以在 VAE 开始前完成 warmup 和 graph capture，共置时
只执行 graph replay，绕过每轮 attention backend 的 host launch 配置路径。

如果 graph replay 不再高频调用 `cudaFuncSetAttribute`，就可能同时保留 optimized
attention 性能，并消除主要的 host-side gap。

## 更新后的结论

1. Green Context C++ 分区逻辑没有导致设备侧错误串行。
2. Python GIL 不是主要间隙来源。
3. 问题不是外部 FlashAttention 独有。
4. PyTorch Flash SDP 和 efficient SDPA 都存在同类
   `cudaFuncSetAttribute` 写锁等待。
5. 本轮 DiT 约 24.5% 的 slowdown 中，只有约 3.2% 来自 kernel 本身，绝大多数
   来自 host/driver submission gap。
6. VAE 只慢约 4.2%，因为它不走这个高频 attribute-setting 路径。
7. 下一步最有价值的是小规模 Math SDPA 验证或 CUDA Graph replay，而不是继续
   更换 optimized attention backend。
