# Exploration Log: DiT/VAE Green Context Colocation

## Status: draft

## Context

Investigate why SM-partitioned DiT and VAE workloads slow down when colocated,
despite improving when run independently with fewer SMs. The current prototype
is in `green+pytorch_demo/` and runs both workloads in separate Green Context
streams on one GPU.

## Progress

- [x] Inspect the Green Context benchmark and timing boundaries.
- [x] Confirm the profiling environment in `cjh-Fastvideo-cu128`.
- [ ] Establish repeatable isolated and colocated baselines with fixed clocks.
- [ ] Attribute slowdown to individual DiT and VAE kernel classes.
- [ ] Test memory/L2, power/clock, launch, and library-workspace hypotheses.
- [ ] Evaluate mitigations after identifying the dominant resource.

## Findings

- The container has `CAP_SYS_ADMIN`, Nsight Systems 2024.6, and Nsight Compute
  2025.1; the repository is mounted at `/FastVideo` and Python is in `.venv`.
- The benchmark uses CUDA Events on each Green Context stream, so its per-stream
  elapsed times include GPU-side queuing/stalls during colocation as intended.
- DiT uses BF16 autocast. The VAE input and decoder are currently FP32 and execute
  multiple frame-wise 3D convolution/cache operations, making memory bandwidth,
  L2 traffic, cuDNN workspace, and power/clock pressure initial hypotheses.
- Green Context partitions SM resources, but shared L2, DRAM bandwidth, copy
  engines, power/thermal budget, and front-end/library resources remain shared.

## Mistakes / Dead Ends

- Do not begin with a full Nsight Compute metric set over the complete model. It
  replays kernels, perturbs overlap, and makes colocated attribution ambiguous.
- Wall-clock time alone cannot distinguish GPU resource contention from CPU
  submission or synchronization effects.

## Proposed Standardization

If the investigation converges, turn the staged experiment matrix and profiler
commands into a GPU colocation diagnosis workflow.


## CUDA Graph follow-up

- Added `green+pytorch_demo/vae_dit_testbench_cuda_graph.py` to compare
  eager DiT against VAE Graph replay; the observed DiT slowdown fell from about
  28.8% with eager VAE to about 8.9% with VAE Graph replay.
- Nsight Systems showed only one `pthread_rwlock_wrlock` call (1.84 ms total)
  in the VAE Graph report, versus roughly 979 ms in the eager colocation run.
- Added `green+pytorch_demo/vae_dit_testbench_dual_cuda_graph.py` to measure
  DiT Graph alone and concurrent DiT/VAE Graph replay in the same process.
- The dual-Graph script intentionally uses separate graph-private memory pools
  because the two graphs replay concurrently. Runtime validation is pending.

- First DiT Graph capture attempt failed in `get_nd_rotary_pos_embed` because
  `full_grid.to(device)` copied pageable CPU memory during capture.
- The dual-Graph testbench now installs a local CUDA-resident RoPE cache after
  eager measurements. Warmup populates it before capture, and capture/replay
  reuse the same CUDA tensor addresses without modifying core FastVideo code.

- A subsequent DiT capture exposed a CPU-first timestep frequency construction
  in `visual_embedding.timestep_embedding`. The testbench now replaces it
  locally with an equivalent CUDA-resident frequency cache populated during
  warmup. A scan of the active causal forward found no other obvious
  CPU-first tensor construction after the RoPE, grid-size, and timestep fixes.
