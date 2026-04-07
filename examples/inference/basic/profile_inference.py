"""Inference profiling script for FastVideo.

Measures:
1. Inference latency (model loading + video generation)
2. GPU memory peak usage during generation
3. KV Cache size (estimated + actual via forward hook)

Usage:
    python profile_inference.py [--model MODEL_NAME] [--num-runs NUM_RUNS]

Example:
    python profile_inference.py --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --num-runs 2

Note:
    - Estimated KV Cache: computed from model config (rough approximation)
    - Actual KV Cache: captured via PyTorch forward hook if available
    - Both values are displayed for comparison during inference
"""

import argparse
import time
from typing import Optional

import torch

from fastvideo import VideoGenerator


OUTPUT_PATH = "video_samples"

# Test prompts
PROMPTS = [
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    "wide with interest. The playful yet serene atmosphere is complemented by soft "
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones.",
    
    "A majestic lion strides across the golden savanna, its powerful frame "
    "glistening under the warm afternoon sun. The tall grass ripples gently in "
    "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
    "embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic.",
]


def get_detailed_memory_info() -> dict[str, float]:
    """Get detailed GPU memory info with error handling."""
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "cuda_available": False
        }
    
    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "cuda_available": True
        }
    except Exception as e:
        print(f"Error getting CUDA memory: {e}")
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "cuda_available": False
        }


def print_memory_header(label: str):
    """Print a formatted memory measurement header."""
    print(f"\n{'='*70}")
    print(f"  {label:<50}")
    print(f"{'='*70}")


def print_memory_info(label: str, info: dict[str, float]):
    """Print formatted memory info."""
    if not info.get("cuda_available", True):
        print(f"{label:40s} CUDA not available - memory info N/A")
        return
    
    allocated = info.get("allocated_gb", 0.0)
    reserved = info.get("reserved_gb", 0.0)
    max_allocated = info.get("max_allocated_gb", 0.0)
    
    if max_allocated > 0:
        print(f"{label:40s} Allocated: {allocated:6.2f} GB | "
              f"Reserved: {reserved:6.2f} GB | Max: {max_allocated:6.2f} GB")
    else:
        print(f"{label:40s} Allocated: {allocated:6.2f} GB | "
              f"Reserved: {reserved:6.2f} GB")


def estimate_kv_cache_size(
    num_frames: int,
    num_heads: int = 12,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float32,
) -> tuple[float, dict]:
    """Estimate KV cache size for a video generation.
    
    Args:
        num_frames: Number of frames in generated video
        num_heads: Number of attention heads (default: 32 for Wan2.1)
        head_dim: Dimension per head (default: 64)
        dtype: Data type (default: float32 = 4 bytes)
    
    Returns:
        (total_bytes, {k_cache_mb, v_cache_mb, total_mb})
    """
    bytes_per_element = 4  # float32
    
    # KV cache shape per token: (batch_size, num_heads, seq_len, head_dim)
    # For video: seq_len is roughly num_frames * num_patches
    # Typical: 4096-16384 tokens per frame (patches + conditioning)
    patch_tokens_per_frame = 4096  # rough estimate
    total_tokens = num_frames * patch_tokens_per_frame
    
    # K cache + V cache = 2 * (batch_size * total_tokens * num_heads * head_dim * bytes)
    batch_size = 1  # typical inference
    k_cache_bytes = batch_size * total_tokens * num_heads * head_dim * bytes_per_element
    v_cache_bytes = k_cache_bytes  # V cache same size as K cache
    total_cache_bytes = k_cache_bytes + v_cache_bytes
    
    return total_cache_bytes, {
        "k_cache_mb": k_cache_bytes / (1024**2),
        "v_cache_mb": v_cache_bytes / (1024**2),
        "total_mb": total_cache_bytes / (1024**2),
    }


def register_kv_cache_hook(model) -> dict:
    """Register forward hook to capture actual KV cache statistics.
    
    Returns a dictionary that will be populated with cache info after forward pass.
    """
    cache_stats = {"captured": False, "size_mb": 0.0, "num_blocks": 0}
    
    def hook_fn(module, input, output):
        """Hook to extract KV cache info from transformer output."""
        cache_stats["captured"] = False
        
        # Output format from transformer.forward(return_kv=True):
        # (hidden_states, kv_cache_dict)
        if isinstance(output, tuple) and len(output) >= 2:
            kv_cache_dict = output[1]
            
            if isinstance(kv_cache_dict, dict):
                total_size_bytes = 0
                block_count = 0
                
                for block_idx in sorted(kv_cache_dict.keys()):
                    cache_entry = kv_cache_dict[block_idx]
                    
                    # Handle tuple format: (k, v) or list format
                    if isinstance(cache_entry, (tuple, list)):
                        k_cache, v_cache = cache_entry[0], cache_entry[1]
                    elif isinstance(cache_entry, dict):
                        # May have keys like 'k', 'v'
                        k_cache = cache_entry.get("k")
                        v_cache = cache_entry.get("v")
                    else:
                        continue
                    
                    # Compute size if both k and v tensors exist
                    if k_cache is not None and hasattr(k_cache, "numel"):
                        total_size_bytes += k_cache.numel() * k_cache.element_size()
                    if v_cache is not None and hasattr(v_cache, "numel"):
                        total_size_bytes += v_cache.numel() * v_cache.element_size()
                    
                    block_count += 1
                
                cache_stats["size_mb"] = total_size_bytes / (1024**2)
                cache_stats["num_blocks"] = block_count
                cache_stats["captured"] = True
    
    # Try to find and hook the transformer's forward method
    # Look for transformer/DiT modules in the pipeline
    hooked = False
    for name, module in model.named_modules():
        # Look for transformer or DiT modules
        if any(keyword in name.lower() for keyword in ["transformer", "dit", "ditmodel"]):
            try:
                module.register_forward_hook(hook_fn)
                hooked = True
                print(f"Debug: Successfully hooked module: {name}")
                break
            except Exception as e:
                print(f"Debug: Failed to hook {name}: {e}")
                continue
    
    if not hooked:
        print("Debug: No suitable transformer module found for hooking")
    
    return cache_stats


def profile_inference(
    model_name: str,
    num_runs: int = 1,
    disable_cpu_offload: bool = False,
):
    """Profile video generation inference.
    
    Args:
        model_name: Hugging Face model identifier
        num_runs: Number of inference runs to benchmark
        disable_cpu_offload: If True, don't use CPU offloading (requires more GPU memory)
    """
    
    # ===== CUDA Environment Check =====
    print(f"\n{'*'*70}")
    print(f"CUDA Environment Check")
    print(f"{'*'*70}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        device_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {device_props.name}")
        print(f"Total GPU memory: {device_props.total_memory / (1024**3):.1f} GB")
    else:
        print("Warning: CUDA not available - memory measurements will be 0")
    print(f"PyTorch version: {torch.__version__}")
    print(f"{'*'*70}\n")
    
    print(f"\n{'*'*70}")
    print(f"FastVideo Inference Profiling")
    print(f"{'*'*70}")
    print(f"Model: {model_name}")
    print(f"Runs: {num_runs}")
    print(f"Output directory: {OUTPUT_PATH}\n")
    
    # ===== Model Loading =====
    print_memory_header("MODEL LOADING PHASE")
    
    initial_mem = get_detailed_memory_info()
    print_memory_info("Before loading", initial_mem)
    
    start_load = time.perf_counter()
    
    generator = VideoGenerator.from_pretrained(
        model_name,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=disable_cpu_offload,
        vae_cpu_offload=disable_cpu_offload,
        text_encoder_cpu_offload=not disable_cpu_offload,
        pin_cpu_memory=True,
    )
    
    load_time = time.perf_counter() - start_load
    loaded_mem = get_detailed_memory_info()
    
    print_memory_info("After loading ", loaded_mem)
    print(f"\n{'Model loading time':<40s} {load_time:>8.2f} seconds")
    if loaded_mem["cuda_available"]:
        print(f"{'GPU memory increase':<40s} {loaded_mem['allocated_gb'] - initial_mem['allocated_gb']:>8.2f} GB")
    else:
        print(f"{'GPU memory increase':<40s} N/A (CUDA not available)")
    
    # ===== Register KV Cache Hook =====
    # Try to hook into the executor to capture real KV cache sizes
    cache_stats_list = []
    try:
        if hasattr(generator, "executor") and hasattr(generator.executor, "pipeline"):
            pipeline = generator.executor.pipeline
            cache_stats = register_kv_cache_hook(pipeline)
            cache_stats_list.append(cache_stats)
    except Exception as e:
        print(f"Warning: Could not register KV cache hook: {e}")
    
    # ===== Inference Runs =====
    print_memory_header("INFERENCE PHASE")
    
    generation_times = []
    peak_memories = []
    
    for run_idx in range(num_runs):
        prompt = PROMPTS[run_idx % len(PROMPTS)]
        prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
        
        print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        print(f"Prompt: {prompt_short}")
        
        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        print("\nMemory before generation:")
        pre_gen_mem = get_detailed_memory_info()
        print_memory_info("  ", pre_gen_mem)
        
        # Reset peak memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Generate video
        start_gen = time.perf_counter()
        video = generator.generate_video(
            prompt,
            output_path=OUTPUT_PATH,
            save_video=True,
        )
        gen_time = time.perf_counter() - start_gen
        
        # Get peak memory and current memory immediately after generation
        torch.cuda.synchronize()  # Ensure all operations are complete
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        post_gen_mem = get_detailed_memory_info()
        
        generation_times.append(gen_time)
        peak_memories.append(peak_memory_gb)
        
        print(f"\nGeneration results:")
        print(f"  Latency:          {gen_time:>8.2f} seconds")
        if post_gen_mem["cuda_available"]:
            print(f"  Peak GPU memory:  {peak_memory_gb:>8.2f} GB")
            print(f"  Current memory:   {post_gen_mem['allocated_gb']:>8.2f} GB (allocated) | "
                  f"{post_gen_mem['reserved_gb']:>8.2f} GB (reserved)")
        else:
            print(f"  Peak GPU memory:  N/A (CUDA not available)")
            print(f"  Current memory:   N/A (CUDA not available)")
        
        # Estimate and display KV cache size
        # Note: This is an estimate; actual values depend on model config
        num_frames = 45  # typical for Wan2.1-T2V
        # kv_cache_bytes, kv_estimates = estimate_kv_cache_size(num_frames)
        
        # print(f"\nKV Cache (estimated, num_frames={num_frames}):")
        # print(f"  K cache:         {kv_estimates['k_cache_mb']:>8.2f} MB")
        # print(f"  V cache:         {kv_estimates['v_cache_mb']:>8.2f} MB")
        # print(f"  Total estimate:  {kv_estimates['total_mb']:>8.2f} MB")
        
        # Display actual KV cache size if hook captured it
        if cache_stats_list and cache_stats_list[0].get("captured"):
            actual_cache_mb = cache_stats_list[0].get("size_mb", 0)
            num_blocks = cache_stats_list[0].get("num_blocks", 0)
            print(f"\nKV Cache (ACTUAL, measured by hook):")
            print(f"  Total actual:    {actual_cache_mb:>8.2f} MB ({num_blocks} blocks)")
            # print(f"  Estimated vs Actual: {kv_estimates['total_mb']/actual_cache_mb if actual_cache_mb > 0 else 1:.2f}x")
    
    # ===== Summary =====
    print_memory_header("SUMMARY")
    
    avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
    max_gen_time = max(generation_times) if generation_times else 0
    min_gen_time = min(generation_times) if generation_times else 0
    
    avg_peak_mem = sum(peak_memories) / len(peak_memories) if peak_memories else 0
    max_peak_mem = max(peak_memories) if peak_memories else 0
    
    print(f"\nLatency Statistics:")
    print(f"  Average generation time: {avg_gen_time:>8.2f} seconds")
    print(f"  Min:                     {min_gen_time:>8.2f} seconds")
    print(f"  Max:                     {max_gen_time:>8.2f} seconds")
    
    print(f"\nMemory Statistics:")
    if torch.cuda.is_available():
        print(f"  Average peak GPU usage:  {avg_peak_mem:>8.2f} GB")
        print(f"  Max peak GPU usage:      {max_peak_mem:>8.2f} GB")
    else:
        print(f"  Average peak GPU usage:  N/A (CUDA not available)")
        print(f"  Max peak GPU usage:      N/A (CUDA not available)")
    
    print(f"\nModel Loading:")
    print(f"  Load time:               {load_time:>8.2f} seconds")
    if loaded_mem["cuda_available"]:
        print(f"  Model footprint:         {loaded_mem['allocated_gb']:>8.2f} GB")
    else:
        print(f"  Model footprint:         N/A (CUDA not available)")
    
    print(f"\nKV Cache Monitoring:")
    if cache_stats_list and cache_stats_list[0].get("captured"):
        print(f"  Status:                  ✓ Hook successfully captured actual cache sizes")
    else:
        print(f"  Status:                  ✗ Hook not available; using estimated values only")
    
    print(f"\n{'*'*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Profile FastVideo inference performance"
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Model name or path (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of inference runs to benchmark (default: 2)",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable CPU offloading (requires more GPU memory)",
    )
    
    args = parser.parse_args()
    
    profile_inference(
        model_name=args.model,
        num_runs=args.num_runs,
        disable_cpu_offload=args.no_cpu_offload,
    )


def test_cuda_environment():
    """Test CUDA environment and print diagnostic info."""
    print("CUDA Environment Diagnostics:")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            print(f"  torch.cuda.current_device(): {device}")
            props = torch.cuda.get_device_properties(device)
            print(f"  GPU Name: {props.name}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.1f} GB")
            
            # Test memory allocation
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"  Test allocation successful: {test_tensor.shape}")
            allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"  Memory allocated after test: {allocated:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  CUDA test failed: {e}")
    else:
        print("  CUDA not available - check your PyTorch installation")
    
    print()


if __name__ == "__main__":
    # Run CUDA diagnostics first
    test_cuda_environment()
    
    main()
