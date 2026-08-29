"""Benchmark sequential and optimal-GC VAE CUDA Graph pipelines.

Run one variant per process.  The first request performs lazy initialization
and, for the CUDA Graph variant, VAE warmup and capture.  Only subsequent
requests are included in the steady-state statistics.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Any

from fastvideo import SamplingParam, VideoGenerator

MODEL_NAME = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, "
    "its eyes wide with interest. The playful yet serene atmosphere is "
    "complemented by soft natural light filtering through the petals. "
    "Mid-shot, warm and cheerful tones."
)
V2_PIPELINE_NAME = (
    "WanCausalDMDDynamicGreenContextCUDAGraphV2Pipeline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("sequential", "v2-optimal-graph"),
        required=True,
        help="Pipeline variant to benchmark in this process.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=6,
        help="Total requests, including warmup/capture requests.",
    )
    parser.add_argument(
        "--discard-requests",
        type=int,
        default=1,
        help="Number of leading requests excluded from statistics.",
    )
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--save-last-video",
        action="store_true",
        help="Save only the final request for output validation.",
    )
    parser.add_argument(
        "--output-path",
        default="video_samples_causal_vae_graph_optimal_gc_ab",
    )
    args = parser.parse_args()
    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.discard_requests < 0:
        parser.error("--discard-requests must be non-negative")
    if args.discard_requests >= args.num_requests:
        parser.error(
            "--discard-requests must be smaller than --num-requests"
        )
    return args


def create_generator(variant: str) -> VideoGenerator:
    override_pipeline_cls_name = (
        V2_PIPELINE_NAME if variant == "v2-optimal-graph" else None
    )
    return VideoGenerator.from_pretrained(
        MODEL_NAME,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        dp_decoding=False,
        override_pipeline_cls_name=override_pipeline_cls_name,
    )


def summarize(
    variant: str,
    request_results: list[dict[str, Any]],
    discard_requests: int,
) -> dict[str, Any]:
    steady_results = request_results[discard_requests:]
    steady_times = [
        float(result["generation_time"]) for result in steady_results
    ]
    steady_peak_memory = [
        float(result["peak_memory_mb"]) for result in steady_results
    ]
    return {
        "variant": variant,
        "discarded_requests": discard_requests,
        "steady_request_count": len(steady_results),
        "steady_generation_times_seconds": steady_times,
        "steady_mean_seconds": statistics.fmean(steady_times),
        "steady_median_seconds": statistics.median(steady_times),
        "steady_std_seconds": (
            statistics.stdev(steady_times)
            if len(steady_times) > 1
            else 0.0
        ),
        "steady_min_seconds": min(steady_times),
        "steady_max_seconds": max(steady_times),
        "steady_peak_memory_mb": steady_peak_memory,
        "max_steady_peak_memory_mb": max(steady_peak_memory),
    }


def main() -> None:
    args = parse_args()
    generator = create_generator(args.variant)
    request_results: list[dict[str, Any]] = []

    for request_index in range(args.num_requests):
        save_video = (
            args.save_last_video
            and request_index == args.num_requests - 1
        )
        result = generator.generate_video(
            PROMPT,
            output_path=args.output_path,
            save_video=save_video,
            sampling_param=SamplingParam.from_pretrained(MODEL_NAME),
            num_frames=args.num_frames,
            seed=args.seed,
        )
        logging_info = result.get("logging_info")
        graph_memory = (
            logging_info.get("vae_cuda_graph_memory_mib")
            if isinstance(logging_info, dict)
            else None
        )
        request_summary = {
            "variant": args.variant,
            "request": request_index,
            "included_in_steady_statistics": (
                request_index >= args.discard_requests
            ),
            "generation_time": result["generation_time"],
            "peak_memory_mb": result["peak_memory_mb"],
            "size": result["size"],
            "vae_cuda_graph_memory_mib": graph_memory,
            "video_path": result.get("video_path"),
        }
        request_results.append(request_summary)
        print(request_summary, flush=True)

    print(
        summarize(
            args.variant,
            request_results,
            args.discard_requests,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
