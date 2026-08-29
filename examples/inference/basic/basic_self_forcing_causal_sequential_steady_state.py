"""Measure two same-process requests with the stock sequential pipeline.

The same ``VideoGenerator`` serves both requests so model loading is excluded
from both reported generation times.
"""

from fastvideo import SamplingParam, VideoGenerator

MODEL_NAME = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
PROMPT = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers. "
    "Mid-shot, warm tones."
)


def run_request(
    generator: VideoGenerator,
    request_index: int,
) -> dict:
    result = generator.generate_video(
        PROMPT,
        output_path=(
            "video_samples_causal_sequential_steady_state"
        ),
        save_video=False,
        sampling_param=SamplingParam.from_pretrained(MODEL_NAME),
        num_frames=81,
        seed=1024 + request_index,
    )
    summary = {
        "request": request_index,
        "size": result["size"],
        "generation_time": result["generation_time"],
        "peak_memory_mb": result["peak_memory_mb"],
    }
    print(summary, flush=True)
    return summary


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        MODEL_NAME,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        dp_decoding=False,
    )

    first = run_request(generator, request_index=1)
    steady = run_request(generator, request_index=2)
    print(
        {
            "first_request_seconds": first["generation_time"],
            "steady_state_seconds": steady["generation_time"],
            "first_minus_second_seconds": (
                first["generation_time"] - steady["generation_time"]
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
