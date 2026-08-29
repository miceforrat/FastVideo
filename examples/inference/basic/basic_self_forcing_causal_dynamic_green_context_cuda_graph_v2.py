"""Run a real SFWan2.1 request with the graph-safe dynamic GC pipeline."""

from fastvideo import SamplingParam, VideoGenerator

MODEL_NAME = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
OUTPUT_PATH = "video_samples_causal_dynamic_gc_cuda_graph_v2"


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
        override_pipeline_cls_name=(
            "WanCausalDMDDynamicGreenContextCUDAGraphV2Pipeline"),
    )
    sampling_param = SamplingParam.from_pretrained(MODEL_NAME)
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow "
        "sunflowers, its eyes wide with interest. The playful yet serene "
        "atmosphere is complemented by soft natural light filtering through "
        "the petals. Mid-shot, warm and cheerful tones."
    )
    result = generator.generate_video(
        prompt,
        output_path=OUTPUT_PATH,
        save_video=True,
        sampling_param=sampling_param,
        num_frames=81,
        seed=1024,
    )
    print({
        "size": result["size"],
        "generation_time": result["generation_time"],
        "peak_memory_mb": result["peak_memory_mb"],
        "video_path": result["video_path"],
    })


if __name__ == "__main__":
    main()
