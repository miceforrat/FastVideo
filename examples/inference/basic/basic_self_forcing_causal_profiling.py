import os
import time
from fastvideo import VideoGenerator, SamplingParam

from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

import torch

# 下面这段patching没用了，因为我发现他们原生就能观测peak memory （mb）
logger = init_logger(__name__)

@torch.no_grad()
def hack_pipeline_forward(
    self,
    batch: ForwardBatch,
    fastvideo_args: FastVideoArgs,
) -> ForwardBatch:
    """
    Generate a video or image using the pipeline.

    Args:
        batch: The batch to generate from.
        fastvideo_args: The inference arguments.
    Returns:
        ForwardBatch: The batch with the generated video or image.
    """
    import time
    import torch
    import torch.distributed as dist

    if not self.post_init_called:
        self.post_init()

    # Identify worker / rank / device for logging
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    # Reset worker-level peak stats for the whole pipeline execution
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        start_alloc = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)
    else:
        start_alloc = 0
        start_reserved = 0
    stages_duration = []
    start_time = time.perf_counter()
    last_time = start_time

    logger.info("Running pipeline stages: %s", self._stage_name_mapping.keys(), local_main_process_only=False)

    # try:
    for stage in self.stages:
        batch = stage(batch, fastvideo_args)
        end_time = time.perf_counter()
        duration = end_time-last_time
        last_time = end_time
        stages_duration.append(duration)
        batch.extra["durations"] = stages_duration
    return batch
    # finally:
    #     batch.extra["durations"] = stages_duration
    #     end_time = time.perf_counter()

    #     if torch.cuda.is_available():
    #         torch.cuda.synchronize(device)

    #         peak_alloc = torch.cuda.max_memory_allocated(device)
    #         peak_reserved = torch.cuda.max_memory_reserved(device)
    #         end_alloc = torch.cuda.memory_allocated(device)
    #         end_reserved = torch.cuda.memory_reserved(device)

    #         mib = 1024 ** 2
    #         # logger.info(
    #         #     "[worker forward][rank=%d][cuda:%s] "
    #         #     "time_s=%.4f | "
    #         #     "start_alloc_mb=%.2f start_reserved_mb=%.2f | "
    #         #     "peak_alloc_mb=%.2f peak_reserved_mb=%.2f | "
    #         #     "end_alloc_mb=%.2f end_reserved_mb=%.2f",
    #         #     rank,
    #         #     str(device),
    #         #     end_time - start_time,
    #         #     start_alloc / mib,
    #         #     start_reserved / mib,
    #         #     peak_alloc / mib,
    #         #     peak_reserved / mib,
    #         #     end_alloc / mib,
    #         #     end_reserved / mib,
    #         #     local_main_process_only=False
    #         # )
    #         # logger.info(f"duration: {stages_duration}",  local_main_process_only=False)
    #     else:
    #         logger.info(
    #             "[worker forward][rank=%d] time_s=%.4f (CUDA unavailable)",
    #             rank,
    #             end_time - start_time,
    #             local_main_process_only=False
    #         )

ComposedPipelineBase.forward = hack_pipeline_forward

# prompts=[
#     "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
#     "wide with interest. The playful yet serene atmosphere is complemented by soft "
#     "natural light filtering through the petals. Mid-shot, warm and cheerful tones.",
#     "A majestic lion strides across the golden savanna, its powerful frame "
#     "glistening under the warm afternoon sun. The tall grass ripples gently in "
#     "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
#     "embodying the raw energy of the wild. Low angle, steady tracking shot, "
#     "cinematic."
# ]


OUTPUT_PATH = "video_samples_causal"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    model_name = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=8,
        use_fsdp_inference=True, # set to True if GPU is out of memory
        text_encoder_cpu_offload=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        log_kv_cache_size=True
    )

    sampling_param = SamplingParam.from_pretrained(model_name)

    # prompt = (
    #     "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    #     "wide with interest. The playful yet serene atmosphere is complemented by soft "
    #     "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    # )
    import time


    fake_prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "+\
    "wide with interest. The playful yet serene atmosphere is complemented by soft "+\
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    prompts = [fake_prompt] * 2
    bs = 3
    warmup_iters = 3
    for _ in range(warmup_iters):
        results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, save_video=False, sampling_param=sampling_param, num_videos_per_prompt=bs)
    
    run_times = 2
    chunk_size = 2
    # assert run_times % chunk_size == 0
    assert chunk_size % bs == 0
    
    gen_times = chunk_size // bs
    
    denoising_durations = []
    durations = []
    peak_memory_mbs = []
    for i in range(run_times):
        start_time = time.time()
        for j in range(gen_times):
            results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, save_video=False, sampling_param=sampling_param, num_videos_per_prompt=bs)
            print(f"peak_memory_mb: {results["peak_memory_mb"]}; generation_time: {results["generation_time"]}")
            denoising_durations.append(results["durations"][-2])
            peak_memory_mbs.append(results["peak_memory_mb"])
        end_time = time.time()
        full_duration = end_time-start_time
        durations.append(full_duration)
        print(f"forward 2 videos time: {full_duration}")

    # start_time = time.time()
    
    # results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, save_video=False, sampling_param=sampling_param, num_videos_per_prompt=2)
    # # results = generator.generate_batches_video(2, prompts=prompts, output_path=OUTPUT_PATH, save_video=False, sampling_param=sampling_param)
    # print(f"peak_memory_mb: {results["peak_memory_mb"]}; generation_time: {results["generation_time"]}")
    
    # end_time = time.time()
    # print(f"batching forward 2 videos time: {end_time-start_time}")    
    avg_duration = sum(durations) / len(durations)
    max_peak_memory = max(peak_memory_mbs)
    avg_denoising = sum(denoising_durations) / len(denoising_durations)
    print(f"avg total duration: {avg_duration}")
    print(f"avg denoising duration: {avg_denoising}")
    print(f"max peak memory: {max_peak_memory}")
    
    
if __name__ == "__main__":
    main()
