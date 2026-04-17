import os
import time
from fastvideo import VideoGenerator, SamplingParam

from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

import torch
import argparse
import time
from copy import deepcopy
from fastvideo.profiling.time_profiler import get_global_time_profiler


# 下面这段patching大部分没用了，因为我发现他们原生就能观测peak memory （mb）
logger = init_logger(__name__)

# @torch.no_grad()
# def hack_pipeline_forward(
#     self,
#     batch: ForwardBatch,
#     fastvideo_args: FastVideoArgs,
# ) -> ForwardBatch:
#     """
#     Generate a video or image using the pipeline.

#     Args:
#         batch: The batch to generate from.
#         fastvideo_args: The inference arguments.
#     Returns:
#         ForwardBatch: The batch with the generated video or image.
#     """

#     if not self.post_init_called:
#         self.post_init()
        
#     logger.info("Running pipeline stages: %s", self._stage_name_mapping.keys(), local_main_process_only=False)
#     # stages_duration = []
#     stages_duration_dict = {}
#     start_time = time.perf_counter()
#     last_time = start_time
#     # try:
#     for stage in self.stages:
#         batch = stage(batch, fastvideo_args)
#         end_time = time.perf_counter()
#         duration = end_time-last_time
#         last_time = end_time
#         # stages_duration.append(duration)
#         stages_duration_dict[type(stage).__name__] = duration
#     # batch.extra["durations"] = stages_duration
#     if get_global_time_profiler().time_profile:
#         get_global_time_profiler().submit_outer(stages_duration_dict)
    
#     # print(f"fastvideo_args: {fastvideo_args}")
    
#     return batch

# ComposedPipelineBase.forward = hack_pipeline_forward

import fastvideo.profiling.hack_transformer_block

OUTPUT_PATH = "video_samples_causal"
def main():
    
    parser = argparse.ArgumentParser(
        prog="profile sf ar diffusers",
        description="args for profiling ar diffusers",
        epilog="Example: examples/inference/basic/basic_self_forcing_causal_profiling.py \
            --fsdp --num_gpus 2 --bs 1"
    )
    # 是否开启 FSDP
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP when using multiple GPUs"
    )

    # GPU 数量
    parser.add_argument(
        "--num_gpus",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Number of GPUs to use"
    )

    # batch size
    parser.add_argument(
        "--bs",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Batch size"
    )

    args = parser.parse_args()

    if args.num_gpus < 2:
        args.fsdp=False

    print("fsdp =", args.fsdp)
    print("num_gpus =", args.num_gpus)
    print("bs =", args.bs)
    
    
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    model_name = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=args.num_gpus,
        use_fsdp_inference=args.fsdp, # set to True if GPU is out of memory
        text_encoder_cpu_offload=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        log_kv_cache_size=True,
        dp_decoding=True
    )

    sampling_param = SamplingParam.from_pretrained(model_name)

    fake_prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "+\
    "wide with interest. The playful yet serene atmosphere is complemented by soft "+\
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    prompts = [fake_prompt] * 2
    bs = args.bs
    warmup_iters = 3
    for _ in range(warmup_iters):
        results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, \
            save_video=False, sampling_param=sampling_param, num_videos_per_prompt=bs)
    
    run_times = 3
    gen_images_cnt = 1
    # assert run_times % chunk_size == 0
    assert gen_images_cnt % bs == 0
    
    video_gen_times = gen_images_cnt // bs
    
    all_stage_durations = []
    full_durations = []
    generate_times = []
    peak_memory_mbs = []
    kv_cache_mibs = []
    crossattn_mibs = []
    
    outer_results = []
    chunkwise_results = []
    for i in range(run_times):
        start_time = time.time()
        for j in range(video_gen_times):
            results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, save_video=False, \
                sampling_param=sampling_param, num_videos_per_prompt=bs, do_profiling=True)
            # print(f"peak_memory_mb: {results["peak_memory_mb"]}; generation_time: {results["generation_time"]}")
            # print(f"durations: {results["durations"]}")
            # all_stage_durations.append(results["durations"])
            # peak_memory_mbs.append(results["peak_memory_mb"])
            # generate_times.append(results["generation_time"])
            # kv_cache_mibs.append(results["kv_cache_mib"])
            # crossattn_mibs.append(results["crossattn_mib"])
            part_res = {}
            part_res["generation_time"] = results["generation_time"]
            part_res["peak_memory_mb"] = results["peak_memory_mb"]
            get_global_time_profiler().submit_outer(part_res)
            # part_res["kv_cache_mib"] = results["kv_cache_mib"]
            # part_res["crossattn_mib"] = results["crossattn_mib"]
            for outer in results["outer"]:
                outer_results.append(outer)
            for chunkwise in results["chunkwise"]:
                chunkwise_results.append(chunkwise)
            
        end_time = time.time()
        full_duration = end_time-start_time
        # full_durations.append(full_duration)
        get_global_time_profiler().submit_outer({"full_duration": full_duration})
        
        print(f"forward {gen_images_cnt} videos time: {full_duration}")

    for outer in outer_results:
        get_global_time_profiler().merge_outer_results(outer)
    
    for chunkwise in chunkwise_results:
        get_global_time_profiler().merge_chunkwise_results(chunkwise)
    
    get_global_time_profiler().print_outer_results()
    var_chunkwise_opts = set()
    var_chunkwise_opts.add("chunk_denoising_time")
    var_chunkwise_opts.add("ts_block_total")
    var_chunkwise_opts.add("ca_sp2head")
    var_chunkwise_opts.add("ca_head2sp")
    p_dict = {}
    p_dict["ts_block_total"]=99
    p_dict["chunk_denoising_time"]=99
    get_global_time_profiler().print_chunkwise_results(calc_variance_set=var_chunkwise_opts,
                                                       calc_p_dict=p_dict)
    # start_time = time.time()
    
    # results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, save_video=False, sampling_param=sampling_param, num_videos_per_prompt=2)
    # # results = generator.generate_batches_video(2, prompts=prompts, output_path=OUTPUT_PATH, save_video=False, sampling_param=sampling_param)
    # print(f"peak_memory_mb: {results["peak_memory_mb"]}; generation_time: {results["generation_time"]}")
    
    # end_time = time.time()
    # print(f"batching forward 2 videos time: {end_time-start_time}")    
    # avg_duration = sum(full_durations) / len(full_durations)
    # avg_generate_time = sum(generate_times) / len(generate_times)
    # all_stage_durations_ave = [sum(col) / len(col) for col in zip(*all_stage_durations)]
    
    # max_peak_memory = max(peak_memory_mbs)
    
    # avg_kv_cache_mib = sum(kv_cache_mibs) / len(kv_cache_mibs)
    # avg_crossattn_mib = sum(crossattn_mibs) / len(crossattn_mibs)
    # row = [
    #     avg_duration,
    #     avg_generate_time,
    #     *all_stage_durations_ave,   # 展开 list
    #     max_peak_memory,
    #     avg_kv_cache_mib,
    #     avg_crossattn_mib,
    # ]
    
    # print("\t".join(map(str, row)))
    
if __name__ == "__main__":
    main()
