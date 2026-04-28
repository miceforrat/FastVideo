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

import fastvideo.profiling.hack_transformer_block
from collections import defaultdict

def explore_node_dict(root: dict):
    stats = defaultdict(lambda: {
        "durations": [],
        "input_shapes": [],
        "output_shapes": [],
        "meta": []
    })
    def dfs(node: dict, path: tuple[str, ...]):
        name = node["name"]
        cur_path = path + (name,)
        key = "->".join(cur_path)

        stats[key]["durations"].append(node.get("duration_ms", 0))
        input_shapes = node.get("input_shapes")
        if input_shapes is not None and input_shapes!= {}:
            stats[key]["input_shapes"].append(input_shapes)
        output_shapes = node.get("output_shapes")
        if output_shapes is not None and output_shapes!= {}:
            stats[key]["output_shapes"].append(output_shapes)
        meta = node.get("meta")
        if meta is not None and output_shapes != {}:
            stats[key]["meta"].append(meta)
        # stats[key]["nodes"].append(node)

        sub_nodes = node.get("sub_nodes", {})
        for sub_name, nodes in sub_nodes.items():
            for child in nodes:
                dfs(child, cur_path)

    dfs(root, ())
    return stats

from statistics import mean

def summarize_multi_rank_stats(rank_stats: list[dict]):
    merged = defaultdict(lambda: {
        "durations": [],
        "input_shapes": [],
        "output_shapes": [],
        "meta": [],
    })

    # merge across ranks
    for stats in rank_stats:
        for chain, item in stats.items():
            merged[chain]["durations"].extend(item.get("durations", []))
            merged[chain]["input_shapes"].extend(item.get("input_shapes", []))
            merged[chain]["output_shapes"].extend(item.get("output_shapes", []))
            merged[chain]["meta"].extend(item.get("meta", []))

    # summarize
    summary = {}

    for chain, item in merged.items():
        durations = item["durations"]

        summary[chain] = {
            "count": len(durations),
            "duration_avg_ms": mean(durations) if durations else None,

            # 不做平均，直接列出所有 rank / all calls 的观测
            "input_shapes": item["input_shapes"],
            "output_shapes": item["output_shapes"],
            "meta": item["meta"],
        }

    return summary

def expand_summary_with_stats(summary: dict) -> dict:
    root = {
        "name": "stat_root",
        "sub_nodes": {},
    }

    for chain, stats in summary.items():
        names = chain.split("->") if isinstance(chain, str) else list(chain)

        cur = root
        for name in names:
            sub_nodes = cur.setdefault("sub_nodes", {})

            if name not in sub_nodes:
                sub_nodes[name] = {
                    "name": name,
                    "sub_nodes": {},
                }

            cur = sub_nodes[name]

        # 关键：直接挂 stats，不动原有字段
        cur["stats"] = stats
    return root

def print_excelwise_data(all_stats:dict):
    
    # 统计chunk无关的信息
    outer_res_names = []
    outer_vals = []
    pipeline_wise_stats = all_stats["sub_nodes"]["ROOT"]["sub_nodes"]["pipeline_stages"]
    whole_pipeline_time = pipeline_wise_stats["stats"]["duration_avg_ms"]
    outer_res_names.append("pipeline_stages")
    outer_vals.append(whole_pipeline_time)
    for name, pipeline_stage_stat in pipeline_wise_stats["sub_nodes"].items():
        outer_res_names.append(name)
        outer_vals.append(pipeline_stage_stat["stats"]["duration_avg_ms"])
    print("\t".join(outer_res_names))
    print("\t".join([str(outer_val) for outer_val in outer_vals]))
    
    denoising_chunks = pipeline_wise_stats["sub_nodes"]["CausalDMDDenosingStage"]["sub_nodes"]
    labels = ["chunk_idx", "dit_fwd", "sharding", "blocks", "all2gather", \
            "block_total", "prepare", "self_attn", "cross_attn_core", "ffn",\
            "fc_in", "act", "fc_out", \
            "hs_norm", "to_q", "to_k", "to_v", "q_rms_norm", "core_attn", "to_out", \
            "qkv_sp2head_all2all", "apply_rotary_emb", "maintaining_kv_cache","real_attn", "out_head2sp_all2all"]
    print("\t".join(labels))
    # chunk_vals = []

    for chunk_name, chunk_stats in denoising_chunks.items():
        res = []
        chunk_idx = chunk_name.split("_")[-1]
        res.append(chunk_idx)
        res.append(chunk_stats["sub_nodes"]["dit_forward"]["stats"]["duration_avg_ms"])
        dit_sons = chunk_stats["sub_nodes"]["dit_forward"]["sub_nodes"]
        res.append(dit_sons["model_sharding"]["stats"]["duration_avg_ms"])
        res.append(dit_sons["dit_blocks"]["stats"]["duration_avg_ms"])
        res.append(dit_sons["all_gather"]["stats"]["duration_avg_ms"])
        
        
        single_block_res = dit_sons["dit_blocks"]["sub_nodes"]["fwd_block"]
        res.append(single_block_res["stats"]["duration_avg_ms"])
        block_modules = single_block_res["sub_nodes"]
        res.append(block_modules["prepare"]["stats"]["duration_avg_ms"])
        res.append(block_modules["self_attn"]["stats"]["duration_avg_ms"])
        res.append(block_modules["cross_attn_core"]["stats"]["duration_avg_ms"])
        res.append(block_modules["ffn"]["stats"]["duration_avg_ms"])
        
        
        ffn_submodules = block_modules["ffn"]["sub_nodes"]
        res.append(ffn_submodules["fc_in"]["stats"]["duration_avg_ms"])
        res.append(ffn_submodules["act"]["stats"]["duration_avg_ms"])
        res.append(ffn_submodules["fc_out"]["stats"]["duration_avg_ms"])
                
        self_attn_submodules = block_modules["self_attn"]["sub_nodes"]
        res.append(self_attn_submodules["hs_norm"]["stats"]["duration_avg_ms"])
        res.append(self_attn_submodules["to_q"]["stats"]["duration_avg_ms"])
        res.append(self_attn_submodules["to_k"]["stats"]["duration_avg_ms"])
        res.append(self_attn_submodules["to_v"]["stats"]["duration_avg_ms"])
        res.append(self_attn_submodules["q_rms_norm"]["stats"]["duration_avg_ms"])
        res.append(self_attn_submodules["core_attn"]["stats"]["duration_avg_ms"])
        res.append(self_attn_submodules["to_out"]["stats"]["duration_avg_ms"])
        
        ca_submodules = self_attn_submodules["core_attn"]["sub_nodes"]
        res.append(ca_submodules["qkv_sp2head_all2all"]["stats"]["duration_avg_ms"])
        res.append(ca_submodules["apply_rotary_emb"]["stats"]["duration_avg_ms"])
        res.append(ca_submodules["maintaining_kv_cache"]["stats"]["duration_avg_ms"])
        res.append(ca_submodules["real_attn"]["stats"]["duration_avg_ms"])
        res.append(ca_submodules["out_head2sp_all2all"]["stats"]["duration_avg_ms"])
        
        # chunk_vals.append(res)
        print("\t".join([str(val) for val in res]))


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
        choices=[1, 2, 3,4],
        default=1,
        help="Number of GPUs to use"
    )

    # batch size
    parser.add_argument(
        "--bs",
        type=int,
        choices=[1, 2, 4, 8],
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
    module_profiling = False
    bs = args.bs
    warmup_iters = 5
    for _ in range(warmup_iters):
        results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, \
            save_video=False, sampling_param=sampling_param, num_videos_per_prompt=bs)
    
    run_times = 5
    gen_images_cnt = 8
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
    ranks_module_dict = {}
    for i in range(run_times):
        start_time = time.time()
        for j in range(video_gen_times):
            results = generator.generate_video(fake_prompt, output_path=OUTPUT_PATH, save_video=False, \
                sampling_param=sampling_param, num_videos_per_prompt=bs, do_profiling=module_profiling,\
                    memory_snapshot=False, nvtx_profiling=False)

            generate_times.append(results["generation_time"])
            peak_memory_mbs.append(results["peak_memory_mb"])
            ranks_module_dict = results["module_profiles_dict"]
            
        end_time = time.time()
        full_duration = end_time-start_time
        full_durations.append(full_duration)
        # get_global_time_profiler().submit_outer({"full_duration": full_duration})
        
        # print(f"forward {gen_images_cnt} videos time: {full_duration}")

    avg_full_duration = mean(full_durations)
    avg_peak_memory_mb = mean(peak_memory_mbs)
    avg_gen_time = mean(generate_times)
    avg_video_thpt = gen_images_cnt / avg_full_duration
    print("full_time\tvideo_thpt\tgen_time\tpeak_memory_mib")
    print(f"{avg_full_duration}\t{avg_video_thpt}\t{avg_gen_time}\t{avg_peak_memory_mb}")
    
    if module_profiling:
        per_rank_res = [explore_node_dict(module_dict) for rank_idx, module_dict in ranks_module_dict.items()]
        summarized_roots_stats = summarize_multi_rank_stats(per_rank_res)
        # for routes in summarized_roots_stats.keys():
        #     print(routes)
        
        expanded_stats = expand_summary_with_stats(summarized_roots_stats)
        # print(expanded_stats)
        print_excelwise_data(expanded_stats)
    
    
if __name__ == "__main__":
    main()
