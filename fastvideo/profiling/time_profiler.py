from __future__ import annotations


import torch
import torch.distributed as dist
import numpy as np

_GLOBAL_TIME_PROFILER=None

def _submit_to_list_dict(info: dict[str, int], target):
    for key in info.keys():
        if key not in target.keys():
            target[key] = []
        target[key].append(info[key])

def _get_cur_rank():
    if not dist.is_initialized():
        cur_rank = 0
    else:
        cur_rank = dist.get_rank()
    return cur_rank

class TimeProfiler():
    def __init__(self):
        self.ncu_profile:bool = False
        self.time_profile:bool = False #开启全局profile轮次
        self.chunk = 0
        # self.rank_chunks = {}
        self.chunk_results = {}
        self.outer_results = {}
        self.rank_profiling = {} # 控制局部profile
    
    def set_rank_profiling(self, profiling: bool):
        rank = _get_cur_rank()
        self.rank_profiling[rank] = profiling
    
    def get_rank_profiling(self):
        rank = _get_cur_rank()
        if rank not in self.rank_profiling.keys():
            self.rank_profiling[rank] = False
        return self.rank_profiling[rank]
    
    def set_time_profile(self, time_profile:bool):
        self.time_profile = time_profile
    
    # def set_cur_rank_chunk(self, chunk:int):
    #     cur_rank = _get_cur_rank()
    #     self.rank_chunks[cur_rank] = chunk    
    #     if chunk not in self.chunk_results.keys():
    #         self.chunk_results[chunk] = {}
    
    def set_chunk(self, chunk:int):
        self.chunk = chunk
        if chunk not in self.chunk_results.keys():
            self.chunk_results[chunk] = {}
     
    def submit_by_chunk(self, info:dict[str, int]):
        # 提交denoising内部针对某个chunk的profile信息
        # cur_rank = _get_cur_rank()
        _submit_to_list_dict(info, self.chunk_results[self.chunk])    
    
    def submit_outer(self, info: dict[str, int]):
        # 提交更全局的统计信息
        _submit_to_list_dict(info, self.outer_results)
    
    def print_chunkwise_results(self,
                                calc_variance_set: set = None,
                                calc_p_dict: dict[str, int]= None):
        """
            calc_variance_set：需要计算方差的key
            calc_p_dict：需要计算尾部数据的dict
        """
        
        res_names = ["chunk_idx"]
        res_vals = []
        num_added = False
        for idx in self.chunk_results.keys():
            cur_chunk_res = [idx]
            for key in self.chunk_results[idx].keys():
                if not num_added:
                    res_names.append(key)
                data_list = self.chunk_results[idx][key]
                cur_chunk_res.append(np.mean(data_list))
                if key in calc_variance_set:
                    if not num_added:
                        res_names.append(f"{key}_variance")
                    cur_chunk_res.append(np.var(data_list))
                if key in calc_p_dict:
                    threshold = calc_p_dict[key]
                    if not num_added:
                        res_names.append(f"{key}_p{threshold}")
                    cur_chunk_res.append(np.percentile(data_list, threshold))
            res_vals.append(cur_chunk_res)
            num_added = True
        print("\t".join(res_names))
        for chunk_res in res_vals:
            print("\t".join(str(x) for x in chunk_res))
    
    def print_outer_results(self):
        res_names = []
        res_vals = []
        for key in self.outer_results.keys():
            res_names.append(key)
            res_vals.append(np.mean(self.outer_results[key]))
        print("\t".join(res_names))
        print("\t".join(str(x) for x in res_vals))
    
    def merge_chunkwise_results(self, other_chunkwise_results):
        for idx in other_chunkwise_results.keys():
            if idx not in self.chunk_results.keys():
                self.chunk_results[idx] = other_chunkwise_results[idx]
                continue
            for key in other_chunkwise_results[idx].keys():
                if key not in self.chunk_results[idx]:
                    self.chunk_results[idx][key] = other_chunkwise_results[idx][key]
                else:
                    self.chunk_results[idx][key] = self.chunk_results[idx][key] + other_chunkwise_results[idx][key]
    
    def merge_outer_results(self, other_outer_results):
        print(f"other outer res:")
        for key in other_outer_results.keys():
            if key not in self.outer_results.keys():
                self.outer_results[key] = other_outer_results[key]
            else:
                self.outer_results[key] = self.outer_results[key] + other_outer_results[key]


def get_global_time_profiler() -> TimeProfiler:
    global _GLOBAL_TIME_PROFILER
    if _GLOBAL_TIME_PROFILER == None:
        _GLOBAL_TIME_PROFILER = TimeProfiler()
        
    assert isinstance(_GLOBAL_TIME_PROFILER, TimeProfiler)
    return _GLOBAL_TIME_PROFILER


class TimeProfilingEvent():
    def __init__(self, caller_profilling:bool = True):
        # caller_profilling: 调用者是否开启局部profilling，默认为True
        self.caller_profilling=caller_profilling
        self.event = None
        if self.should_profile():
            self.event = torch.cuda.Event(enable_timing=True)
            
    def record(self):
        if self.should_profile():
            self.event.record()
            
    def should_profile(self):
        return get_global_time_profiler().time_profile and self.caller_profilling
    
    def elapsed_time(self, end_event: TimeProfilingEvent):
        if self.should_profile() and end_event.should_profile():
            return self.event.elapsed_time(end_event.event)
        return 0
    