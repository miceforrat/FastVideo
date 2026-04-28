from __future__ import annotations
import torch
# from contextlib import contextmanager

def _get_tensor_infos(real_tensor: torch.Tensor):
    res = {}
    res["shape"] = real_tensor.shape
    res["dtype"] = real_tensor.dtype
    res["elem_size"] = real_tensor.element_size()
    return res

class ModuleNode():
    
    def __init__(self,
                 name: str,
                 father: ModuleNode|None,
                 inputs: dict[str, torch.Tensor]|None = None,
                 meta: dict | None = None):
        self.name = name
        self.father = father
        self.input_shapes:dict = {}
        self.output_shapes:dict = {}
        self.meta = meta
        self.link:dict = {}
        self.sub_nodes:dict[str, list[ModuleNode]] = {}
        self.duration_ms = 0
        self.begin:torch.cuda.Event|None = torch.cuda.Event(enable_timing=True)
        self.end:torch.cuda.Event|None = torch.cuda.Event(enable_timing=True)
        if inputs is not None:
            for tensor_name in inputs.keys():
                self.input_shapes[tensor_name] = _get_tensor_infos(inputs[tensor_name])
                
        self.begin.record()
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "meta": self.meta,
            "links": self.link,
            "sub_nodes": {
                sub_name: [node.to_dict() for node in nodes]
                for sub_name, nodes in self.sub_nodes.items()
            },
        }
        
    
    def add_sub_node(self,
                     sub_name:str,
                     sub_inputs: dict[str, torch.Tensor]|None = None,
                     sub_meta: dict | None = None,
                     sub_prev_name: str=None):
        if sub_name not in self.sub_nodes.keys():
            self.sub_nodes[sub_name] = []
        
        new_sub_node = ModuleNode(sub_name, self, sub_inputs, sub_meta)
        self.sub_nodes[sub_name].append(new_sub_node)
        if sub_prev_name is not None:
            assert sub_prev_name in self.sub_nodes.keys()
            self.link[sub_prev_name] = (sub_prev_name, sub_name)
        return new_sub_node
            
    def _exit_only_record(self, outputs:dict[str, torch.Tensor]|None = None):
        self.end.record()
        if outputs is not None:
            for tensor_name in outputs.keys():
                self.output_shapes[tensor_name] = _get_tensor_infos(outputs[tensor_name])
        
    def recursively_calc_node_duration(self):
        if self.begin is not None and self.end is not None:
            self.duration_ms = self.begin.elapsed_time(self.end)
            self.begin=None
            self.end=None
        for nodes in self.sub_nodes.values():
            for node in nodes:
                node.recursively_calc_node_duration()
    
    def exit_node(self,
                outputs:dict[str, torch.Tensor]|None = None, 
                need_sync_calculation_durations:bool=False):
        self._exit_only_record(outputs)
        if need_sync_calculation_durations:
            torch.cuda.synchronize()
            self.recursively_calc_node_duration()

        
class SimpleProfiler():
    def __init__(self):
        self.do_module_profiling = False
        self.nvtx_profiling=False
        self.root_node = ModuleNode("ROOT", None)
        self.cur_node = self.root_node
        self.depth = 0
    
    def set_module_profiling(self, profiling:bool):
        self.do_module_profiling = profiling
    
    def set_nvtx_profiling(self, nvtx_profiling:bool):
        self.nvtx_profiling = nvtx_profiling
    
    def enter(self, 
              module_name:str,
              module_inputs: dict[str, torch.Tensor]|None = None,
              module_meta: dict | None = None,
              prev_module_name: str=None):
        self.depth += 1
        if self.nvtx_profiling:
            torch.cuda.nvtx.range_push(module_name)
        
        if not self.do_module_profiling:
            return
        new_node = self.cur_node.add_sub_node(module_name,
                                   module_inputs,
                                   module_meta,
                                   prev_module_name)
        
        self.cur_node = new_node
    
    def exit(self,
             outputs:dict[str, torch.Tensor]|None = None,
             sync_and_calc_durations:bool=False):
        assert self.depth > 0, "can not exit more than enter"
        
        self.depth -= 1
        
        if self.nvtx_profiling:
            torch.cuda.nvtx.range_pop()
        if not self.do_module_profiling:
            return            
        self.cur_node.exit_node(outputs, sync_and_calc_durations)
        self.cur_node = self.cur_node.father
    
    def collect_all_nodes_as_dict(self):
        return self.root_node.to_dict()
        

_SIMPLE_PROFILER:SimpleProfiler=None

def get_current_simple_profiler():
    global _SIMPLE_PROFILER
    if _SIMPLE_PROFILER is None:
        _SIMPLE_PROFILER=SimpleProfiler()
    return _SIMPLE_PROFILER
