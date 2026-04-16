from fastvideo.models.dits.causal_wanvideo import CausalWanTransformerBlock
import fastvideo.models.dits.causal_wanvideo
from torch.nn.attention.flex_attention import BlockMask
from fastvideo.platforms import AttentionBackendEnum, current_platform
import torch


elapsed_times = []

warmup_iters = 3
profile_times=7

def debug_print_inputs(
    hidden_states,
    encoder_hidden_states,
    temb,
    freqs_cis,
    block_mask,
    kv_cache=None,
    crossattn_cache=None,
    current_start=0,
    cache_start=None,
):
    import torch

    def tensor_info(x):
        return f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"

    def maybe_tensor(x):
        return isinstance(x, torch.Tensor)

    def print_tensor(name, x):
        if x is None:
            print(f"{name}: None")
        elif maybe_tensor(x):
            print(f"{name}: {tensor_info(x)}")
        else:
            print(f"{name}: type={type(x)}")

    def print_tuple(name, x):
        if x is None:
            print(f"{name}: None")
        elif isinstance(x, tuple):
            print(f"{name}: tuple(len={len(x)})")
            for i, item in enumerate(x):
                if maybe_tensor(item):
                    print(f"  {name}[{i}]: {tensor_info(item)}")
                else:
                    print(f"  {name}[{i}]: type={type(item)}")
        else:
            print(f"{name}: type={type(x)}")

    def print_dict(name, d):
        if d is None:
            print(f"{name}: None")
            return
        if not isinstance(d, dict):
            print(f"{name}: type={type(d)}")
            return
        print(f"{name}: dict(keys={list(d.keys())})")
        for k, v in d.items():
            if maybe_tensor(v):
                print(f"  {name}[{k}]: {tensor_info(v)}")
            elif isinstance(v, dict):
                print(f"  {name}[{k}]: dict(len={len(v)})")
            else:
                print(f"  {name}[{k}]: type={type(v)}")

    def print_block_mask(name, bm):
        if bm is None:
            print(f"{name}: None")
            return
        print(f"{name}: type={type(bm)}")
        # 尝试打印常见属性（不保证都有）
        for attr in ["shape", "seq_len", "mask", "block_size"]:
            if hasattr(bm, attr):
                val = getattr(bm, attr)
                if maybe_tensor(val):
                    print(f"  {name}.{attr}: {tensor_info(val)}")
                else:
                    print(f"  {name}.{attr}: {val}")

    print("\n===== DEBUG INPUTS =====")

    print_tensor("hidden_states", hidden_states)
    print_tensor("encoder_hidden_states", encoder_hidden_states)
    print_tensor("temb", temb)

    print_tuple("freqs_cis", freqs_cis)
    print_block_mask("block_mask", block_mask)

    print_dict("kv_cache", kv_cache)
    print_dict("crossattn_cache", crossattn_cache)

    print(f"current_start: {current_start}")
    print(f"cache_start: {cache_start}")

    print("========================\n")

class HackCausalWanTransformerBlock(CausalWanTransformerBlock):
    
    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 local_attn_size: int = -1,
                 sink_size: int = 0,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__(dim,
                         ffn_dim,
                         num_heads,
                         local_attn_size,
                         sink_size,
                         qk_norm,
                         cross_attn_norm,
                         eps,
                         added_kv_proj_dim,
                         supported_attention_backends,
                         prefix)
        self.profile_time=False
        self.layer_idx = int(prefix.split(".")[-1])
        self.fwd_times=0
        self.iter_fwds=7 * 4 # chunk nums * timesteps
#         print(f"input_args: \
# \tdim: {dim} \
# \tffn_dim: {ffn_dim} \
# \tnum_heads: {num_heads} \
# \tlocal_attn_size: {local_attn_size} \
# \tsink_size: {sink_size} \
# \tqk_norm: {qk_norm} \
# \tcross_attn_norm: {cross_attn_norm} \
# \teps: {eps} \
# \tadded_kv_proj_dim: {added_kv_proj_dim} \
# \tsupported_attn_backends: {supported_attention_backends} \
# \tprefix: {prefix}")
        if self.layer_idx >30 or self.layer_idx < 0:
            self.profile_time=True
            

    def _new_timer_events(self):
        return {
            "start": torch.cuda.Event(enable_timing=True),
            "prepare_end": torch.cuda.Event(enable_timing=True),

            # self-attn 细分
            "qkv_end": torch.cuda.Event(enable_timing=True),
            "attn_core_end": torch.cuda.Event(enable_timing=True),
            "self_attn_end": torch.cuda.Event(enable_timing=True),

            # 后面
            "cross_attn_end": torch.cuda.Event(enable_timing=True),
            "ffn_end": torch.cuda.Event(enable_timing=True),
        }

    def _print_profile(self, events):
        if self.profile_time:
            torch.cuda.synchronize()

            prepare_ms = events["start"].elapsed_time(events["prepare_end"])

            qkv_ms = events["prepare_end"].elapsed_time(events["qkv_end"])
            attn_core_ms = events["qkv_end"].elapsed_time(events["attn_core_end"])
            out_proj_ms = events["attn_core_end"].elapsed_time(events["self_attn_end"])

            self_attn_total = events["prepare_end"].elapsed_time(events["self_attn_end"])

            cross_attn_ms = events["self_attn_end"].elapsed_time(events["cross_attn_end"])
            ffn_ms = events["cross_attn_end"].elapsed_time(events["ffn_end"])
            total_ms = events["start"].elapsed_time(events["ffn_end"])

            print(f"profiling ts block {self.layer_idx}:")
            print(f"\tprepare:        {prepare_ms:.6f}")

            print(f"\tself_attn_total:{self_attn_total:.6f}")
            print(f"\t  qkv_proj:     {qkv_ms:.6f}")
            print(f"\t  attn_core:    {attn_core_ms:.6f}")
            print(f"\t  out_proj:     {out_proj_ms:.6f}")

            print(f"\tcross_attn:     {cross_attn_ms:.6f}")
            print(f"\tffn:            {ffn_ms:.6f}")
            print(f"\ttotal:          {total_ms:.6f}")
        
    def record(self, event):
        if self.profile_time:
            event.record()
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        original_seq_len: int,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        self.warmup_fwds = warmup_iters * self.iter_fwds
        do_profile = self.fwd_times >= self.warmup_fwds and self.fwd_times < self.warmup_fwds+profile_times

        if not do_profile:
            hidden_states = super().forward(
                hidden_states,
                encoder_hidden_states,
                temb,
                freqs_cis,
                block_mask,
                original_seq_len,
                kv_cache,
                crossattn_cache,
                current_start,
                cache_start,
            )
            self.fwd_times += 1
            return hidden_states

        # if do_profile and self.profile_time:
        #     debug_print_inputs(
        #         hidden_states,
        #         encoder_hidden_states,
        #         temb,
        #         freqs_cis,
        #         block_mask,
        #         kv_cache,
        #         crossattn_cache,
        #         current_start,
        #         cache_start,
        #     )
        
        # print("outof warmup brs!!!!!!!!!!!!!!!!!!!!!")
        with torch.cuda.nvtx.range(f"ts_block_{self.layer_idx}_full"):
            events = self._new_timer_events()
            self.record(events["start"])

            # hidden_states.shape: [batch_size, seq_length, inner_dim]
            # temb.shape: [batch_size, num_frames, 6, inner_dim]
            with torch.cuda.nvtx.range(f"ts_block_{self.layer_idx}_prepare"):
                if hidden_states.dim() == 4:
                    hidden_states = hidden_states.squeeze(1)
                num_frames = temb.shape[1]
                frame_seqlen = hidden_states.shape[1] // num_frames    
                bs, seq_length, _ = hidden_states.shape
                orig_dtype = hidden_states.dtype
                # assert orig_dtype != torch.float32
                e = self.scale_shift_table + temb
                # e.shape: [batch_size, num_frames, 6, inner_dim]
                assert e.shape == (bs, num_frames, 6, self.hidden_dim)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
                    6, dim=2)
            # *_msa.shape: [batch_size, num_frames, 1, inner_dim]
            # assert shift_msa.dtype == torch.float32
            self.record(events["prepare_end"])
            
            # 1. Self-attention
            with torch.cuda.nvtx.range(f"ts_block_{self.layer_idx}_qkv_proj"):
                norm_hidden_states = (self.norm1(hidden_states).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
                                (1 + scale_msa) + shift_msa).flatten(1, 2)
                query, _ = self.to_q(norm_hidden_states)
                key, _ = self.to_k(norm_hidden_states)
                value, _ = self.to_v(norm_hidden_states)

                if self.norm_q is not None:
                    query = self.norm_q.forward_native(query)
                if self.norm_k is not None:
                    key = self.norm_k.forward_native(key)

                query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
                key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
                value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

            self.record(events["qkv_end"])
            
            with torch.cuda.nvtx.range(f"ts_block_{self.layer_idx}_attn_core"):
                # print(f"q shape before ca: {query.shape}")
                # print(f"k shape before ca: {key.shape}")
                # print(f"v shape before ca: {value.shape}")
                # print(f"freqs cis size: {freqs_cis[0].shape}, {freqs_cis[1].shape}")
                attn_output = self.attn1(query, key, value, freqs_cis, block_mask, original_seq_len, kv_cache, current_start, cache_start)
            
            self.record(events["attn_core_end"])
            
            with torch.cuda.nvtx.range(f"ts_block_{self.layer_idx}_out_proj"):
                attn_output = attn_output.flatten(2)
                attn_output, _ = self.to_out(attn_output)
                attn_output = attn_output.squeeze(1)

                null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
                norm_hidden_states, hidden_states = self.self_attn_residual_norm(
                    hidden_states, attn_output, gate_msa, null_shift, null_scale)
                norm_hidden_states, hidden_states = norm_hidden_states.to(
                    orig_dtype), hidden_states.to(orig_dtype)

            self.record(events["self_attn_end"])
            # 2. Cross-attention
            with torch.cuda.nvtx.range(f"ts_block_{self.layer_idx}_cross_attn"):
                attn_output = self.attn2(norm_hidden_states,
                                        context=encoder_hidden_states,
                                        context_lens=None,
                                        crossattn_cache=crossattn_cache)
                norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
                    hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

            self.record(events["cross_attn_end"])
            # 3. Feed-forward
            with torch.cuda.nvtx.range(f"ts_block_{self.layer_idx}_ffn"):
                ff_output = self.ffn(norm_hidden_states)
                hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
            
            self.record(events["ffn_end"])

        self._print_profile(events)

        self.fwd_times += 1
        return hidden_states
        
fastvideo.models.dits.causal_wanvideo.CausalWanTransformerBlock = HackCausalWanTransformerBlock