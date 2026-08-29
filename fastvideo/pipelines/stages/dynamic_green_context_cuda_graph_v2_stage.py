# SPDX-License-Identifier: Apache-2.0
"""Dynamic Green Context pipeline with per-chunk VAE CUDA Graphs."""

from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.green_context import GreenContextPairPool
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.pingpong_stage import (
    PingPongDenoisingDecodingStage,
)
from fastvideo.profiling.small_node_profiler import (
    get_current_simple_profiler,
)
from fastvideo.utils import PRECISION_TO_TYPE, bytes_to_mib, tensor_bytes

logger = init_logger(__name__)


class DynamicGreenContextCUDAGraphV2DenoisingDecodingStage(
        PingPongDenoisingDecodingStage):
    """Replay per-chunk VAE graphs alongside eager DiT chunks."""

    DIT_SMS_BY_CHUNK = {
        1: 100,
        2: 94,
        3: 100,
        4: 106,
        5: 112,
        6: 112,
    }
    NUM_CHUNKS = 7

    def __init__(
        self,
        transformer,
        scheduler,
        vae,
        transformer_2=None,
        pipeline=None,
    ) -> None:
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            transformer_2=transformer_2,
            pipeline=pipeline,
        )
        self._green_context_pool: GreenContextPairPool | None = None
        self._vae_graphs: list[torch.cuda.CUDAGraph] | None = None
        self._vae_graph_inputs: list[torch.Tensor] = []
        self._vae_graph_outputs: list[torch.Tensor] = []
        self._vae_graph_cache: list[torch.Tensor | None] | None = None
        self._vae_graph_signature: tuple[object, ...] | None = None
        self._vae_graph_memory_mib: dict[str, float] = {}

    def _get_green_context_pool(self) -> GreenContextPairPool:
        if self._green_context_pool is None:
            self._green_context_pool = GreenContextPairPool(
                self.DIT_SMS_BY_CHUNK.values(),
                device=torch.cuda.current_device(),
                ignore_sm_coscheduling=True,
            )
            for requested_sms in (
                    self._green_context_pool.requested_dit_sm_counts):
                pair = self._green_context_pool[requested_sms]
                if pair.actual_dit_sms != requested_sms:
                    raise RuntimeError(
                        f"Requested {requested_sms} DiT SMs, but CUDA "
                        f"provisioned {pair.actual_dit_sms}.")
                if (pair.actual_dit_sms + pair.actual_vae_sms !=
                        pair.total_sms):
                    raise RuntimeError(
                        "Green Context pair does not cover all device SMs.")
        return self._green_context_pool

    def _vae_streams(
        self,
        pool: GreenContextPairPool,
    ) -> list[torch.cuda.Stream]:
        return [
            pool[self.DIT_SMS_BY_CHUNK[chunk_idx]].vae_stream
            for chunk_idx in range(1, self.NUM_CHUNKS)
        ] + [pool.full_vae_stream]

    def _prepare_graph_safe_denormalization(
        self,
        sample_latents: torch.Tensor,
    ) -> None:
        cfg = self.vae.config
        if not (hasattr(cfg, "latents_mean")
                and hasattr(cfg, "latents_std")):
            raise NotImplementedError(
                "The VAE graph path currently requires latents_mean and "
                "latents_std in the VAE config.")
        self._vae_latents_mean = torch.as_tensor(
            cfg.latents_mean,
            device=sample_latents.device,
            dtype=sample_latents.dtype,
        ).view(1, -1, 1, 1, 1)
        self._vae_latents_std = torch.as_tensor(
            cfg.latents_std,
            device=sample_latents.device,
            dtype=sample_latents.dtype,
        ).view(1, -1, 1, 1, 1)

    def _graph_safe_streaming_decode(
        self,
        latents: torch.Tensor,
        fastvideo_args: FastVideoArgs,
        cache: list[torch.Tensor | None] | None,
        is_first_chunk: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        autocast_enabled = (vae_dtype != torch.float32
                            and not fastvideo_args.disable_autocast)
        latents = (latents * self._vae_latents_std
                   + self._vae_latents_mean)
        if cache is None:
            cache = self.vae.get_streaming_cache()
        with torch.autocast(
                device_type="cuda",
                dtype=vae_dtype,
                enabled=autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not autocast_enabled:
                latents = latents.to(vae_dtype)
            image, cache = self.vae.streaming_decode(
                latents,
                cache,
                is_first_chunk,
            )
        image = (image / 2 + 0.5).clamp(0, 1)
        return image, cache

    @torch.no_grad()
    def _initialize_vae_graphs(
        self,
        sample_latents: torch.Tensor,
        fastvideo_args: FastVideoArgs,
        pool: GreenContextPairPool,
    ) -> None:
        signature = (
            tuple(sample_latents.shape),
            sample_latents.dtype,
            sample_latents.device,
            fastvideo_args.pipeline_config.vae_precision,
            fastvideo_args.disable_autocast,
            fastvideo_args.pipeline_config.vae_tiling,
        )
        if self._vae_graphs is not None:
            if signature != self._vae_graph_signature:
                raise RuntimeError(
                    "VAE CUDA Graphs were captured for a different request "
                    f"signature: {self._vae_graph_signature} != {signature}.")
            return

        streams = self._vae_streams(pool)
        self._prepare_graph_safe_denormalization(sample_latents)
        allocated_before = torch.cuda.memory_allocated(sample_latents.device)
        reserved_before = torch.cuda.memory_reserved(sample_latents.device)
        warmup_input = torch.zeros_like(sample_latents)

        logger.info(
            "Warming VAE on %d capture streams before CUDA Graph capture",
            len({stream.cuda_stream for stream in streams}),
        )
        for stream in dict.fromkeys(streams):
            warmup_cache = None
            with torch.cuda.stream(stream):
                _, warmup_cache = self._graph_safe_streaming_decode(
                    warmup_input,
                    fastvideo_args,
                    cache=warmup_cache,
                    is_first_chunk=True,
                )
                _, warmup_cache = self._graph_safe_streaming_decode(
                    warmup_input,
                    fastvideo_args,
                    cache=warmup_cache,
                    is_first_chunk=False,
                )
            stream.synchronize()
        del warmup_cache
        del warmup_input
        torch.cuda.synchronize()

        graph_pool = torch.cuda.graph_pool_handle()
        graphs: list[torch.cuda.CUDAGraph] = []
        static_inputs: list[torch.Tensor] = []
        static_outputs: list[torch.Tensor] = []
        capture_cache: list[torch.Tensor | None] | None = None

        logger.info("Capturing %d chained VAE CUDA Graphs", self.NUM_CHUNKS)
        for chunk_idx, stream in enumerate(streams):
            static_input = torch.empty_like(sample_latents)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                    graph,
                    pool=graph_pool,
                    stream=stream,
                    capture_error_mode="global"):
                static_output, capture_cache = (
                    self._graph_safe_streaming_decode(
                        static_input,
                        fastvideo_args,
                        cache=capture_cache,
                        is_first_chunk=(chunk_idx == 0),
                    ))
            graphs.append(graph)
            static_inputs.append(static_input)
            static_outputs.append(static_output)

        torch.cuda.synchronize()
        allocated_after = torch.cuda.memory_allocated(sample_latents.device)
        reserved_after = torch.cuda.memory_reserved(sample_latents.device)
        mib = 1024**2
        self._vae_graph_memory_mib = {
            "allocated_delta": (allocated_after - allocated_before) / mib,
            "reserved_delta": (reserved_after - reserved_before) / mib,
            "allocated_total": allocated_after / mib,
            "reserved_total": reserved_after / mib,
        }
        logger.info(
            "VAE graph capture memory MiB: allocated_delta=%.2f, "
            "reserved_delta=%.2f, allocated_total=%.2f, reserved_total=%.2f",
            self._vae_graph_memory_mib["allocated_delta"],
            self._vae_graph_memory_mib["reserved_delta"],
            self._vae_graph_memory_mib["allocated_total"],
            self._vae_graph_memory_mib["reserved_total"],
        )

        self._vae_graphs = graphs
        self._vae_graph_inputs = static_inputs
        self._vae_graph_outputs = static_outputs
        self._vae_graph_cache = capture_cache
        self._vae_graph_signature = signature

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        self._validate_supported_request(batch, fastvideo_args)
        if batch.latents is None:
            raise ValueError("latents must be provided")

        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            and not fastvideo_args.disable_autocast)
        latents = batch.latents
        _, _, num_latent_frames, height, width = latents.shape
        patch_size = self.transformer.config.arch_config.patch_size
        self.frame_seq_length = height * width // (
            patch_size[-1] * patch_size[-2])

        timesteps = torch.tensor(
            fastvideo_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
        ).cpu()
        if fastvideo_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat((
                self.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32),
            ))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())

        prompt_embeds = batch.prompt_embeds
        if prompt_embeds:
            if not isinstance(prompt_embeds[0], torch.Tensor):
                raise TypeError("prompt_embeds[0] must be a tensor")
            if batch.num_videos_per_prompt > 1:
                prompt_embeds[0] = prompt_embeds[0].repeat_interleave(
                    batch.num_videos_per_prompt,
                    dim=0,
                )
        if not prompt_embeds or torch.isnan(prompt_embeds[0]).any():
            raise ValueError("prompt embeddings are empty or contain NaN")

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {"encoder_attention_mask": batch.prompt_attention_mask},
        )
        kv_cache = self._initialize_kv_cache(
            batch_size=latents.shape[0],
            dtype=target_dtype,
            device=latents.device,
        )
        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=latents.shape[0],
            max_text_len=(fastvideo_args.pipeline_config
                          .text_encoder_configs[0].arch_config.text_len),
            dtype=target_dtype,
            device=latents.device,
        )

        if num_latent_frames % self.num_frames_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frames_per_block")
        num_chunks = num_latent_frames // self.num_frames_per_block
        if num_chunks != self.NUM_CHUNKS:
            raise ValueError(
                "The calibrated Green Context schedule requires exactly "
                f"{self.NUM_CHUNKS} chunks, but got {num_chunks}.")

        pool = self._get_green_context_pool()
        decode_enabled = fastvideo_args.output_type != "latent"
        if decode_enabled:
            sample_latents = latents[
                :, :, :self.num_frames_per_block, :, :]
            self._initialize_vae_graphs(
                sample_latents,
                fastvideo_args,
                pool,
            )
        decoded_chunks: list[torch.Tensor] = []
        runtime_vae_cache: list[torch.Tensor | None] | None = None
        context_noise = getattr(
            fastvideo_args.pipeline_config,
            "context_noise",
            0,
        )

        def run_dit_chunk(
            chunk_idx: int,
            stream: torch.cuda.Stream,
            wait_events: tuple[torch.cuda.Event, ...],
            progress_bar,
        ) -> tuple[torch.Tensor, torch.cuda.Event]:
            start_index = chunk_idx * self.num_frames_per_block
            current_num_frames = self.num_frames_per_block
            with torch.cuda.stream(stream):
                for event in wait_events:
                    stream.wait_event(event)
                torch.cuda.nvtx.range_push(
                    f"DynamicGC_DiT_chunk_{chunk_idx}")
                try:
                    current_latents = latents[
                        :,
                        :,
                        start_index:start_index + current_num_frames,
                        :,
                        :,
                    ]
                    noise_latents_btchw = current_latents.permute(
                        0, 2, 1, 3, 4)
                    raw_latent_shape = noise_latents_btchw.shape
                    attn_metadata = None

                    for step_idx, timestep in enumerate(timesteps):
                        noise_latents = noise_latents_btchw.clone()
                        latent_model_input = current_latents.to(target_dtype)
                        timestep_expanded = timestep.view(1, 1).expand(
                            latent_model_input.shape[0],
                            noise_latents.shape[1],
                        )
                        attn_metadata = self._build_attention_metadata(
                            step_idx,
                            current_num_frames,
                            height,
                            width,
                            patch_size,
                            fastvideo_args,
                        )

                        with torch.autocast(
                                device_type="cuda",
                                dtype=target_dtype,
                                enabled=autocast_enabled
                        ), set_forward_context(
                                current_timestep=step_idx,
                                attn_metadata=attn_metadata,
                                forward_batch=batch):
                            model_timestep = timestep * torch.ones(
                                (latent_model_input.shape[0], 1),
                                device=latent_model_input.device,
                                dtype=torch.long,
                            )
                            pred_noise_btchw = self.transformer(
                                latent_model_input,
                                prompt_embeds,
                                model_timestep,
                                kv_cache=kv_cache,
                                crossattn_cache=crossattn_cache,
                                current_start=(
                                    start_index * self.frame_seq_length),
                                start_frame=start_index,
                                **pos_cond_kwargs,
                            ).permute(0, 2, 1, 3, 4)

                        pred_video_btchw = pred_noise_to_pred_video(
                            pred_noise=pred_noise_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=timestep_expanded,
                            scheduler=self.scheduler,
                        ).unflatten(0, pred_noise_btchw.shape[:2])

                        if step_idx < len(timesteps) - 1:
                            next_timestep = (
                                timesteps[step_idx + 1] * torch.ones(
                                    [1],
                                    dtype=torch.long,
                                    device=pred_video_btchw.device,
                                ))
                            generator = (
                                batch.generator[0]
                                if isinstance(batch.generator, list)
                                else batch.generator)
                            noise_btchw = torch.randn(
                                raw_latent_shape,
                                dtype=pred_video_btchw.dtype,
                                generator=generator,
                            ).to(self.device)
                            noise_latents_btchw = self.scheduler.add_noise(
                                pred_video_btchw.flatten(0, 1),
                                noise_btchw.flatten(0, 1),
                                next_timestep,
                            ).unflatten(0, pred_video_btchw.shape[:2])
                            current_latents = noise_latents_btchw.permute(
                                0, 2, 1, 3, 4)
                        else:
                            current_latents = pred_video_btchw.permute(
                                0, 2, 1, 3, 4)

                        if progress_bar is not None:
                            progress_bar.update()

                    latents[
                        :,
                        :,
                        start_index:start_index + current_num_frames,
                        :,
                        :,
                    ] = current_latents

                    context_timestep = torch.ones(
                        [latents.shape[0]],
                        device=latents.device,
                        dtype=torch.long,
                    ) * int(context_noise)
                    with torch.autocast(
                            device_type="cuda",
                            dtype=target_dtype,
                            enabled=autocast_enabled
                    ), set_forward_context(
                            current_timestep=0,
                            attn_metadata=attn_metadata,
                            forward_batch=batch):
                        self.transformer(
                            current_latents.to(target_dtype),
                            prompt_embeds,
                            context_timestep.unsqueeze(1),
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            current_start=(
                                start_index * self.frame_seq_length),
                            start_frame=start_index,
                            **pos_cond_kwargs,
                        )
                finally:
                    torch.cuda.nvtx.range_pop()
                done = torch.cuda.Event()
                done.record(stream)
            return current_latents, done

        def run_vae_chunk(
            chunk_idx: int,
            chunk_latents: torch.Tensor,
            stream: torch.cuda.Stream,
            wait_events: tuple[torch.cuda.Event, ...],
        ) -> torch.cuda.Event:
            nonlocal runtime_vae_cache
            if self._vae_graphs is None:
                raise RuntimeError("VAE CUDA Graphs are not initialized.")
            graph_index_by_chunk = getattr(
                self, "_vae_graph_index_by_chunk", None)
            graph_idx = (
                chunk_idx if graph_index_by_chunk is None else
                graph_index_by_chunk.get(chunk_idx))
            with torch.cuda.stream(stream):
                for event in wait_events:
                    stream.wait_event(event)
                if graph_idx is None:
                    torch.cuda.nvtx.range_push(
                        f"DynamicGC_VAE_eager_chunk_{chunk_idx}")
                    try:
                        decoded, runtime_vae_cache = (
                            self._graph_safe_streaming_decode(
                                chunk_latents,
                                fastvideo_args,
                                cache=runtime_vae_cache,
                                is_first_chunk=(chunk_idx == 0),
                            ))
                        decoded_chunks.append(decoded)
                    finally:
                        torch.cuda.nvtx.range_pop()
                    done = torch.cuda.Event()
                    done.record(stream)
                    return done
                get_current_simple_profiler().enter(
                    f"vae_graph_chunk_{chunk_idx}")
                torch.cuda.nvtx.range_push(
                    f"DynamicGC_VAE_graph_replay_chunk_{chunk_idx}")
                try:
                    torch.cuda.nvtx.range_push(
                        f"DynamicGC_VAE_static_input_copy_chunk_{chunk_idx}")
                    self._vae_graph_inputs[graph_idx].copy_(
                        chunk_latents,
                        non_blocking=True,
                    )
                    torch.cuda.nvtx.range_pop()
                    graph_cache_inputs = getattr(
                        self, "_vae_graph_cache_inputs", None)
                    graph_cache_outputs = getattr(
                        self, "_vae_graph_cache_outputs", None)
                    if (graph_cache_inputs is not None
                            and graph_cache_inputs[graph_idx]):
                        if runtime_vae_cache is None:
                            raise RuntimeError(
                                "Warm VAE graph requires a runtime cache.")
                        cache_inputs = graph_cache_inputs[graph_idx]
                        cache_outputs = runtime_vae_cache
                        for cache_input, cache_output in zip(
                                cache_inputs, cache_outputs, strict=True):
                            if not isinstance(cache_input, torch.Tensor):
                                continue
                            if not isinstance(cache_output, torch.Tensor):
                                raise TypeError(
                                    "Active graph cache output is not a tensor.")
                            cache_input.copy_(
                                cache_output, non_blocking=True)
                    self._vae_graphs[graph_idx].replay()
                    if graph_cache_outputs is not None:
                        runtime_vae_cache = graph_cache_outputs[graph_idx]
                    decoded_buffers = getattr(
                        self, "_vae_decoded_output_buffers", None)
                    if decoded_buffers is None:
                        decoded_chunks.append(
                            self._vae_graph_outputs[graph_idx])
                    else:
                        decoded_buffers[chunk_idx].copy_(
                            self._vae_graph_outputs[graph_idx],
                            non_blocking=True,
                        )
                        decoded_chunks.append(decoded_buffers[chunk_idx])
                finally:
                    torch.cuda.nvtx.range_pop()
                    get_current_simple_profiler().exit()
                done = torch.cuda.Event()
                done.record(stream)
            return done

        inputs_ready = torch.cuda.Event()
        inputs_ready.record(torch.cuda.current_stream())
        previous_vae_done: torch.cuda.Event | None = None

        with self.progress_bar(
                total=num_chunks * len(timesteps)) as progress_bar:
            get_current_simple_profiler().enter("chunk_0")
            previous_latents, previous_dit_done = run_dit_chunk(
                0,
                pool.full_dit_stream,
                (inputs_ready,),
                progress_bar,
            )
            get_current_simple_profiler().exit()

            for dit_chunk_idx in range(1, num_chunks):
                pair = pool[self.DIT_SMS_BY_CHUNK[dit_chunk_idx]]
                wait_events = (
                    (previous_dit_done,)
                    if previous_vae_done is None else
                    (previous_dit_done, previous_vae_done))

                if decode_enabled:
                    previous_vae_done = run_vae_chunk(
                        dit_chunk_idx - 1,
                        previous_latents,
                        pair.vae_stream,
                        wait_events,
                    )

                get_current_simple_profiler().enter(
                    f"chunk_{dit_chunk_idx}")
                current_latents, current_dit_done = run_dit_chunk(
                    dit_chunk_idx,
                    pair.dit_stream,
                    wait_events,
                    progress_bar,
                )
                get_current_simple_profiler().exit()
                previous_latents = current_latents
                previous_dit_done = current_dit_done

            if decode_enabled:
                final_wait_events = (
                    (previous_dit_done,)
                    if previous_vae_done is None else
                    (previous_dit_done, previous_vae_done))
                final_vae_done = run_vae_chunk(
                    num_chunks - 1,
                    previous_latents,
                    pool.full_vae_stream,
                    final_wait_events,
                )
                final_vae_done.synchronize()
            else:
                previous_dit_done.synchronize()

        batch.latents = latents
        if decode_enabled:
            if len(decoded_chunks) != num_chunks:
                raise RuntimeError(
                    f"Expected {num_chunks} VAE chunks, produced "
                    f"{len(decoded_chunks)}.")
            batch.output = torch.cat(decoded_chunks, dim=2).to(torch.float32)
        else:
            batch.output = latents.to(torch.float32)

        batch.extra["pingpong_num_decoded_chunks"] = len(decoded_chunks)
        batch.extra["pingpong_execution_order"] = (
            "DiT_0_full__VAE_previous_parallel_DiT_current__VAE_6_full")
        batch.extra["green_context_dit_sms_by_chunk"] = dict(
            self.DIT_SMS_BY_CHUNK)
        batch.extra["vae_cuda_graph_memory_mib"] = dict(
            self._vae_graph_memory_mib)
        if fastvideo_args.log_kv_cache_size:
            batch.extra["kv_cache_mib"] = bytes_to_mib(
                tensor_bytes(kv_cache))
            batch.extra["crossattn_mib"] = bytes_to_mib(
                tensor_bytes(crossattn_cache))
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch


