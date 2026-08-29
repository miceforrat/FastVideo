# SPDX-License-Identifier: Apache-2.0
"""Chunkwise causal DMD denoising and streaming VAE decoding."""

import torch
import torch.distributed as dist

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.causal_denoising import CausalDMDDenosingStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.profiling.small_node_profiler import get_current_simple_profiler
from fastvideo.utils import bytes_to_mib, tensor_bytes

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend,
    )
    vsa_available = True
except ImportError:
    VideoSparseAttentionBackend = None  # type: ignore
    vsa_available = False


class PingPongDenoisingDecodingStage(CausalDMDDenosingStage):
    """Connect real causal DiT chunks to the streaming Wan VAE.

    This first implementation uses the ordinary CUDA execution context and
    does not partition SMs. Its execution order is
    ``DiT_0 -> VAE_0 -> DiT_1 -> VAE_1 -> ...``. Keeping both operations in
    one stage establishes the boundary needed to overlap ``VAE_i`` with
    ``DiT_{i+1}`` in the next development step.
    """

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
            transformer_2=transformer_2,
            vae=vae,
        )
        self.decoding_stage = DecodingStage(vae=vae, pipeline=pipeline)

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("output", batch.output,
                         [V.is_tensor, V.with_dims(5)])
        return result

    def _validate_supported_request(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        boundary_ratio = fastvideo_args.pipeline_config.dit_config.boundary_ratio
        if self.transformer_2 is not None or boundary_ratio is not None:
            raise NotImplementedError(
                "Ping-pong currently supports the single-transformer "
                "SFWan2.1 causal DMD pipeline only.")
        if batch.pil_image is not None:
            raise NotImplementedError("Ping-pong currently supports T2V only.")
        if batch.return_trajectory_decoded:
            raise NotImplementedError(
                "Trajectory decoding is not supported by ping-pong.")
        if (fastvideo_args.dp_decoding and dist.is_available()
                and dist.is_initialized() and dist.get_world_size() > 1):
            raise NotImplementedError(
                "WanVAE streaming decode does not support multi-rank "
                "dp_decoding yet.")
        if not fastvideo_args.model_loaded.get("vae", True):
            raise RuntimeError("VAE must be loaded before ping-pong starts.")
        if not hasattr(self.vae, "get_streaming_cache") or not hasattr(
                self.vae, "streaming_decode"):
            raise TypeError("The configured VAE has no streaming decode API.")

    def _build_attention_metadata(
        self,
        step_idx: int,
        current_num_frames: int,
        height: int,
        width: int,
        patch_size,
        fastvideo_args: FastVideoArgs,
    ):
        if not (vsa_available
                and self.attn_backend == VideoSparseAttentionBackend):
            return None
        builder_cls = self.attn_backend.get_builder_cls()
        if builder_cls is None:
            return None
        self.attn_metadata_builder_cls = builder_cls
        self.attn_metadata_builder = builder_cls()
        metadata = self.attn_metadata_builder.build(
            current_timestep=step_idx,
            raw_latent_shape=(current_num_frames, height, width),
            patch_size=patch_size,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
            device=get_local_torch_device(),
        )
        if metadata is None:
            raise RuntimeError("VSA metadata builder returned None.")
        return metadata

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
        decode_enabled = fastvideo_args.output_type != "latent"
        decoded_chunks: list[torch.Tensor] = []
        vae_cache: list[torch.Tensor | None] | None = None
        start_index = 0

        with self.progress_bar(
                total=num_chunks * len(timesteps)) as progress_bar:
            for chunk_idx in range(num_chunks):
                get_current_simple_profiler().enter(f"chunk_{chunk_idx}")
                current_num_frames = self.num_frames_per_block
                current_latents = latents[:, :, start_index:start_index +
                                           current_num_frames, :, :]
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                raw_latent_shape = noise_latents_btchw.shape
                attn_metadata = None

                for step_idx, timestep in enumerate(timesteps):
                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = current_latents.to(target_dtype)
                    timestep_expanded = timestep.view(1, 1).expand(
                        latent_model_input.shape[0], noise_latents.shape[1])
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
                            enabled=autocast_enabled), set_forward_context(
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
                            current_start=start_index * self.frame_seq_length,
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
                        next_timestep = timesteps[step_idx + 1] * torch.ones(
                            [1],
                            dtype=torch.long,
                            device=pred_video_btchw.device,
                        )
                        generator = (batch.generator[0] if isinstance(
                            batch.generator, list) else batch.generator)
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

                latents[:, :, start_index:start_index + current_num_frames,
                        :, :] = current_latents

                context_noise = getattr(
                    fastvideo_args.pipeline_config,
                    "context_noise",
                    0,
                )
                context_timestep = torch.ones(
                    [latents.shape[0]],
                    device=latents.device,
                    dtype=torch.long,
                ) * int(context_noise)
                with torch.autocast(
                        device_type="cuda",
                        dtype=target_dtype,
                        enabled=autocast_enabled), set_forward_context(
                            current_timestep=0,
                            attn_metadata=attn_metadata,
                            forward_batch=batch):
                    self.transformer(
                        current_latents.to(target_dtype),
                        prompt_embeds,
                        context_timestep.unsqueeze(1),
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=start_index * self.frame_seq_length,
                        start_frame=start_index,
                        **pos_cond_kwargs,
                    )

                if decode_enabled:
                    get_current_simple_profiler().enter(
                        f"vae_chunk_{chunk_idx}")
                    torch.cuda.nvtx.range_push(
                        f"PingPong_VAE_chunk_{chunk_idx}")
                    try:
                        decoded_chunk, vae_cache = (
                            self.decoding_stage.streaming_decode(
                                current_latents,
                                fastvideo_args,
                                cache=vae_cache,
                                is_first_chunk=(chunk_idx == 0),
                            ))
                        decoded_chunks.append(decoded_chunk)
                    finally:
                        torch.cuda.nvtx.range_pop()
                        get_current_simple_profiler().exit()

                start_index += current_num_frames
                get_current_simple_profiler().exit()

        batch.latents = latents
        if decode_enabled:
            if not decoded_chunks:
                raise RuntimeError("No VAE chunks were produced.")
            batch.output = torch.cat(decoded_chunks, dim=2).to(torch.float32)
        else:
            batch.output = latents.to(torch.float32)

        batch.extra["pingpong_num_decoded_chunks"] = len(decoded_chunks)
        batch.extra["pingpong_execution_order"] = "DiT_i_then_VAE_i_serial"
        if fastvideo_args.log_kv_cache_size:
            batch.extra["kv_cache_mib"] = bytes_to_mib(
                tensor_bytes(kv_cache))
            batch.extra["crossattn_mib"] = bytes_to_mib(
                tensor_bytes(crossattn_cache))
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch
