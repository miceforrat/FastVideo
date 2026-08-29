# SPDX-License-Identifier: Apache-2.0
"""Wan causal DMD pipeline with dynamic GC and graph-safe VAE replay."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    ConditioningStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
)
from fastvideo.pipelines.stages.dynamic_green_context_cuda_graph_v2_stage import (
    DynamicGreenContextCUDAGraphV2DenoisingDecodingStage,
)


class WanCausalDMDDynamicGreenContextCUDAGraphV2Pipeline(
        LoRAPipeline, ComposedPipelineBase):
    """Run dynamic SM splits with graph-safe VAE replay."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage("input_validation_stage", InputValidationStage())
        self.add_stage(
            "prompt_encoding_stage",
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage("conditioning_stage", ConditioningStage())
        self.add_stage(
            "latent_preparation_stage",
            LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )
        self.add_stage(
            "dynamic_green_context_cuda_graph_v2_stage",
            DynamicGreenContextCUDAGraphV2DenoisingDecodingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            ),
        )


EntryClass = WanCausalDMDDynamicGreenContextCUDAGraphV2Pipeline
