# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..trainer.utils import OnPolicyConfig
from transformers.training_args import TrainingArguments
from transformers.generation.configuration_utils import GenerationConfig

@dataclass
class ExItConfig(TrainingArguments):
    r"""
    Configuration class for the [`PPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        model_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        num_expert_iteration_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        expert_generation_config (`GenerationConfig`, *optional*, defaults to GenerationConfig(do_sample=True, top_p=0.95, temperature=0.7, max_new_tokens=256, num_return_sequences=8)):
            Determines how samples are taken from the expert, and how many are taken per prompt (through num_return_sequences).
        # ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
        #     This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
        #     improving generation speed. However, disabling this option allows training models that exceed the VRAM
        #     capacity of a single GPU, albeit at the cost of slower generation.
    """

    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    model_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    num_expert_iteration_epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs to train, sampling from the expert and training the apprentice."},
    )
    
    expert_generation_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            max_new_tokens=256,
            num_return_sequences=8
        ),
        metadata={"help": "Determines how samples are taken from the expert, and how many are taken per prompt (through num_return_sequences)."},
    )
    
    # TODO: Deal with this
    # ds3_gather_for_generation: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
    #         "generation, improving generation speed. However, disabling this option allows training models that "
    #         "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
    #     },
    # )
