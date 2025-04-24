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

from typing import Optional
from collections.abc import Callable 

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import Dataset
from ExIt_config import ExItConfig
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from trl import SFTTrainer
from transformers.generation.configuration_utils import GenerationConfig
from datasets import load_dataset

def rejection_sampling_generate_maker(generation_filter, keep_surplus=False):
    def apprentice_to_generate(apprentice):
        def generate_surplus(generation_config:GenerationConfig):
            # May generate a few more completions than requested.
            if not generation_config.do_sample:
                raise ValueError("For rejection sampling generation, the generation_config must have do_sample=True")
            num_return_sequences = generation_config.num_return_sequences
            legal_generations = []
            pbar = tqdm(total=num_return_sequences, desc="Generating valid sequences")
            counter = 1
            while len(legal_generations) < num_return_sequences:
                generated_ids = apprentice.generate(generation_config)
                filtered_generations = generation_filter(generated_ids)
                legal_generations.extend(filtered_generations)
                pbar.update(len(filtered_generations))
                counter += 1
            pbar.close()

            # Pad sequences to the same length and stack them
            padded_sequences = pad_sequence(legal_generations, batch_first=True, padding_value=0)
            return padded_sequences

        def generate(generation_config:GenerationConfig):
            # Generates exactly the amount requested by discarding some completions.
            num_returned = generation_config.num_return_sequences
            return generate_surplus(generation_config)[:num_returned]
        
        return generate_surplus if keep_surplus else generate
    return apprentice_to_generate


class RejectionSamplingExpert:
    def __init__(self, model, condition):
        self.model = model
        self.condition = condition

    def generate_surplus(self, **kwargs):
        # May generate a few more completions than requested.
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        kwargs['do_sample'] = True
        legal_generations = []
        pbar = tqdm(total=num_return_sequences, desc="Generating valid sequences")
        counter = 1
        while len(legal_generations) < num_return_sequences:
            generated_ids = self.model.generate(**kwargs)
            mask = torch.tensor([self.condition(row) for row in generated_ids])
            filtered_generations = generated_ids[mask]
            legal_generations.extend(filtered_generations)
            pbar.update(len(filtered_generations))
            counter += 1
        pbar.close()

        # Pad sequences to the same length and stack them
        padded_sequences = pad_sequence(legal_generations, batch_first=True, padding_value=0)
        return padded_sequences

    def generate(self, **kwargs):
        # Generates exactly the amount requested by discarding some completions.
        return self.generate_surplus(**kwargs)[:kwargs.get('num_return_sequences', 1)]

# class HighestRewardExpert:
# Just use BestOfNSampler
#     def __init__(self, model, reward_function, samples):
#         self.model = model
#         self.reward_function = reward_function
#         self.samples = samples

#     def generate(self, **kwargs):

class ExItTrainer(Trainer):
    def __init__(
            self, 
            args:ExItConfig,
            train_dataset: Dataset,
            apprentice: nn.Module,
            expert_generate_maker:Callable[[nn.Module], Callable[[GenerationConfig], Tensor]],
            tokenizer,
            data_collator: Optional[DataCollatorWithPadding] = None
    ) -> None:
        # expert_generate_maker must be a higher-order function which takes a policy (an `apprentice` model) and returns a function which has the same interface as `generate`
        super().__init__(model=apprentice)
        self.args = args
        self.apprentice = apprentice
        self.expert_generate_maker = expert_generate_maker
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.expert_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True, # needed; otherwise the last batch will be of ragged shape
        )

    def train(self):
        for it in range(self.args.num_expert_iteration_epochs):
            expert_pairs = []
            for expert_batch in self.dataloader:
                expert_generate = self.expert_generate_maker(self.apprentice)
                expert_completions = expert_generate(self.args.expert_generation_config, **expert_batch)
                text_completions = self.tokenizer.batch_decode(
                    expert_completions, skip_special_tokens=True
                )
                
                # Create prompt+completion pairs
                for i, completion in enumerate(text_completions):
                    prompt = self.tokenizer.decode(expert_batch["input_ids"][i], skip_special_tokens=True)
                    expert_pairs.append({"text": prompt + completion})
            
            # Build dataset for this round
            ds_round = Dataset.from_list(expert_pairs)
            
            # Train with SFTTrainer
            training_args = TrainingArguments(
                output_dir=f"exit_round_{it}",
                per_device_train_batch_size=self.args.per_device_train_batch_size,
                learning_rate=self.args.learning_rate,
                num_train_epochs=1,
                logging_steps=20,
                save_strategy="no"
            )
            
            # The SFTTrainer from trl has different parameters
            sft_trainer = SFTTrainer(
                model=self.apprentice,
                tokenizer=self.tokenizer,
                train_dataset=ds_round,
                args=training_args,
                max_seq_length=512
            )
            
            sft_trainer.train()
            self.apprentice = sft_trainer.model
            
            # Save checkpoint
            self.apprentice.save_pretrained(f"./checkpoints/exit_iter_{it}")


if __name__ == '__main__':
    from ExIt_config import ExItConfig
    
    # Load model and tokenizer
    BASE_MODEL = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to('cuda')
    
    # Simple completion filter
    def length_condition(x):
        return len(tokenizer.decode(x)) > 30
    
    # Create rejection sampling generate function
    rs_generate_maker = rejection_sampling_generate_maker(
        generation_filter=lambda ids: [row for row in ids if length_condition(row)]
    )
    
    # Load a small dataset
    dataset = load_dataset("gsm8k", "main", split="train[:5]")
    
    # Prepare dataset and convert to the correct format
    def format_example(example):
        return {"text": example["question"] + "\n\n###\n"}
    
    formatted_dataset = dataset.map(format_example)
    train_dataset = Dataset.from_dict({
        "text": formatted_dataset["text"],
        "input_ids": tokenizer(formatted_dataset["text"], return_tensors="pt", padding=True)["input_ids"]
    })
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create config
    config = ExItConfig(
        num_expert_iteration_epochs=2,
        expert_batch_size=2,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        expert_generation_config=GenerationConfig(
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            max_new_tokens=256,
            num_return_sequences=4
        )
    )
    
    # Create trainer
    trainer = ExItTrainer(
        args=config,
        train_dataset=train_dataset,
        apprentice=model,
        expert_generate_maker=rs_generate_maker,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train
    trainer.train()
