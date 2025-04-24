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

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from trl import IterativeSFTTrainer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from ExIt_config import ExItConfig

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
            print(f"Iteration {counter}")
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
    def __init__(self, args:ExItConfig, train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None, apprentice: nn.Module, expert_generate_maker):
        # expert_generate_maker must be a higher-order function which takes a policy (an `apprentice` model) and returns a function which has the same interface as `generate`
        self.args = args
        self.apprentice = apprentice
        self.expert_generate_maker = expert_generate_maker

    def train(self):
        for it in range(self.args.num_expert_iteration_epochs):



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to('cuda')
    length_condition = lambda x: len(tokenizer.decode(x)) > 30
    matthew_condition = lambda x: "Matt" in tokenizer.decode(x)
    rse = RejectionSamplingExpert(model=model, condition=matthew_condition)
    input_ids = tokenizer("Some names which start with M are Mark,", return_tensors='pt').to('cuda')
    results = rse.generate_surplus(tokenizer=tokenizer, num_return_sequences=2, max_new_tokens=30, **input_ids)
    string_results = tokenizer.batch_decode(results)
    print(string_results)
