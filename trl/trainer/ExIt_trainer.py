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
    """
    Creates a function that wraps a model's generate method with rejection sampling.
    
    Args:
        generation_filter: Function to filter generated sequences
        keep_surplus: Whether to return more sequences than requested
        
    Returns:
        A function that takes a model and returns a modified generate function
    """
    def apprentice_to_generate(apprentice):
        """
        Takes a model and returns a generate function with rejection sampling.
        
        Args:
            apprentice: The model to generate from
            
        Returns:
            A modified generate function
        """
        # Get the original generate function
        original_generate = apprentice.generate
        
        def generate(**kwargs):
            """
            Generate completions with rejection sampling.
            
            Args:
                **kwargs: Arguments passed to the model's generate method
                
            Returns:
                Tensor of filtered generated sequences
            """
            # Make sure we have a GenerationConfig object
            generation_config = kwargs.pop('generation_config', None)
            if generation_config is None:
                if hasattr(apprentice, "generation_config"):
                    generation_config = apprentice.generation_config
                else:
                    generation_config = GenerationConfig()
            
            # Ensure do_sample is set for rejection sampling
            if not generation_config.do_sample:
                generation_config = GenerationConfig(**{**generation_config.to_dict(), "do_sample": True})
                
            # Set num_return_sequences if not already set
            num_return_sequences = generation_config.num_return_sequences or 1
            
            # Update kwargs with the generation_config
            kwargs['generation_config'] = generation_config
            
            # Collect valid generations
            legal_generations = []
            pbar = tqdm(total=num_return_sequences, desc="Generating valid sequences")
            
            while len(legal_generations) < num_return_sequences:
                # Generate sequences using HuggingFace's standard interface with unpacked kwargs
                generated_ids = original_generate(**kwargs)
                
                # Apply the filter to get valid generations
                filtered_generations = generation_filter(generated_ids)
                
                # Add valid generations to our collection
                if isinstance(filtered_generations, torch.Tensor):
                    if filtered_generations.dim() == 2:  # Already properly shaped
                        legal_generations.append(filtered_generations)
                    else:  # Single tensor containing multiple sequences
                        legal_generations.append(filtered_generations)
                else:  # List of tensors
                    legal_generations.extend(filtered_generations)
                
                # Update progress
                added = len(filtered_generations) if isinstance(filtered_generations, list) else filtered_generations.size(0)
                pbar.update(added)
                
                # Check if we have enough generations
                if len(legal_generations) >= num_return_sequences:
                    break
                    
            pbar.close()
            
            # Concatenate all generations if we have multiple tensors
            if isinstance(legal_generations, list):
                if isinstance(legal_generations[0], torch.Tensor):
                    # Get pad token from model config or default to 0
                    pad_token_id = getattr(apprentice.config, "pad_token_id", 0) if hasattr(apprentice, "config") else 0
                    # Stack all tensors with padding
                    legal_generations = pad_sequence(legal_generations, batch_first=True, padding_value=pad_token_id)
                elif not legal_generations:  # Empty list
                    raise ValueError("No valid generations found that match the filter criteria")
            
            # Return exactly the requested number of sequences
            if not keep_surplus and isinstance(legal_generations, torch.Tensor):
                return legal_generations[:num_return_sequences]
            return legal_generations
        
        return generate
    
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
            expert_generate_maker:Callable,  # Simplified type annotation
            processing_class,
            data_collator: Optional[DataCollatorWithPadding] = None
    ) -> None:
        # expert_generate_maker must be a higher-order function which takes a policy (an `apprentice` model) and returns a function which has the same interface as `generate`
        # Don't call super().__init__() as we're setting up things manually
        self.args = args
        self.model = apprentice  # Set model for compatibility with Trainer methods
        self.apprentice = apprentice
        self.expert_generate_maker = expert_generate_maker
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.processing_class = processing_class
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.expert_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True, # needed; otherwise the last batch will be of ragged shape
        )

    def train(self):
        args = self.args
        model = self.apprentice
        
        for it in range(args.num_expert_iteration_epochs):
            print(f"===== Expert Iteration {it+1}/{args.num_expert_iteration_epochs} =====")
            
            # Process each batch with the expert
            for batch_idx, expert_batch in enumerate(self.dataloader):
                # Get expert generate function for the current model
                expert_generate = self.expert_generate_maker(model)
                
                # Pass parameters directly using kwargs unpacking for cleaner interface
                expert_completions = expert_generate(
                    input_ids=expert_batch["input_ids"],
                    attention_mask=expert_batch.get("attention_mask", None),
                    generation_config=args.expert_generation_config
                )
                
                # Decode completions to text
                text_completions = self.processing_class.batch_decode(
                    expert_completions, skip_special_tokens=True
                )
                
                # Create prompt+completion pairs for this batch
                expert_pairs = []
                for i, completion in enumerate(text_completions):
                    prompt = self.processing_class.decode(expert_batch["input_ids"][i], skip_special_tokens=True)
                    expert_pairs.append({"text": prompt + completion})
                
                # Create a small dataset just for this batch
                ds_batch = Dataset.from_list(expert_pairs)
                
                # Set up training arguments
                training_args = TrainingArguments(
                    output_dir=f"exit_round_{it}_batch_{batch_idx}",
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    learning_rate=args.learning_rate,
                    num_train_epochs=1,
                    logging_steps=1,
                    save_strategy="no"
                )
                
                # Use SFTTrainer from trl to update the model on this batch
                # Import SFTTrainer locally to avoid circular imports
                from trl import SFTTrainer
                
                # Create SFTTrainer with processing_class
                sft_trainer = SFTTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=ds_batch,
                    processing_class=self.processing_class
                )
                
                # Train on this batch
                sft_trainer.train()
                
                # Update our model reference
                model = sft_trainer.model
                
                print(f"Completed batch {batch_idx+1}/{len(self.dataloader)} for iteration {it+1}")
            
            # Save checkpoint after each iteration
            model_path = f"./checkpoints/exit_iter_{it}"
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(model_path)
                print(f"Saved model checkpoint to {model_path}")
            else:
                print(f"Model does not support save_pretrained, skipping checkpoint at {model_path}")
            
            # Update the apprentice reference
            self.apprentice = model
            
        # Return the final model
        return model


if __name__ == '__main__':
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.generation.configuration_utils import GenerationConfig
    from datasets import load_dataset, Dataset
    from ExIt_config import ExItConfig
    import torch
    
    # Load model and tokenizer
    BASE_MODEL = "Qwen/Qwen2.5-0.5B"
    processing_class = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to('cuda')
    
    # Simple completion filter function that checks sequence length
    def length_condition(x):
        return len(processing_class.decode(x)) > 30
    
    # Create rejection sampling generate function with padding
    # Returns a tensor of properly padded sequences
    rs_generate_maker = rejection_sampling_generate_maker(
        generation_filter=lambda ids: torch.nn.utils.rnn.pad_sequence(
            [row for row in ids if length_condition(row)], 
            batch_first=True,
            padding_value=processing_class.pad_token_id
        )
    )
    
    # Test the generate function directly
    generate_fn = rs_generate_maker(model)
    test_input = processing_class("Test prompt:", return_tensors="pt").to('cuda')
    
    # Create a generation config
    gen_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        max_new_tokens=20,
        num_return_sequences=2
    )
    
    # Try generating with the function
    try:
        completions = generate_fn(
            input_ids=test_input.input_ids,
            attention_mask=test_input.attention_mask,
            generation_config=gen_config
        )
        if isinstance(completions, torch.Tensor):
            print(f"Successfully generated {completions.size(0)} completions")
        else:
            print(f"Successfully generated {len(completions)} completions")
        print(processing_class.batch_decode(completions, skip_special_tokens=True))
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # Load a small dataset
    dataset = load_dataset("gsm8k", "main", split="train[:5]")
    
    # Prepare dataset and convert to the correct format
    def format_example(example):
        return {"text": example["question"] + "\n\n###\n"}
    
    # Process the dataset
    formatted_dataset = dataset.map(format_example)
    
    # Create a more compatible dataset format
    processed_texts = formatted_dataset["text"] if isinstance(formatted_dataset, Dataset) else [ex["text"] for ex in formatted_dataset]
    inputs = processing_class(processed_texts, return_tensors="pt", padding=True)
    
    train_dataset = Dataset.from_dict({
        "text": processed_texts,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]  # Include attention_mask
    })
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=processing_class)
    
    # Create config with generation parameters
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
    
    # Create trainer with explicit typing for expert_generate_maker
    trainer = ExItTrainer(
        args=config,
        train_dataset=train_dataset,
        apprentice=model,
        expert_generate_maker=rs_generate_maker,
        processing_class=processing_class,
        data_collator=data_collator
    )
    
    # Train the model
    print("Starting ExpertIteration training...")
    trained_model = trainer.train()
