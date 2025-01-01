# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import unittest

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import ModelConfig
from trl.trainer import compute_accuracy
from trl.trainer.utils import (
    DataCollatorForChatML,
    batch_generation,
    decode_and_strip_padding,
    generate_model_card,
    get_peft_config,
    pad,
    EntropyRegularizerConfig,
    EntropyRegularizer,
)


if is_peft_available():
    from peft import LoraConfig


class TestPad(unittest.TestCase):
    def test_pad_1_dim_left(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_1_dim_right(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_left(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[0, 0], [5, 6]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right_multidim(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 0], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))


@require_peft
class TestGetPEFTConfig(unittest.TestCase):
    def test_create_peft_config_use_peft_false(self):
        """Test that when use_peft is False, the function returns None."""
        model_args = ModelConfig(use_peft=False)
        peft_config = get_peft_config(model_args)
        self.assertIsNone(peft_config)

    def test_create_peft_config_use_peft_true(self):
        """Test that when use_peft is True, the function returns a LoraConfig object."""
        # Provide non-default values to the model config for testing
        peft_kwargs = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_task_type": "SEQ_CLS",
            "use_rslora": True,
            "lora_target_modules": ["up_proj", "down_proj"],
            "lora_modules_to_save": ["up_proj"],
        }
        model_args = ModelConfig(use_peft=True, **peft_kwargs)
        peft_config = get_peft_config(model_args)
        self.assertTrue(isinstance(peft_config, LoraConfig))
        for arg, value in peft_kwargs.items():
            # Test that lists of modules are converted to sets
            if arg == "lora_target_modules":
                value = set(value)
            # Rename the argument to match the LoraConfig attribute name
            if arg in ["lora_r", "lora_task_type", "lora_target_modules", "lora_modules_to_save"]:
                arg = arg[len("lora_") :] if arg.startswith("lora_") else arg

            self.assertEqual(getattr(peft_config, arg), value)


class TestDecodeAndStripPadding(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

    def test_example_with_padding(self):
        inputs = self.tokenizer(["Hello world", "Hello"], padding=True, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello world", "Hello"])

    def test_example_without_padding(self):
        inputs = self.tokenizer(["Hello", "Hello"], padding=False, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello", "Hello"])


class TestGenerateModelCard(unittest.TestCase):
    def test_full(self):
        model_card = generate_model_card(
            base_model="username/my_base_model",
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name="username/my_dataset",
            tags=["trl", "trainer-tag"],
            wandb_url="https://wandb.ai/username/project_id/runs/abcd1234",
            comet_url="https://www.comet.com/username/project_id/experiment_id",
            trainer_name="My Trainer",
            trainer_citation="@article{my_trainer, ...}",
            paper_title="My Paper",
            paper_id="1234.56789",
        )
        card_text = str(model_card)
        self.assertIn("[username/my_base_model](https://huggingface.co/username/my_base_model)", card_text)
        self.assertIn("my_model", card_text)
        self.assertIn('pipeline("text-generation", model="username/my_hub_model", device="cuda")', card_text)
        self.assertIn("datasets: username/my_dataset", card_text)
        self.assertIn("](https://wandb.ai/username/project_id/runs/abcd1234)", card_text)
        self.assertIn("](https://www.comet.com/username/project_id/experiment_id", card_text)
        self.assertIn("My Trainer", card_text)
        self.assertIn("```bibtex\n@article{my_trainer, ...}\n```", card_text)
        self.assertIn("[My Paper](https://huggingface.co/papers/1234.56789)", card_text)

    def test_val_none(self):
        model_card = generate_model_card(
            base_model=None,
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name=None,
            tags=[],
            wandb_url=None,
            comet_url=None,
            trainer_name="My Trainer",
            trainer_citation=None,
            paper_title=None,
            paper_id=None,
        )
        card_text = str(model_card)
        self.assertIn("my_model", card_text)
        self.assertIn('pipeline("text-generation", model="username/my_hub_model", device="cuda")', card_text)
        self.assertIn("My Trainer", card_text)


class TestDataCollatorForChatML(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define token IDs
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        # Token ID for "true", the last assistant's response in the example:
        self.ignore_index = -100
        self.max_length = 1024
        self.messages_key = "messages"

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset.to_list()

        # Initialize the data collator
        self.collator = DataCollatorForChatML(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
        )

    def test_data_collator_for_chatml(self):
        # Process the data
        data = self.collator(self.examples)

        # Verify basic shapes and types
        self.assertIn("input_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("labels", data)
        self.assertIn("prompts", data)
        self.assertIn("prompt_attention_mask", data)

        # Decode input_ids and labels for verification
        input_ids = data["input_ids"][0].tolist()
        labels = data["labels"][0].tolist()
        prompt_only = data["prompts"][0].tolist()

        # Get the last assistant's response for comparison
        last_message = self.examples[0][self.messages_key][-1]
        self.assertEqual(last_message["role"], "assistant", "Last message should be from assistant")
        last_assistant_response = last_message["content"]

        # Verify that input_ids contain both prompt and response
        decoded_input = self.tokenizer.decode(input_ids)
        self.assertIn(last_assistant_response, decoded_input, "Input should contain assistant's response")

        # Verify that prompts only contain the conversation up to the last response
        decoded_prompt = self.tokenizer.decode(prompt_only)
        self.assertNotIn(last_assistant_response, decoded_prompt, "Prompt should not contain assistant's response")

        # Verify labels are -100 for non-assistant parts
        prompt_length = len(prompt_only)
        self.assertTrue(
            all(label == self.ignore_index for label in labels[:prompt_length]),
            "Labels should be ignore_index for prompt tokens",
        )

        # Verify labels match assistant response after prompt
        # Add a filter to remove any trailing tokens after the first <|im_end|>
        last_assistant_response_with_end = last_assistant_response + self.tokenizer.eos_token
        last_assistant_response_tokens = self.tokenizer.encode(
            last_assistant_response_with_end, add_special_tokens=False
        )

        response_labels = []
        for label in labels[prompt_length:]:
            if label == self.ignore_index:
                continue
            response_labels.append(label)
            if label == self.tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break
        self.assertEqual(
            response_labels,
            last_assistant_response_tokens,
            "Labels should match assistant response tokens",
        )

        # Verify there isn't a generation prompt at the end
        generation_prompt = "<|im_start|>assistant"
        self.assertFalse(
            decoded_input.strip().endswith(generation_prompt),
            f"Input should not end with generation prompt '{generation_prompt}'",
        )

        self.assertEqual(
            response_labels,
            last_assistant_response_tokens,
            "Labels should match assistant response tokens",
        )


class TestBatchGeneration(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.5,
            do_sample=True,
            top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset["messages"]
        self.mini_batch_size = 3

    def test_mini_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"]
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, self.mini_batch_size, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        self.assertGreater(max_length_query, context_length)
        self.assertEqual(query_responses.shape, (bs, max_length_query))
        self.assertEqual(logits.shape, (bs, max_length_logits, self.model.config.vocab_size))

    def test_single_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"]
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, bs, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        self.assertGreater(max_length_query, context_length)
        self.assertEqual(query_responses.shape, (bs, max_length_query))
        self.assertEqual(logits.shape, (bs, max_length_logits, self.model.config.vocab_size))


class TestComputeAccuracy(unittest.TestCase):
    def test_token_classification_task(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[0, 1], [1, 0]]),
        )
        expected_accuracy = 0.5  # 2 matches, 2 mismatches
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_token_classification_task_with_ignored_tokens_0(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[1, 0], [1, -100]]),
        )
        expected_accuracy = 1.0  # All non-ignored tokens match
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_token_classification_task_with_ignored_tokens_1(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[1, 1], [0, -100]]),
        )
        expected_accuracy = 1 / 3  # 1 match, 2 mismatch, 1 ignored
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_rewards_comparison_task(self):
        eval_pred = (
            np.array(
                [
                    [0.9, 0.1],  # Batch 1
                    [0.6, 0.4],  # Batch 2
                    [0.5, 0.5],  # Batch 3 (equal)
                ]
            ),
            np.array([0, 1, 1]),
        )
        expected_accuracy = 0.5  # 1 match, 1 mismatch, 1 equal (ignored)

        with self.assertWarns(UserWarning) as cm:
            result = compute_accuracy(eval_pred)

        self.assertAlmostEqual(result["accuracy"], expected_accuracy)
        expected_warning = (
            "There are 1 out of 3 instances where the predictions for both options are equal. "
            "These instances are ignored in the accuracy computation."
        )
        self.assertEqual(str(cm.warning), expected_warning)


class TestEntropyRegularizer(unittest.TestCase):
    def setUp(self):
        self.target_entropy = 0.5
        self.config = EntropyRegularizerConfig(
            target_entropy=self.target_entropy,
            start_temperature=0.7,
            temperature_optimizer_lr=0.01,
            temperature_optimizer_momentum=0.9,
            temperature_scheduler_gamma=0.99
        )
        
    def test_initialization(self):
        # Test initialization without target entropy
        config_no_target = EntropyRegularizerConfig(
            target_entropy=None,
            start_temperature=self.config.start_temperature
        )
        regularizer = EntropyRegularizer(config_no_target)
        self.assertAlmostEqual(regularizer.alpha.item(), self.config.start_temperature)
        self.assertIsNone(regularizer.target_entropy)
        self.assertIsNone(regularizer.optimizer)
        self.assertIsNone(regularizer.lr_scheduler)

        # Test initialization with target entropy
        regularizer = EntropyRegularizer(self.config)
        self.assertAlmostEqual(regularizer.alpha.item(), self.config.start_temperature)
        self.assertAlmostEqual(regularizer.target_entropy, self.target_entropy)
        self.assertIsNotNone(regularizer.optimizer)
        self.assertIsNotNone(regularizer.lr_scheduler)

    def test_temperature(self):
        regularizer = EntropyRegularizer(self.config)
        self.assertAlmostEqual(regularizer.temperature().item(), self.config.start_temperature)

    def test_regularization(self):
        regularizer = EntropyRegularizer(self.config)
        # Test with uniform distribution
        probs = torch.tensor([[0.5, 0.5]])
        expected_reg = -self.config.start_temperature * (-0.693147)  # -0.693147 is ln(2)
        reg = regularizer.regularization(probs)
        self.assertAlmostEqual(reg.item(), expected_reg, places=5)

    def test_step(self):
        regularizer = EntropyRegularizer(self.config)
        
        # Initial temperature
        initial_temp = regularizer.temperature().item()
        
        # Perform a step with uniform distribution
        probs = torch.tensor([[0.5, 0.5]], requires_grad=True)
        J_alpha_0 = regularizer.step(probs)
        J_alpha_1 = regularizer.step(probs)
        
        # Temperature should change after step
        self.assertNotEqual(regularizer.temperature().item(), initial_temp)
        self.assertLess(J_alpha_1, J_alpha_0)

class TestEntropyRegularizerConfig(unittest.TestCase):
    def test_default_values(self):
        config = EntropyRegularizerConfig()
        self.assertIsNone(config.target_entropy)
        self.assertEqual(config.start_temperature, 0.7)
        self.assertEqual(config.temperature_optimizer_lr, 1e-4)
        self.assertEqual(config.temperature_optimizer_momentum, 0.9)
        self.assertEqual(config.temperature_scheduler_gamma, 0.99)

    def test_custom_values(self):
        custom_config = EntropyRegularizerConfig(
            target_entropy=0.5,
            start_temperature=1.0,
            temperature_optimizer_lr=0.001,
            temperature_optimizer_momentum=0.8,
            temperature_scheduler_gamma=0.95
        )
        self.assertEqual(custom_config.target_entropy, 0.5)
        self.assertEqual(custom_config.start_temperature, 1.0)
        self.assertEqual(custom_config.temperature_optimizer_lr, 0.001)
        self.assertEqual(custom_config.temperature_optimizer_momentum, 0.8)
        self.assertEqual(custom_config.temperature_scheduler_gamma, 0.95)
