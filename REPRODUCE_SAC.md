# Reproducing Matthew's Expert-Iteration / SAC Work

This document is scoped to **Matthew Khoriaty's (GitHub: `AMindToThink`) own additions** to this
fork of [huggingface/trl](https://github.com/huggingface/trl). It does **not** re-document
upstream trl (see the top-level `README.md` for that — it is the unmodified upstream readme).

Verified against branch `expert-iteration` at commit `f5d9233` (2025-04-25), which is checked out
here. Everything below was confirmed by reading the actual files in this repo; anything not
directly verifiable is marked `TODO (Matthew)`.

## 1. What Matthew added

On the `expert-iteration` branch, Matthew's commits (`ab61ddc` "Start Expert Iteration" through
`f5d9233` "continuing to finish up the ExItTrainer", 2025-04-24 to 2025-04-25) add exactly two new
files under `trl/trainer/`, on top of upstream trl at commit `294f35b`:

- **`trl/trainer/ExIt_config.py`** — `ExItConfig`, a `dataclass` subclassing
  `transformers.TrainingArguments`. Adds: `exp_name`, `model_adapter_name`,
  `num_expert_iteration_epochs` (default `4`), `expert_generation_config` (a `GenerationConfig`
  with `do_sample=True, top_p=0.95, temperature=0.7, max_new_tokens=256, num_return_sequences=8`),
  and `expert_batch_size` (default `50`).
- **`trl/trainer/ExIt_trainer.py`** — implements an **Expert Iteration** trainer:
  - `rejection_sampling_generate_maker(generation_filter, keep_surplus=False)`: a higher-order
    function that wraps a model's `.generate()` so it repeatedly samples until it has collected
    `num_return_sequences` completions that pass `generation_filter`.
  - `RejectionSamplingExpert`: a small class with the same idea (`generate_surplus` /
    `generate`), applying a boolean `condition` per generated row.
  - `ExItTrainer(Trainer)`: **does not call `Trainer.__init__`** — it wires up state manually.
    `train()` loops for `args.num_expert_iteration_epochs`, and for each batch from
    `train_dataset`: (1) builds an "expert" generate function from the *current* apprentice model
    via `expert_generate_maker`, (2) generates completions, (3) decodes prompt+completion pairs
    into a small on-the-fly `Dataset`, (4) fine-tunes the apprentice on that batch using trl's own
    `SFTTrainer` (imported from `trl`, `num_train_epochs=1`, `save_strategy="no"`), and (5) saves a
    checkpoint per outer iteration to `./checkpoints/exit_iter_{it}`.

  This is a **self-distillation / rejection-sampling expert-iteration** scheme: the "expert" is
  just the apprentice's own generation wrapped in rejection sampling against a filter, not a
  separate stronger model.

**Important integration gaps (verified, not fabricated):**
- `ExItTrainer`/`ExItConfig` are **not** exported from `trl/trainer/__init__.py` or `trl/__init__.py`
  — `from trl import ExItTrainer` will fail. You must import directly from the file.
- `ExIt_trainer.py` imports its config with a bare `from ExIt_config import ExItConfig` (not
  `from .ExIt_config import ...` or `from trl.trainer.ExIt_config import ...`). This only resolves
  if `trl/trainer/` is on `sys.path`, which happens automatically only when you run the file
  directly as a script (`python trl/trainer/ExIt_trainer.py`) — Python puts the script's own
  directory on `sys.path[0]`. Doing `import trl.trainer.ExIt_trainer` from elsewhere will raise
  `ModuleNotFoundError: No module named 'ExIt_config'`.
- There is no test file for `ExIt_trainer.py`/`ExIt_config.py` (`git ls-files | grep -i exit` finds
  only the two source files) despite a commit titled "About to generate code with Claude for
  tests" (`0698f2a`) — that commit only edited the trainer/config files, no test file was ever
  added or committed.
- No example/CLI script wires this up with argparse; the only runnable entry point is the
  `if __name__ == '__main__':` block at the bottom of `ExIt_trainer.py` (see §4).

**Separate "SAC" work exists on a different branch, not checked out here.** The remote branch
`origin/sac` (not merged into `expert-iteration`, and built on an older/divergent base of trl —
diffing `main`..`origin/sac` touches dozens of unrelated trainer files) contains
`trl/trainer/sac_config.py` (`SACConfig(OnPolicyConfig)`, fields: `reward_model_path` default
`"EleutherAI/pythia-160m"`, `num_sac_epochs=4`, `whiten_rewards=False`, `kl_coef=0.05`, `gamma=1.0`)
and `trl/trainer/sac_trainer.py`. **Both files begin with an unconditional `raise NotImplementedError()`
before any other code**, i.e. they are non-functional stubs as committed. Per the task constraints
this guide does not switch branches to investigate further.
`TODO (Matthew): if the SAC work is meant to be reproducible, decide whether to merge/finish the
sac branch and document it separately — as committed it cannot run.`

There are also several other branches (`kl-estimator-ppo`, `save-ppo-value-model`, `save_value_new`,
`ppo_main_test`, `value_trainer`) visible via `git branch -a` that were not investigated, since the
task scope is the `expert-iteration` branch.
`TODO (Matthew): confirm whether any of those branches contain work you still need documented.`

## 2. Environment

- This repo is a pip-installable package (`setup.py`, `pyproject.toml` present, package name `trl`).
  Install it editable from the repo root:
  ```bash
  pip install -e .
  ```
  (Per user convention, prefer `uv`: `uv pip install -e .` or `uv sync` if a `uv.lock`/`[project]`
  table exists — this repo's dependency metadata lives in `setup.py`, not a full `[project]` table
  in `pyproject.toml`, so `uv pip install -e .` is the safer bet.)
- Python: `setup.py` declares `python_requires=">=3.9"` and lists classifiers for 3.9–3.12.
  `TODO (Matthew): confirm exact Python version you used for the ExIt runs.` (A local, gitignored
  `wandb/` run-metadata file in this working copy — from an *unrelated* upstream PPO example run,
  `examples/scripts/ppo/tmp_ppo.py`, dated 2025-04-15, i.e. 9 days before the ExIt commits — records
  `CPython 3.11.11` on `Linux-5.15.0-1073-kvm-x86_64`. This is circumstantial evidence of the
  environment on this machine around that time, not proof of what was used for ExIt specifically.)
- Base dependencies (`setup.py` `REQUIRED_PKGS`): `accelerate>=0.34.0`, `datasets>=3.0.0`, `rich`,
  `transformers>=4.46.0`. The top-level `requirements.txt` lists the same packages unpinned
  (`accelerate`, `datasets`, `rich`, `transformers>=4.46.0`).
- `ExIt_trainer.py` additionally imports `torch`, `tqdm`, and trl's own `SFTTrainer` — all already
  transitive dependencies of `trl`/`transformers`/`accelerate`; no extra packages beyond the base
  install are needed to run the ExIt code itself.
- GPU: the built-in demo (`__main__` block, see §4) hardcodes `model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to('cuda')`
  and moves inputs `.to('cuda')` — **a CUDA GPU is required to run it as-is**; there is no CPU
  fallback in the script.

## 3. Data

The only dataset referenced by Matthew's ExIt code is loaded in the `__main__` demo block of
`trl/trainer/ExIt_trainer.py`:

```python
dataset = load_dataset("gsm8k", "main", split="train[:5]")
```

This is the public Hugging Face Hub dataset [`gsm8k`](https://huggingface.co/datasets/gsm8k)
(config `"main"`), sliced to the first 5 training examples — clearly a smoke-test slice, not a
full training run. Each example's `question` field is formatted as
`example["question"] + "\n\n###\n"` before tokenization.

No other datasets appear in `ExIt_config.py` or `ExIt_trainer.py`.

## 4. Running

There is no dedicated example script or CLI wired up (no argparse, no `HfArgumentParser` entry
point) for the ExIt trainer. The only way to run it as committed is the demo in the file's
`if __name__ == '__main__':` block:

```bash
# from the repo root, after `pip install -e .` (or `uv pip install -e .`)
python trl/trainer/ExIt_trainer.py
```

This demo, as written:
1. Loads `BASE_MODEL = "Qwen/Qwen2.5-0.5B"` (model + tokenizer) onto `cuda`.
2. Defines a length-based rejection filter (`len(decoded) > 30` chars) and builds a
   rejection-sampling `generate` wrapper via `rejection_sampling_generate_maker`.
3. Sanity-checks that wrapper on a single `"Test prompt:"` input.
4. Loads `gsm8k`/`main`, `train[:5]`, formats it, and tokenizes it into a small `Dataset` with
   `input_ids`/`attention_mask`.
5. Builds `ExItConfig(num_expert_iteration_epochs=2, expert_batch_size=2,
   per_device_train_batch_size=2, learning_rate=2e-5, expert_generation_config=GenerationConfig(do_sample=True, top_p=0.95, temperature=0.7, max_new_tokens=256, num_return_sequences=4))`.
6. Constructs `ExItTrainer(args=config, train_dataset=train_dataset, apprentice=model,
   expert_generate_maker=rs_generate_maker, processing_class=processing_class,
   data_collator=data_collator)` and calls `trainer.train()`.

Outputs of a run: per-batch `SFTTrainer` runs write to `output_dir=f"exit_round_{it}_batch_{batch_idx}"`
(relative to cwd); end-of-outer-iteration checkpoints are saved to `./checkpoints/exit_iter_{it}`
via `model.save_pretrained(...)`.

To run with different hyperparameters, data, or a real (non-toy) dataset size, you currently have
to edit the constants inside the `__main__` block directly — there is no CLI flag surface.
`TODO (Matthew): if you have a version of this script with a larger/real training run (batch size,
epoch count, dataset size beyond train[:5]) and its actual results, that configuration is not
present in this repo and should be added.`

## 5. External dependencies

- **Hugging Face Hub model**: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) —
  public/ungated, downloaded via `AutoModelForCausalLM.from_pretrained` / `AutoTokenizer.from_pretrained`.
  Requires internet access to the Hub (or a local cache) the first time it runs.
- **Hugging Face Hub dataset**: `gsm8k` (config `"main"`) — public, via `datasets.load_dataset`.
- **Weights & Biases (wandb)**: the ExIt code itself never imports `wandb` or sets `report_to=`.
  However, `TrainingArguments`/`Trainer` (used both directly and inside `SFTTrainer`) defaults to
  logging to any installed reporting integration unless `report_to` is set — and this working copy
  has a gitignored `wandb/` directory with prior local run data, indicating wandb has been used in
  this environment before. If wandb is installed and not explicitly disabled
  (`report_to="none"` in the config, or `WANDB_MODE=disabled`), a run may attempt to log to wandb
  and prompt for/require a `WANDB_API_KEY`. `TODO (Matthew): confirm whether ExIt runs are meant to
  log to wandb, and if so, which project/entity.`
- **HF authentication**: neither `Qwen/Qwen2.5-0.5B` nor `gsm8k` are gated, so no `HF_TOKEN` is
  strictly required for the demo as committed. `TODO (Matthew): confirm if any other
  models/datasets you used (e.g. for the real, non-toy runs) are gated and need `HF_TOKEN`.`
- No other environment variables or API keys are referenced anywhere in `ExIt_config.py` or
  `ExIt_trainer.py` (verified via `grep -n "wandb\|WANDB\|os.environ\|HF_TOKEN\|report_to"` against
  both files — no matches).

## Summary of required env vars

| Variable | Required? | Purpose |
|---|---|---|
| `HF_TOKEN` | Not required for the committed demo (public model/dataset) | Would be needed only if using gated HF assets |
| `WANDB_API_KEY` (or `WANDB_MODE=disabled`) | Conditionally — only if wandb is installed and `report_to` is left at its default | Experiment logging |

`TODO (Matthew): if there are other env vars used in your actual (non-demo) runs — e.g. for a
reward model API, DeepSpeed config, or multi-GPU launch — they are not present in the committed
ExIt files and should be added here.`
