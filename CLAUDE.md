# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authoritative Reference

**`HRPO_REPRODUCTION_GUIDE.md` is the source of truth** for architecture, hyperparameters, dataset prep, eval pipeline, version pinning, and known gotchas. Read it before making non-trivial changes. This file is the high-level orientation; the guide is the detail.

## What This Repo Is

PyTorch implementation of **HRPO** (Hybrid Latent Reasoning via Reinforcement Learning, [arXiv:2505.18454](https://arxiv.org/abs/2505.18454)). Core idea: during generation, while the model has not yet emitted the answer marker `####`, blend the prior hidden state into the current token embedding via a learnable gate (`ThinkingResidualLambda`). After `####`, generation reverts to pure token embeddings. Training is GRPO-based via TRL + Unsloth.

## Vendored Libraries (Critical)

The repo ships **modified copies** of three libraries at the top level — `transformers/`, `trl/`, and `unsloth/`. These are imported directly (no install step). Modifications live in:

- `transformers/models/qwen2/modeling_qwen2.py` and `transformers/models/llama/modeling_llama.py` — `ThinkingResidualLambda` module, `thinking_residual_gate_r/i`, and the `thinking_residual()` blending function
- `transformers/generation/utils.py` — generation-time logic that tracks `is_thinking`, builds `last_thinking_states` (probability-weighted embedding average), and switches between greedy/sampling via `is_inference`
- `unsloth/models/llama.py` — fast prefill (line ~663) and decode (line ~940) paths that apply `thinking_residual` blending

When editing model code, **change both Qwen2 and LLaMA files** — they implement the same logic and must stay in sync.

## Architecture Overview

```
hrpo_{gsm8k,math,mmlu,rag}.py          # Per-task training entry points
   └── unsloth.FastLanguageModel        # Loads base model w/ HRPO modules auto-attached
   └── PEFT LoRA + modules_to_save      # Trains LoRA + thinking_residual_{gate_r,gate_i,Lambda}
   └── trl.GRPOTrainer                  # GRPO loop, group_size completions per prompt
   └── patch.patch_trainer_optimizer    # Replaces create_optimizer with 4 param groups:
                                        #   1. Decay LoRA params      → args.learning_rate
                                        #   2. No-decay LoRA params   → args.learning_rate
                                        #   3. thinking_residual_gate → lr_residual_gate (1e-4)
                                        #   4. thinking_residual_Lambda → lr_residual_Lambda (1e-3)

eval_{gsm8k,math,mmlust,rag,arcc}.py   # Per-task eval scripts
   └── Auto-detects base model + temperature from checkpoint path naming convention
   └── Loads adapter via model.load_adapter(checkpoint_path)
   └── Sets model.answer_start = "####" UNLESS --only_grpo is passed

utils.py                                # ANSWER_START="####", SYSTEM_PROMPT, BASE_MODELS list,
                                        # reward_func_{math,mmlu,rag}, math/mmlu answer parsers
patch.py                                # Optimizer surgery (4 param groups)
prepare_data.py                         # Unified train + eval data preparation (all tasks)
run_hrpo_all.sh / run_grpo_all.sh       # Unified train+eval orchestration with smart skip logic
```

### HRPO vs GRPO baseline (`--only_grpo`)

Every training and eval script accepts `--only_grpo`. This is the ablation switch that disables HRPO entirely:

- Training: skips `model.answer_start`, `modules_to_save`, `reset_lambda_parameters()`, and `patch_trainer_optimizer()`. Result: vanilla GRPO LoRA training.
- Eval: skips `model.answer_start`. Generation runs without thinking residual blending.
- Output dirs differ: HRPO uses `...-group{G}-lora{R}-rmin{MIN}-temp{T}`, GRPO uses `...-grpo-group{G}-lora{R}-temp{T}`. Both can coexist under `experiments/`.
- **When evaluating a GRPO checkpoint, you MUST pass `--only_grpo`.** Otherwise the untrained `thinking_residual_*` modules (random init) get applied during generation and corrupt outputs.

The safety net for this lives in `transformers/generation/utils.py`: thinking-state computation is guarded by `if getattr(self, 'answer_start', None) is not None`.

### Experiment naming convention (load-bearing)

Eval scripts parse the checkpoint path to recover the base model and temperature:

```python
# from utils.detect_base_model + eval_*.py
base_model = next(m for m in BASE_MODELS if m.split('/')[-1] in checkpoint_path)
temperature = float(checkpoint_path.split('-temp')[-1].split('/')[0])
```

If you rename experiment dirs or invent new naming patterns, eval will break. The format is enforced in each `hrpo_*.py` `main()` function.

## Common Commands

The conda env name is **`rot`** (referenced by the unified shell scripts).

### Unified pipeline (preferred)

```bash
# HRPO: train + eval all four tasks (GSM8K, MATH, MMLU, RAG) on GPU 0
bash run_hrpo_all.sh

# Fresh checkout: prepare all data then train + eval
bash run_hrpo_all.sh --prep-data

# Specific task / GPU / model
bash run_hrpo_all.sh --gpu 1 --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct

# Resume training from latest checkpoint (full state restore)
bash run_hrpo_all.sh --resume --tasks gsm8k

# Eval-only against existing checkpoints
bash run_hrpo_all.sh --eval-only --tasks gsm8k,math

# Paper-original batch sizes (default is H200-optimized: BS↑, GA=1, same effective BS)
bash run_hrpo_all.sh --paper-params

# GRPO baseline (same flags, passes --only_grpo to all subcommands)
bash run_grpo_all.sh --tasks gsm8k --no-wandb

# Dry run (print commands, don't execute)
bash run_hrpo_all.sh --dry-run

# Standalone data preparation (can also be invoked directly)
python prepare_data.py                                   # all tasks, train + eval
python prepare_data.py --tasks math,rag --stage train    # only math+rag training data
python prepare_data.py --tasks rag --stage eval --with-retrieval  # RAG eval with BM25
```

Smart skip: training is skipped if `experiments/{exp_name}/checkpoint-*` exists; eval is skipped if `eval_results*.json` exists in the checkpoint dir. Use `--resume` to continue training instead of skipping.

### Individual scripts

```bash
# Train (single task)
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py --model_name Qwen/Qwen2.5-3B-Instruct --group_size 4 --residual_r_min 0.99
CUDA_VISIBLE_DEVICES=0 python hrpo_math.py  --dataset_root ../MATH               --group_size 8
CUDA_VISIBLE_DEVICES=0 python hrpo_mmlu.py  --dataset_root ../MMLU_Train_Merged  --group_size 8
CUDA_VISIBLE_DEVICES=0 python hrpo_rag.py   --dataset_root ../RAG_Train_Merged   --group_size 4

# Eval (path determines base_model and temperature — see naming convention above)
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py --checkpoint_path ./experiments/.../checkpoint-935 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_math.py  --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_mmlust.py --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_rag.py    --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_arcc.py   --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128

# GRPO baseline — note --only_grpo on BOTH training and eval
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py --only_grpo --model_name Qwen/Qwen2.5-1.5B-Instruct --group_size 4
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py --only_grpo --checkpoint_path ./experiments/...-grpo-.../checkpoint-250
```

## Supported Models

`utils.BASE_MODELS` lists the recognized bases (used by `detect_base_model()` in every eval script):

- `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`

Adding a new model requires updating this list AND ensuring the architecture file under `transformers/models/` has the `ThinkingResidualLambda` modifications.

## Key Training Metrics

When monitoring runs (WandB project: `latent-reasoning`):

- `embeds_ratio` — average `a_t` (embedding weight). Starts ~0.997, should decrease as training progresses.
- `hidden_ratio` — average `sqrt(1-a_t^2)` (residual weight). Starts ~0.074, should increase.
- `reward`, `completion_length`, `kl` — standard GRPO signals.

If `embeds_ratio` never moves off ~0.997, the Lambda learning rate (`--lr_residual_Lambda`, default 1e-3 — 200× the main LR) is being suppressed; check that `patch_trainer_optimizer()` ran.

## Environment Pinning Gotchas

These are documented in `HRPO_REPRODUCTION_GUIDE.md` §7 and §8 — read that section before bumping any dep:

- `accelerate==1.5.2` (newer versions break the unsloth-patched `_fast_inner_training_loop` on `FP8BackendType`)
- `tokenizers>=0.21,<0.22` (vendored `transformers/` requires this; unsloth pulls 0.22.x by default)
- `torch==2.10.0+cu128` (unsloth caps at `torch<2.11.0`)
- `peft==0.18.1`, `trl==0.24.0`, `datasets==4.3.0`, `bitsandbytes==0.49.2`, `math-verify==0.9.0`
