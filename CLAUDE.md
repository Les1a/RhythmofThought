# CLAUDE.md

This file provides high-level orientation for Claude Code when working in this repository. It is intentionally shorter than the reproduction guide and should be read as a workspace map, not as the final source of truth.

## Authoritative Reference

**`HRPO_REPRODUCTION_GUIDE.md` is the source of truth** for architecture, hyperparameters, dataset prep, evaluation flow, version pinning, and known gotchas. Read it before making non-trivial changes. This file is the short orientation; the guide carries the detail.

## What This Repo Is

PyTorch implementation of **HRPO** (Hybrid Latent Reasoning via Reinforcement Learning, [arXiv:2505.18454](https://arxiv.org/abs/2505.18454)). Core idea: during generation, while the model has not yet emitted the answer marker `####`, blend the previous hidden state into the current token embedding via a learnable gate (`ThinkingResidualLambda`). After `####`, generation returns to the pure token path. Training is GRPO-based and runs through TRL plus Unsloth.

When the requested change is documentation-only or CLI-surface-only, start with the project-owned top-level files before touching vendored libraries.

## Vendored Libraries (Critical)

The repo ships **modified copies** of three libraries at the top level — `transformers/`, `trl/`, and `unsloth/`. These are imported directly (no install step). Modifications live in:

- `transformers/models/qwen2/modeling_qwen2.py` and `transformers/models/llama/modeling_llama.py` — `ThinkingResidualLambda` module, `thinking_residual_gate_r/i`, and the `thinking_residual()` blending function
- `transformers/generation/utils.py` — generation-time logic that tracks `is_thinking`, builds `last_thinking_states` for residual reasoning, caches hidden states for time conditioning, and switches between greedy/sampling via `is_inference`
- `unsloth/models/llama.py` — fast prefill (line ~663) and decode (line ~940) paths that apply `thinking_residual` blending

When editing model code, **change both Qwen2 and LLaMA files** — they implement the same logic and must stay in sync.

## Architecture Overview

```
hrpo_{gsm8k,math,mmlu,rag}.py          # Per-task training entry points
   └── unsloth.FastLanguageModel        # Loads base model w/ HRPO modules auto-attached
   └── PEFT LoRA + modules_to_save      # Saves LoRA plus residual/time-conditioning modules for the selected mode
   └── trl.GRPOTrainer                  # GRPO loop, group_size completions per prompt
   └── patch.patch_trainer_optimizer    # Replaces create_optimizer with mode-aware families:
                                        #   base LoRA params          → args.learning_rate
                                        #   thinking_residual_gate    → lr_residual_gate (HRPO/THRPO)
                                        #   thinking_residual_Lambda  → lr_residual_Lambda (HRPO/THRPO)
                                        #   time_conditioning modules → lr_time_conditioning (TGRPO/THRPO)
                                        # Each family is split into decay/no-decay groups when parameters exist.

eval_{gsm8k,math,mmlust,rag,arcc}.py   # Per-task eval scripts
   └── Reads base model + mode + temperature from adapter_config.json:rot_metadata
   └── Loads adapter via model.load_adapter(checkpoint_path)
   └── Restores runtime toggles from adapter metadata

utils.py                                # ANSWER_START="####", SYSTEM_PROMPT,
                                        # mode parsing, adapter metadata IO,
                                        # reward_func_{math,mmlu,rag}, math/mmlu answer parsers
time_conditioning.py                    # Predictor, Fourier time embedding, AdaLN projection,
                                        # rollout/replay state, aux loss
time_predictor_warmup.py                # Optional predictor-only warmup for TGRPO/THRPO
patch.py                                # Optimizer surgery by parameter family
prepare_data.py                         # Unified train + eval data preparation (all tasks)
run_{grpo,tgrpo,hrpo,thrpo}_all.sh      # Unified train+eval orchestration with smart skip logic
```

### Training Modes (`--mode`)

Training scripts use a single explicit `--mode {grpo,tgrpo,hrpo,thrpo}` switch:

- `grpo` — vanilla GRPO baseline
- `tgrpo` — GRPO + time conditioning, no thinking residual
- `hrpo` — thinking residual only
- `thrpo` — thinking residual + time conditioning

Eval scripts no longer accept mode flags. They restore mode, base model, and temperature from adapter metadata.

### Adapter Metadata (load-bearing)

Every saved adapter config includes a `rot_metadata` block with the training mode, base model, task, temperature, and key hyperparameters. Eval loads this metadata instead of inferring anything from the checkpoint path. Directory naming is still useful for humans, but no longer load-bearing for runtime correctness.

For time-conditioned checkpoints, `thinking_time_predictor_num_hidden_states` is load-bearing too. Resume uses it in the experiment directory name and validates it against metadata after finding a checkpoint; eval restores it directly from metadata so predictor input width matches the saved adapter.

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

# GRPO and TGRPO baselines
bash run_grpo_all.sh --tasks gsm8k --no-wandb
bash run_tgrpo_all.sh --tasks gsm8k --no-wandb

# Time-conditioned HRPO, with explicit predictor history and warmup fraction
bash run_thrpo_all.sh --tasks gsm8k --predictor-hidden-states 3 --time-predictor-warmup-fraction 0.2

# Dry run (print commands, don't execute)
bash run_hrpo_all.sh --dry-run

# Standalone data preparation (can also be invoked directly)
python prepare_data.py                                   # all tasks, train + eval
python prepare_data.py --tasks math,rag --stage train    # only math+rag training data
python prepare_data.py --tasks rag --stage eval --with-retrieval  # RAG eval with BM25
```

Smart skip: training is skipped if `experiments/{exp_name}/checkpoint-*` exists; eval is skipped if `eval_results*.json` exists in the checkpoint dir. Use `--resume` to continue training instead of skipping.

Time-predictor warmup is only for `tgrpo` and `thrpo`. It is skipped on resume, uses a deterministic shuffled subset, freezes all non-predictor parameters, and writes no checkpoints.

### Individual scripts

```bash
# Train (single task)
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py --mode hrpo --model_name Qwen/Qwen2.5-3B-Instruct --group_size 4 --residual_r_min 0.99
CUDA_VISIBLE_DEVICES=0 python hrpo_math.py  --mode hrpo --dataset_root ../MATH               --group_size 8
CUDA_VISIBLE_DEVICES=0 python hrpo_mmlu.py  --mode hrpo --dataset_root ../MMLU_Train_Merged  --group_size 8
CUDA_VISIBLE_DEVICES=0 python hrpo_rag.py   --mode hrpo --dataset_root ../RAG_Train_Merged   --group_size 4

# Eval (reads mode/base model/temperature from adapter metadata)
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py --checkpoint_path ./experiments/.../checkpoint-935 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_math.py  --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_mmlust.py --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_rag.py    --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128
CUDA_VISIBLE_DEVICES=0 python eval_arcc.py   --checkpoint_path ./experiments/.../checkpoint-500 --batch_size 128
```

## Key Training Metrics

When monitoring runs (WandB project: `latent-reasoning`):

- `residual_active_fraction`, `residual_embed_ratio`, `residual_hidden_ratio` — rollout-time residual usage
- `thinking_time_aux_loss`, `thinking_time_pred_error`, `thinking_time_rollout_error`, `thinking_time_embed_std` — time-conditioning supervision
- `reward`, `completion_length`, `kl` — standard GRPO signals.

## Environment Pinning Gotchas

These are documented in `HRPO_REPRODUCTION_GUIDE.md` under environment and version notes. Read that section before bumping dependencies:

- `requirements.txt` currently pins `accelerate==1.5.2`, `datasets==3.4.1`, `peft==0.15.0`, `tokenizers==0.21.1`, `torch==2.5.1`, `vllm==0.7.3`, and `wandb==0.19.8`.
- `transformers/`, `trl/`, and `unsloth/` are vendored and patched; do not replace them with upstream installs unless you re-audit residual and time-conditioning hooks.
- Eval import order matters: use `utils.build_generation_config()` so Unsloth is imported before `GenerationConfig`.
