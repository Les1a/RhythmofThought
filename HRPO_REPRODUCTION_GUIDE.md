# Rhythm of Thought Reproduction Guide

This document is the authoritative reproduction guide for the current workspace. It explains the supported training modes, launcher surface, adapter metadata contract, and the practical verification steps needed to reproduce or audit experiments without reverse-engineering the codebase.

The repository supports four training modes over the same task-specific training and evaluation entrypoints:

| Mode | Meaning |
| --- | --- |
| `grpo` | Vanilla GRPO baseline |
| `tgrpo` | GRPO with thinking-time conditioning |
| `hrpo` | Hybrid residual reasoning |
| `thrpo` | HRPO with thinking-time conditioning |

The mode is always selected with a single explicit flag:

```bash
--mode {grpo,tgrpo,hrpo,thrpo}
```

Legacy boolean mode toggles are no longer part of the supported interface and should be treated as deprecated.

## Repository Layout

Project-owned files keep the split-by-task layout:

- `hrpo_gsm8k.py`, `hrpo_math.py`, `hrpo_mmlu.py`, `hrpo_rag.py`
- `eval_gsm8k.py`, `eval_math.py`, `eval_mmlust.py`, `eval_arcc.py`, `eval_rag.py`
- `run_grpo_all.sh`, `run_tgrpo_all.sh`, `run_hrpo_all.sh`, `run_thrpo_all.sh`
- `utils.py`, `time_conditioning.py`, `time_predictor_warmup.py`, `patch.py`
- `prepare_data.py`, `srun.sh`, and the `test/` regression suite

The shell launchers are intentionally thin task runners:

- `run_grpo_all.sh` runs `--mode grpo`
- `run_tgrpo_all.sh` runs `--mode tgrpo`
- `run_hrpo_all.sh` runs `--mode hrpo`
- `run_thrpo_all.sh` runs `--mode thrpo`

For smoke tests and short validation runs, the shell launchers and Python training entrypoints also support:

- `--max-steps N`
- `--max-train-samples N`

## Data Preparation

`prepare_data.py` is the supported data-preparation entrypoint. It uses the same comma-separated `--tasks` shape as the launchers and can prepare train, eval, or both stages.

```bash
python prepare_data.py
python prepare_data.py --tasks math,mmlu --stage train
python prepare_data.py --tasks rag --stage eval --with-retrieval
python prepare_data.py --tasks rag --stage eval --force-retrieval-only
```

Per-task output contract:

| Task | Train output | Eval output |
| --- | --- | --- |
| `gsm8k` | no local output; `openai/gsm8k` auto-downloads | no local output |
| `math` | `../MATH/train/<subject>/<idx>.json` | `../MATH/test/<subject>/<idx>.json`; `eval_math.py` also auto-downloads MATH-500 |
| `mmlu` | `../MMLU_Train_Merged` via Hugging Face `save_to_disk` | no local prep; MMLU-STEM and ARC-C auto-download |
| `rag` | `../RAG_Train_Merged` from SQuAD | `../RAG_Eval/{HotpotQA,2Wiki,NQ,TQ,Bamboogle}_Eval` |

For RAG eval, HotpotQA and 2Wiki use gold contexts from FlashRAG. NQ, TriviaQA, and Bamboogle are closed-book unless `--with-retrieval` builds the lightweight BM25 retriever. The BM25 path is a convenience approximation over HotpotQA and 2Wiki paragraphs; missing `rank_bm25` falls back to closed-book outputs with a warning.

## Training

### Direct Python Entry

```bash
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py \
  --mode hrpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --group_size 8 \
  --residual_r_min 0.98
```

For time-conditioning variants, switch the mode and provide the time-conditioning hyperparameters you want to tune:

```bash
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py \
  --mode tgrpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --group_size 8 \
  --thinking_time_loss_weight 0.1 \
  --lr_time_conditioning 5e-6
```

Task-specific scripts still own dataset loading and reward definitions. The mode flag only controls the shared training behavior and runtime wiring.

### Shell Entry

```bash
bash run_grpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_tgrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_hrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_thrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
```

These wrappers preserve the previous per-mode workflow while standardizing the internal interface and metadata behavior.

Launcher options that map directly to the Python training stack:

| Launcher flag | Python argument | Notes |
| --- | --- | --- |
| `--max-steps N` | `--max_steps N` | Smoke-run cap passed into `GRPOConfig.max_steps` |
| `--max-train-samples N` | `--max_train_samples N` | Deterministic dataset truncation before preprocessing |
| `--exp-suffix NAME` | `--exp-suffix NAME` | Sanitized and appended to the computed experiment directory |
| `--resume` | `--resume` | Continues from the latest `checkpoint-*` in the computed experiment directory |
| `--predictor-hidden-states N` | `--thinking_time_predictor_num_hidden_states N` | Applies only to `tgrpo` and `thrpo` |
| `--time-predictor-warmup-fraction V` | `--time_predictor_warmup_fraction V` | Applies only to `tgrpo` and `thrpo`; default `0.2` |

The launchers skip training when a matching checkpoint already exists and skip evaluation when the expected eval-results file exists. Use `--resume` to continue training instead of skipping; use `--exp-suffix` to keep a new run separate from old checkpoints.

## Evaluation

Evaluation now restores the training configuration from saved adapter metadata instead of inferring behavior from checkpoint naming.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py \
  --checkpoint_path PATH/TO/CHECKPOINT \
  --batch_size 16 \
  --greedy
```

The evaluator reads `adapter_config.json` inside the checkpoint and restores:

- `mode`
- `base_model`
- `task`
- `temperature`
- key mode-specific hyperparameters

This removes the old requirement to pass separate evaluation-side mode flags and makes checkpoint directories less brittle as a runtime interface.

## Adapter Metadata

Every saved adapter now includes a `rot_metadata` block inside `adapter_config.json`.

Example structure:

```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
  "rot_metadata": {
    "schema_version": 2,
    "task": "gsm8k",
    "mode": "thrpo",
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "temperature": 0.7,
    "group_size": 4,
    "lora_rank": 32,
    "max_prompt_length": 1024,
    "max_completion_length": 1024,
    "use_thinking_residual": true,
    "use_time_conditioning": true,
    "residual_r_min": 0.99,
    "residual_r_max": 0.999,
    "thinking_time_loss_weight": 0.1,
    "thinking_time_predictor_num_hidden_states": 3,
    "lr": 5e-6,
    "lr_residual_gate": 1e-4,
    "lr_residual_lambda": 1e-3,
    "lr_time_conditioning": 5e-6
  }
}
```

`rot_metadata` is the standard artifact contract for training and evaluation in this repository.

## Mode Semantics

### `grpo`

- No residual reasoning path is used during rollout or replay.
- `answer_start` is cleared so generation stays on the plain token path.

### `tgrpo`

- Keeps the GRPO path.
- Adds thinking-time conditioning and its auxiliary supervision.

### `hrpo`

- Enables residual reasoning modules and their optimizer groups.
- Uses `answer_start` so generation can switch from thinking to answer mode.

### `thrpo`

- Combines HRPO residual reasoning with thinking-time conditioning.

## Time-Conditioning Runtime

Time-conditioned modes (`tgrpo`, `thrpo`) add three trainable components:

- `thinking_time_predictor`: predicts a scalar in `[0, 1]` from the most recent hidden-state outputs.
- `reasoning_time_embedding`: maps the scalar into Fourier-style features with learnable bandwidth.
- per-layer `adaln_proj`: projects the embedding into bounded AdaLN modulation chunks for attention/MLP sites.

The predictor input width is `hidden_size * thinking_time_predictor_num_hidden_states`; the default history length is `3`. Rollout uses a one-step-lag contract: the next token's conditioning is predicted from cached hidden states produced by previous generated tokens. Finished rows are masked after the `####` answer marker so time conditioning only applies during the thinking span.

Training replay uses position-aligned tensors from TRL:

1. `thinking_mask` marks the generated thinking span before `####`.
2. `compute_thinking_time_targets()` creates a normalized `0 -> 1` ramp over that span.
3. `select_training_thinking_time_embedding()` reuses rollout-predicted thinking time, adds small fixed-range noise, masks non-thinking positions, and caches the resulting embedding.
4. `compute_thinking_time_aux_loss()` predicts from replay hidden states using rollout-equivalent lagged carry and applies reward-scaled auxiliary weights.

This split keeps online rollout behavior and replay-time supervision aligned without using future hidden states as predictor input.

## Predictor Warmup

`time_predictor_warmup.py` can run one predictor-only pass before RL for `tgrpo` and `thrpo`.

- Enabled by `--time_predictor_warmup_fraction` / `--time-predictor-warmup-fraction`; default is `0.2`.
- Disabled for `grpo` and `hrpo`, for fraction `0`, and whenever `--resume` resolves a checkpoint.
- Builds a deterministic shuffled subset using `ceil(len(dataset) * fraction)`.
- Reuses the same trainer class and optimizer patch, but freezes every parameter except `thinking_time_predictor`.
- Uses an isolated `predictor_warmup` output directory, disables saving and external reporting, then restores gradient flags and clears cached time-conditioning state.

Warmup metrics are forwarded with a `warmup/` prefix when a metrics callback is supplied.

## Checkpoint and Resume Contract

Experiment directories are computed from model short name, task, mode, group size, LoRA rank, residual radius when applicable, temperature, predictor history length when applicable, and optional suffix. Examples:

```text
./experiments/Qwen2.5-3B-Instruct-gsm8k-grpo-group4-lora32-temp0.5
./experiments/Qwen2.5-3B-Instruct-gsm8k-tgrpo-group4-lora32-temp0.5-last3
./experiments/Qwen2.5-3B-Instruct-gsm8k-hrpo-group4-lora32-rmin0.99-temp0.5
./experiments/Qwen2.5-3B-Instruct-gsm8k-thrpo-group4-lora32-rmin0.99-temp0.5-last3
```

`--resume` resolves the latest checkpoint in that computed directory. For time-conditioned modes, the directory name includes the predictor history length, so resuming a non-default history length requires passing the same `--predictor-hidden-states` / `--thinking_time_predictor_num_hidden_states` value used by the original run. After the checkpoint is found, an explicitly supplied predictor history length must match `rot_metadata.thinking_time_predictor_num_hidden_states`; implicit CLI defaults are replaced by the checkpoint value before model construction.

## Meaningful WandB Metrics

The logging surface is intentionally narrower so dashboards focus on decisions rather than debugging noise.

Always-available training metrics:

- `reward`
- `reward_std`
- `completion_length`
- `kl`
- `rewards/<reward_fn_name>`

Residual-mode metrics:

- `residual_active_fraction`
- `residual_embed_ratio`
- `residual_hidden_ratio`

Time-conditioning metrics:

- `thinking_time_aux_loss`
- `thinking_time_pred_error`
- `thinking_time_rollout_error`
- `thinking_time_embed_std`

Older low-signal metrics such as `embeds_ratio`, `hidden_ratio`, replay noise debug scalars, and auxiliary-weight summary stats are no longer part of the standard logging contract.

## Implementation Boundaries

The simplification pass moved most mode handling, metadata IO, and state restoration into repository-owned code:

- `utils.py` owns parser setup, mode normalization, adapter metadata write/read, and eval model restore.
- `time_conditioning.py` owns thinking-time modules, rollout/replay state preparation, and auxiliary metric calculation.
- `time_predictor_warmup.py` owns optional predictor-only warmup before RL.
- `patch.py` owns optimizer grouping for base, residual, and time-conditioning parameter families, including decay/no-decay splits.
- vendored code remains the thin integration layer required for patched TRL, Transformers, and Unsloth behavior.

## Environment and Version Notes

Use the `rot` conda environment in this workspace unless you intentionally rebuild the dependency stack. `requirements.txt` records the current Python package pins used by the project, including `accelerate==1.5.2`, `datasets==3.4.1`, `peft==0.15.0`, `tokenizers==0.21.1`, `torch==2.5.1`, `vllm==0.7.3`, and `wandb==0.19.8`.

The repository imports vendored `transformers/`, `trl/`, and `unsloth/` from the workspace. Do not replace those directories with upstream packages without re-auditing the residual and time-conditioning hooks.

Known gotchas:

- Import Unsloth before constructing `GenerationConfig`; `utils.build_generation_config()` exists to preserve that order in eval scripts.
- Keep Qwen2 and LLaMA residual-model patches in sync when editing vendored model code.
- Time-conditioned resume runs must use the predictor hidden-state count saved in checkpoint metadata, and the CLI value must also match the computed experiment directory name.
- The RAG BM25 retriever is best-effort convenience tooling, not the paper's E5 + Wikipedia ANN retrieval stack.

## Recommended Verification

Use the `rot` environment for tests:

```bash
/storage/home/hcoda1/7/kxia39/workspace/conda3/envs/rot/bin/python -m pytest \
  test/test_time_conditioning.py \
  test/test_training_mode_utils.py -q
```

Dry-run the shell entrypoints before launching full jobs:

```bash
bash run_grpo_all.sh --dry-run --tasks gsm8k
bash run_tgrpo_all.sh --dry-run --tasks gsm8k
bash run_hrpo_all.sh --dry-run --tasks gsm8k
bash run_thrpo_all.sh --dry-run --tasks gsm8k
```

For a real 1-step launcher smoke test on GPU:

```bash
bash srun.sh --name thrpo-smoke --time 01:00:00 \
  "bash run_thrpo_all.sh --tasks mmlu --skip-eval --no-wandb --max-steps 1 --max-train-samples 8 --exp-suffix smoke"
```

Successful smoke-test logs should include all of the following:

- `Launch config: max_steps=1, max_train_samples=8`
- `Num examples = 8 | ... | Total steps = 1`
- `[train-config] ... dataset_size=8 max_steps=1`
- `checkpoint: checkpoint-1`
- `All tasks completed successfully!`

If a real run needs GPU scheduling, submit it through the same `srun.sh`-style workflow already used in this workspace so environment activation, logging, and Slurm defaults stay consistent.
