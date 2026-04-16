# Rhythm of Thought Reproduction Guide

This repository supports four training modes over the same task-specific training and evaluation entrypoints:

| Mode | Meaning |
| --- | --- |
| `grpo` | Vanilla GRPO baseline |
| `tgrpo` | GRPO with thinking-time conditioning |
| `hrpo` | Hybrid residual reasoning |
| `thrpo` | HRPO with thinking-time conditioning |

The mode is now always selected with a single explicit flag:

```bash
--mode {grpo,tgrpo,hrpo,thrpo}
```

Old boolean mode flags are no longer part of the supported interface.

## Repository Layout

The code keeps the existing split-by-task layout:

- `hrpo_gsm8k.py`, `hrpo_math.py`, `hrpo_mmlu.py`, `hrpo_rag.py`
- `eval_gsm8k.py`, `eval_math.py`, `eval_mmlust.py`, `eval_arcc.py`, `eval_rag.py`
- `run_grpo_all.sh`, `run_tgrpo_all.sh`, `run_hrpo_all.sh`, `run_thrpo_all.sh`

The shell scripts are thin task runners:

- `run_grpo_all.sh` runs `--mode grpo`
- `run_tgrpo_all.sh` runs `--mode tgrpo`
- `run_hrpo_all.sh` runs `--mode hrpo`
- `run_thrpo_all.sh` runs `--mode thrpo`

For smoke tests and short validation runs, the launchers and Python training entrypoints also support:

- `--max-steps N`
- `--max-train-samples N`

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
  --lr_time_conditioning 1e-4
```

Task-specific scripts still own dataset loading and reward definitions. The mode flag only controls the training behavior.

### Shell Entry

```bash
bash run_grpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_tgrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_hrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_thrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
```

These wrappers keep the previous per-mode workflow while standardizing the internal interface.

## Evaluation

Evaluation now loads the training configuration from the saved adapter metadata instead of inferring behavior from checkpoint naming.

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

This removes the old requirement to pass separate eval-side mode flags.

## Adapter Metadata

Every saved adapter now includes a `rot_metadata` block inside `adapter_config.json`.

Example structure:

```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
  "rot_metadata": {
    "schema_version": 1,
    "task": "gsm8k",
    "mode": "thrpo",
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "temperature": 0.7
  }
}
```

`rot_metadata` is now the standard artifact contract for training and evaluation in this repository.

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

## Meaningful WandB Metrics

The logging surface is intentionally narrower now.

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
- `time_conditioning.py` owns thinking-time state preparation and auxiliary metric calculation.
- vendored code remains only as the thin integration layer required for patched TRL, Transformers, and Unsloth behavior.

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

If a real run needs GPU scheduling, submit it through the same `srun.sh`-style workflow already used in this workspace.
