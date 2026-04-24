# Hybrid Latent Reasoning via Reinforcement Learning

This repository contains the PyTorch implementation of **Hybrid Latent Reasoning via Reinforcement Learning** ([paper](https://arxiv.org/abs/2505.18454)). The project studies how to train LLMs to reason with both discrete token trajectories and latent hidden-state signals under RL. The central method, **HRPO**, introduces a learnable residual pathway that blends prior hidden states into the current token embedding during the model's "thinking" phase, while preserving the standard token-only pathway for answer generation.

The repository has two layers:

- **Project-owned entrypoints and utilities** at the top level (`hrpo_*.py`, `eval_*.py`, `utils.py`, `time_conditioning.py`, `run_*_all.sh`).
- **Vendored integration code** in `transformers/`, `trl/`, and `unsloth/`, which contains the minimal patched behavior needed to support residual reasoning and time conditioning.

If you are reproducing results or making non-trivial changes, treat [`HRPO_REPRODUCTION_GUIDE.md`](HRPO_REPRODUCTION_GUIDE.md) as the authoritative reference.

<img src=assets/intro.png width=1000>

## Training Modes

All task-specific training entrypoints now use the same explicit mode switch:

```bash
--mode {grpo,tgrpo,hrpo,thrpo}
```

| Mode | Meaning |
| --- | --- |
| `grpo` | Vanilla GRPO baseline |
| `tgrpo` | GRPO with thinking-time conditioning |
| `hrpo` | Hybrid residual reasoning |
| `thrpo` | HRPO with thinking-time conditioning |

This mode flag controls which runtime modules are enabled, which optimizer parameter groups are created, and which metadata is saved into the adapter config. Time-conditioned modes add a predictor that estimates normalized "thinking time" from recent hidden states and injects that signal through per-layer AdaLN projections.

## Repository Layout

- `hrpo_{gsm8k,math,mmlu,rag}.py`: per-task training entrypoints
- `eval_{gsm8k,math,mmlust,arcc,rag}.py`: per-task evaluation entrypoints
- `utils.py`: shared parser setup, adapter metadata, reward functions, answer normalization, and model restore helpers
- `time_conditioning.py`: time-conditioning modules, rollout/replay state, one-step-lag predictor logic, and auxiliary losses
- `time_predictor_warmup.py`: optional one-pass predictor warmup before RL for `tgrpo` and `thrpo`
- `patch.py`: trainer optimizer patch that separates base, residual, and time-conditioning parameter families, with decay/no-decay splits when applicable
- `run_{grpo,tgrpo,hrpo,thrpo}_all.sh`: multi-task orchestration wrappers
- `prepare_data.py`: unified dataset preparation for train and eval assets

## Quick Start

### Train from Python

Example: train GSM8K in `hrpo` mode.

```bash
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py \
  --mode hrpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --group_size 8 \
  --residual_r_min 0.98
```

Important arguments:

- `--mode`: one of `grpo`, `tgrpo`, `hrpo`, `thrpo`
- `--model_name`: Hugging Face model name or local base-model path
- `--group_size`: number of sampled completions per prompt inside GRPO
- `--residual_r_min`: lower bound for residual-radius initialization in residual modes
- `--temperature`: rollout sampling temperature
- `--max_steps`: short smoke-run override for the trainer
- `--max_train_samples`: dataset truncation before preprocessing
- `--thinking_time_predictor_num_hidden_states`: number of recent hidden-state outputs concatenated for the time predictor in `tgrpo`/`thrpo`
- `--time_predictor_warmup_fraction`: fraction of the training set used for predictor-only warmup before RL; set `0` to disable

Example: train GSM8K in `tgrpo` mode with an explicit predictor history length.

```bash
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py \
  --mode tgrpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --group_size 4 \
  --thinking_time_predictor_num_hidden_states 3 \
  --time_predictor_warmup_fraction 0.2
```

### Train from Shell Wrappers

The shell wrappers keep the original per-mode workflow while normalizing the internal CLI shape:

```bash
bash run_grpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_tgrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_hrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
bash run_thrpo_all.sh --tasks gsm8k --model Qwen/Qwen2.5-3B-Instruct
```

Use `--dry-run` first if you want to inspect the generated commands before launching a real job. Common launcher flags include `--tasks`, `--gpu`, `--model`, `--prep-data`, `--resume`, `--eval-only`, `--skip-eval`, `--no-wandb`, `--exp-suffix`, `--max-steps`, and `--max-train-samples`.

### Dataset Notes

- GSM8K uses `openai/gsm8k` directly at runtime; no local preparation is required.
- MATH uses local JSON files under `../MATH/{train,test}/<subject>/`.
- MMLU training expects a merged dataset at `../MMLU_Train_Merged`; MMLU-STEM and ARC-C evaluation datasets auto-download.
- RAG training expects `../RAG_Train_Merged`; RAG evaluation uses `../RAG_Eval/*_Eval`.
- `prepare_data.py` is the supported way to build or refresh train/eval assets for MATH, MMLU, and RAG:

```bash
python prepare_data.py --tasks math,mmlu,rag --stage all
python prepare_data.py --tasks rag --stage eval --with-retrieval
```

## Evaluation

Example: evaluate a saved adapter on GSM8K.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py \
  --checkpoint_path PATH/TO/CHECKPOINT \
  --batch_size 32 \
  --greedy
```

Evaluation restores the following directly from `adapter_config.json:rot_metadata` inside the checkpoint:

- training `mode`
- `base_model`
- `temperature`
- task-specific runtime knobs such as residual and time-conditioning settings

Each evaluator writes metrics and per-example generations back into the checkpoint directory.

## Artifacts and Metadata

Every saved adapter includes a `rot_metadata` block in `adapter_config.json`. That metadata is part of the runtime contract: evaluators depend on it to restore the correct mode and base model without guessing from directory names.

In practice, this means:

- checkpoint naming remains useful for humans
- checkpoint metadata remains authoritative for code
- documentation and launcher defaults should stay aligned with that contract

The metadata stores the training mode, base model, task, temperature, prompt/completion lengths, LoRA rank, group size, residual settings, time-conditioning settings, and per-family learning rates. When `--resume` is used, the launcher still computes the experiment directory from the current CLI values. For time-conditioned runs with a non-default predictor history length, pass the same `--predictor-hidden-states` / `--thinking_time_predictor_num_hidden_states` value used by the original run so the directory resolves correctly; after a checkpoint is found, the saved metadata validates the predictor input width before training continues.

## Requirements

- Python >= 3.11
- Python packages, including PyTorch/CUDA runtime packages, pinned in [`requirements.txt`](requirements.txt)
- Adapted copies of `transformers`, `trl`, and `unsloth` are vendored in this repository

## Citation

If you use this repository or the HRPO method in your research, please cite:

```bibtex
@article{yue2025hybrid,
  title={Hybrid Latent Reasoning via Reinforcement Learning},
  author={Yue, Zhenrui and Jin, Bowen and Zeng, Huimin and Zhuang, Honglei and Qin, Zhen and Yoon, Jinsung and Shang, Lanyu and Han, Jiawei and Wang, Dong},
  journal={arXiv preprint arXiv:2505.18454},
  year={2025}
}
```
