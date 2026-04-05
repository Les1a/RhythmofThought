# HRPO (Hybrid Latent Reasoning via Reinforcement Learning) -- Full Reproduction Guide

> Paper: [Hybrid Latent Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.18454)
>
> Yue, Jin, Zeng, Zhuang, Qin, Yoon, Shang, Han, Wang (2025)

---

## 1. Overview

HRPO is an RL-based hybrid latent reasoning approach that allows LLMs to "think in hidden space" while generating tokens. Unlike Chain-of-Thought which uses explicit text tokens, HRPO blends hidden states from the "thinking" phase into token embeddings via a learnable gating mechanism, enabling latent reasoning through continuous representations.

**Core Idea:** During generation, if the model has NOT yet produced the answer marker `####`, it's in "thinking" mode. In this phase, the current token embedding is blended with a weighted summary of prior hidden states (the "thinking residual"). Once `####` appears, the model switches to "answer" mode and uses pure token embeddings.

**Training Framework:** Built on GRPO (Group Relative Policy Optimization) via TRL + Unsloth, with custom parameter groups and learning rates for the gating mechanism.

---

## 2. Architecture Modifications (Key Code Changes)

### 2.1 ThinkingResidualLambda Module

**Files:**
- `transformers/models/qwen2/modeling_qwen2.py` (line 469)
- `transformers/models/llama/modeling_llama.py` (line 491)

This is the learnable decay parameter that controls how strongly hidden states are blended:

```python
class ThinkingResidualLambda(nn.Module):
    c = 8.0  # Fixed constant controlling decay sharpness

    def __init__(self, config):
        super().__init__()
        self.Lambda = nn.Parameter(torch.randn(config.hidden_size))

    def reset_lambda_parameters(self, r_min=0.9, r_max=0.999):
        """Initialize Lambda so that a_t starts close to 1 (mostly embedding, little residual)."""
        with torch.no_grad():
            nn.init.uniform_(self.Lambda, a=r_min, b=r_max)
            # Apply inverse-sigmoid-like transform: Lambda = -log(Lambda^(-1/c) - 1)
            self.Lambda.data.copy_(
                - torch.log((self.Lambda ** (-1. / self.c)) - 1)
            )

    def forward(self, r_t):
        # a_t = exp(-c * softplus(-Lambda) * r_t)
        # When Lambda is large (initialized near 1), softplus(-Lambda) is small, so a_t is near 1
        a_t = torch.exp(
            - self.c * nn.functional.softplus(-self.Lambda, beta=1, threshold=20) * r_t
        )
        return a_t
```

**Key Insight:** At initialization with `r_min=0.99, r_max=0.999`, the decay `a_t` is very close to 1.0 (observed ~0.997 in training logs). This means the model starts by almost entirely using token embeddings, and gradually learns to incorporate more hidden-state information as training progresses.

---

### 2.2 Thinking Residual Gating in Model

**Files:**
- `transformers/models/qwen2/modeling_qwen2.py` (lines 517-534)
- `transformers/models/llama/modeling_llama.py` (lines 539-556)

Three new modules are added to the base `Model.__init__`:

```python
# In Qwen2Model.__init__ / LlamaModel.__init__:
self.thinking_residual_gate_r = nn.Linear(config.hidden_size, config.hidden_size)
self.thinking_residual_gate_i = nn.Linear(config.hidden_size, config.hidden_size)
self.thinking_residual_Lambda = ThinkingResidualLambda(config)
```

The blending function:

```python
def thinking_residual(self, embeds, residual, eps=1e-8):
    r_t = torch.sigmoid(self.thinking_residual_gate_r(embeds))  # Reset gate
    i_t = torch.sigmoid(self.thinking_residual_gate_i(embeds))  # Input gate
    a_t = self.thinking_residual_Lambda(r_t)                     # Decay amplitude
    # Pythagorean blending: maintains approximate unit norm
    return a_t * embeds + torch.sqrt(1 - a_t.pow(2) + eps) * (i_t * residual), a_t
```

**Mathematical Formula:**

```
output = a_t * embeds + sqrt(1 - a_t^2) * (i_t * residual)
```

Where:
- `embeds`: Current token embedding (shape: [B, H])
- `residual`: Prior hidden state / thinking state (shape: [B, H])
- `r_t = sigmoid(W_r * embeds)`: Reset gate -- controls how much the current embedding influences the decay
- `i_t = sigmoid(W_i * embeds)`: Input gate -- controls how much of the residual passes through
- `a_t = Lambda(r_t)`: Decay amplitude -- how much to keep the embedding vs. blend in residual
- The `sqrt(1 - a_t^2)` term ensures the output maintains approximately unit norm (Pythagorean identity)

---

### 2.3 Generation-Time Logic: Thinking vs. Answer Mode

**File:** `transformers/generation/utils.py` (lines 3286-3381)

During autoregressive generation, two key modifications are made:

#### a) Greedy vs. Sampling based on `is_inference` flag (line 3352):

```python
probs = nn.functional.softmax(next_token_scores, dim=-1)
if is_inference:
    next_tokens = torch.argmax(probs, dim=-1)   # Greedy (eval)
else:
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # Sampling (train)
```

#### b) Computing thinking states and thinking mask (line 3367):

```python
# Check if "####" (answer marker) has appeared in generated text
strs = processing_class.batch_decode(input_ids[:, input_len:])
is_thinking = [self.answer_start not in s for s in strs]

# Compute "last_thinking_states" as probability-weighted embedding average
last_thinking_states = torch.einsum(
    'bv,vd->bd', probs, self.get_input_embeddings().weight
)
# Normalize
last_thinking_states /= torch.sqrt((probs ** 2).sum(-1, keepdim=True))
```

**Key Insight:** `last_thinking_states` is NOT simply the embedding of the sampled token. It's a soft weighted average of ALL token embeddings weighted by the output probabilities. This is richer than a hard one-hot embedding and carries more distributional information.

#### c) Generation initialization (line 3286):

```python
is_prefill = True
is_thinking, last_thinking_states = None, None
thinking_embeds = [self.get_input_embeddings()(input_ids)] if return_thinking_embeds else []
thinking_mask = [torch.zeros_like(input_ids, dtype=torch.bool)] if return_thinking_embeds else []
embeds_ratio = [torch.ones_like(input_ids, dtype=torch.float32)] if return_thinking_embeds else []
```

#### d) Return values (line 3417):

When `return_thinking_embeds=True` (training), the generation returns a 4-tuple:
```python
return input_ids, torch.cat(thinking_embeds, dim=1), torch.cat(thinking_mask, dim=1), torch.cat(embeds_ratio, dim=1)
```

When `return_thinking_embeds=False` (inference), it returns only `input_ids`.

---

### 2.4 Unsloth Fast Inference Path

**File:** `unsloth/models/llama.py`

#### Prefill phase (line 663): Apply thinking_residual during batch prefill

```python
thinking_mask = kwargs.get('thinking_mask')
if thinking_mask is not None:
    new_inputs_embeds = inputs_embeds.clone()
    new_inputs_embeds[thinking_mask] = self.thinking_residual(
        inputs_embeds[thinking_mask], thinking_embeds[thinking_mask],
    )[0].to(inputs_embeds.dtype)
    inputs_embeds = new_inputs_embeds
```

#### Decode phase (line 940): Apply thinking_residual during token-by-token generation

```python
is_thinking = kwargs.get('is_thinking')
last_thinking_states = kwargs.get('last_thinking_states')
if is_thinking is not None and last_thinking_states is not None:
    thinking_embeds = last_thinking_states
    X_hat, a_t = self.model.thinking_residual(
        X, last_thinking_states.unsqueeze(1),
    )
    embeds_ratio = a_t.mean(-1).squeeze()
    embeds_ratio[~torch.tensor(is_thinking)] = 1.  # No blending for answer tokens
    X[is_thinking] = X_hat[is_thinking].to(X.dtype)  # Only blend thinking tokens
```

`hidden_states` is overloaded to return `[thinking_embeds, is_thinking, embeds_ratio]` from the model (line 1025).

---

## 3. Training Pipeline

### 3.1 Overall Flow

```
1. Load pretrained model (Qwen2.5 / LLaMA) via Unsloth FastLanguageModel
   - Model automatically includes thinking_residual modules (newly initialized)

2. Apply LoRA adapters + register thinking_residual modules as `modules_to_save`
   - LoRA targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
   - modules_to_save: thinking_residual_gate_r, thinking_residual_gate_i, thinking_residual_Lambda
   - lora_alpha = lora_rank * 2 (e.g., rank=32 -> alpha=64)

3. Initialize Lambda parameters via reset_lambda_parameters(r_min, r_max)
   - Default: r_min=0.99, r_max=0.999 -> a_t starts near 1.0 (mostly embedding)

4. Configure GRPO training (GRPOConfig):
   - Multiple generations per prompt (group_size=4~8)
   - Temperature-based sampling
   - Task-specific reward functions

5. Patch trainer optimizer (patch.py):
   - Main LoRA params: lr (e.g., 5e-6)
   - thinking_residual_gate: separate lr (e.g., 1e-4)
   - thinking_residual_Lambda: separate lr (e.g., 1e-3)

6. Train with GRPO:
   - For each prompt, generate group_size completions
   - Evaluate with reward function (correctness + format compliance)
   - Compute group-relative advantages and update policy
```

### 3.2 Custom Optimizer Setup (`patch.py`)

The optimizer is patched to use 4 parameter groups with different learning rates:

```python
optimizer_grouped_parameters = [
    {   # Group 1: Regular trainable params WITH weight decay
        "params": [p for n, p in model.named_parameters()
                   if "thinking_residual" not in n and n in decay_parameters],
        "lr": args.learning_rate,         # e.g., 5e-6
        "weight_decay": args.weight_decay,
    },
    {   # Group 2: Regular trainable params WITHOUT weight decay
        "params": [p for n, p in model.named_parameters()
                   if "thinking_residual" not in n and n not in decay_parameters],
        "lr": args.learning_rate,
        "weight_decay": 0.0,
    },
    {   # Group 3: thinking_residual_gate (r and i)
        "params": [p for n, p in model.named_parameters()
                   if "thinking_residual_gate" in n],
        "lr": lr_thinking_residual_gate,   # e.g., 1e-4 (20x main lr)
        "weight_decay": args.weight_decay,
    },
    {   # Group 4: thinking_residual_Lambda
        "params": [p for n, p in model.named_parameters()
                   if "thinking_residual_Lambda" in n],
        "lr": thinking_residual_Lambda,     # e.g., 1e-3 (200x main lr)
        "weight_decay": args.weight_decay,
    },
]
```

**Key Insight:** Lambda parameters need a much higher learning rate (1e-3 vs 5e-6) because they start in a carefully initialized regime and need to move meaningfully to allow hidden-state blending.

The patching is applied via `types.MethodType`, replacing the trainer's `create_optimizer` method while backing up the original.

---

### 3.3 Reward Functions (`utils.py`)

All reward functions follow the same pattern:
1. Extract the answer after `####` from the model response
2. Parse and normalize both predicted and ground-truth answers
3. Check format compliance (exactly one `####` separator)
4. Return 1.0 if both correct AND format-compliant, else 0.0

| Task | Function | Answer Parsing |
|------|----------|---------------|
| GSM8K | `reward_func_math` | `math_verify.parse()` + `math_verify.verify()` |
| MATH | `reward_func_math` | Same as GSM8K |
| MMLU | `reward_func_mmlu` | Extract single letter A-J via `process_mmlu_answer()` |
| RAG | `reward_func_rag` | Normalize text via `process_qa_answer()` (lowercase, remove articles/punctuation) |

---

### 3.4 Task-Specific Default Hyperparameters

Each training script defines task-specific defaults (these are the paper-original values):

| Parameter | GSM8K | MATH | MMLU | RAG |
|-----------|-------|------|------|-----|
| `group_size` | 4 | 8 | 8 | 4 |
| `per_device_train_batch_size` | 8 | 16 | 16 | 16 |
| `gradient_accumulation_steps` | 4 | 4 | 4 | 4 |
| **Effective batch size** | **32** | **64** | **64** | **64** |
| `max_prompt_length` | 1024 | 2048 | 1024 | 2048 |
| `max_completion_length` | 1024 | 2048 | 1024 | 1024 |
| Reward function | `reward_func_math` | `reward_func_math` | `reward_func_mmlu` | `reward_func_rag` |
| Dataset source | HuggingFace `openai/gsm8k` | Local filesystem JSON | `Dataset.load_from_disk()` | `Dataset.load_from_disk()` |
| Dataset root arg | _(none, auto-downloaded)_ | `--dataset_root ../MATH` | `--dataset_root ../MMLU_Train_Merged` | `--dataset_root ../RAG_Train_Merged` |

---

### 3.5 GRPOConfig Common Settings

Shared across all four training scripts:

| Parameter | Value |
|-----------|-------|
| `use_vllm` | `False` |
| `learning_rate` | `5e-6` |
| `beta` | `0.005` |
| `adam_beta1` | `0.9` |
| `adam_beta2` | `0.99` |
| `weight_decay` | `0.1` |
| `warmup_ratio` | `0.1` |
| `lr_scheduler_type` | `"cosine"` |
| `optim` | `"paged_adamw_8bit"` |
| `max_grad_norm` | `0.1` |
| `temperature` | `0.5` |
| `num_train_epochs` | `1` |
| `save_steps` | `250` |
| `save_total_limit` | `3` |
| `logging_steps` | `1` |
| `bf16` | `is_bfloat16_supported()` (auto) |
| `fp16` | `not is_bfloat16_supported()` (auto) |
| `report_to` | `"wandb"` |
| WandB project | `"latent-reasoning"` |
| `seed` | `42` |

---

### 3.6 Experiment Naming Convention

All training scripts generate output directories with this pattern:

```
# HRPO (default)
./experiments/{model_short}-{task}-group{G}-lora{R}-rmin{MIN}-temp{T}

# GRPO baseline (--only_grpo)
./experiments/{model_short}-{task}-grpo-group{G}-lora{R}-temp{T}
```

Examples:
- `./experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.99-temp0.5` (HRPO)
- `./experiments/Qwen2.5-1.5B-Instruct-gsm8k-grpo-group4-lora32-temp0.5` (GRPO)
- `./experiments/Qwen2.5-3B-Instruct-math-group8-lora32-rmin0.99-temp0.5` (HRPO)

GRPO directories use `-grpo-` in the name and omit `-rmin{MIN}` (not applicable). Training scripts check if the experiment directory already exists and contains files -- if so, they exit to prevent accidental overwrites. Checkpoints are saved as `checkpoint-{step}` subdirectories.

---

## 4. Datasets

### 4.1 GSM8K
- **Source:** `openai/gsm8k` (HuggingFace Hub, auto-downloaded)
- **Format:** question -> answer with `####` separator
- **No preprocessing needed** -- loaded directly via `load_dataset`

### 4.2 MATH
- **Source:** `EleutherAI/hendrycks_math` (HuggingFace Hub)
- **Format:** problem + solution with `\boxed{}` answers
- **Preprocessing:** Downloaded and saved as JSON files in folder structure `MATH/train/<subject>/<id>.json`
- **Run:** `--dataset_root /path/to/MATH`

### 4.3 MMLU
- **Source:** `cais/mmlu` (HuggingFace Hub, `auxiliary_train` split, ~99K examples)
- **Format:** question + choices (list) + answer (ClassLabel int)
- **Preprocessing:** Saved via `Dataset.save_to_disk()`. **Important:** answer column is `ClassLabel` type; `process_mmlu` converts int->letter but HF datasets casts back to int. Fix: use `cast_column("answer", Value("int32"))` and `remove_columns` in `.map()` to avoid type conflict.
- **Run:** `--dataset_root /path/to/MMLU_Train_Merged`

### 4.4 RAG (QA)
- **Source:** `rajpurkar/squad` (or any dataset with question + contexts + golden_answers)
- **Format:** question (str), contexts (list[str]), golden_answers (list[str])
- **Preprocessing:** Transform to expected column format and `save_to_disk()`
- **Run:** `--dataset_root /path/to/RAG_Train_Merged`

### 4.5 RAG Evaluation Datasets

Evaluation requires pre-processed datasets saved at `../RAG_Eval/` with subdirectories:

| Subdirectory | Benchmark | `dataset_code` |
|-------------|-----------|-----------------|
| `NQ_Eval/` | Natural Questions | `nq` |
| `TQ_Eval/` | TriviaQA | `tq` |
| `2Wiki_Eval/` | 2WikiMultiHopQA | `2wiki` |
| `HotpotQA_Eval/` | HotpotQA | `hotpotqa` |
| `Bamboogle_Eval/` | Bamboogle | `bamboogle` |

Each must be a HuggingFace `Dataset` saved via `save_to_disk()` with columns: `question`, `contexts`, `golden_answers`.

---

## 5. Evaluation

### 5.1 Common Evaluation Pattern

All eval scripts follow the same flow:

1. Load base model via `FastLanguageModel.from_pretrained()` (no 4-bit, `fast_inference=False`)
2. Set `model.answer_start = ANSWER_START` (`"####"`) — **skipped when `--only_grpo` is passed** (disables thinking residual blending during generation)
3. Load trained LoRA adapter via `model.load_adapter(adapter_path)`
4. Switch to inference mode via `FastLanguageModel.for_inference(model)`
5. Load evaluation dataset
6. Batch-generate with `is_inference` flag (greedy decoding by default)
7. Extract answers from generated text (split on `####`, parse)
8. Compare against ground truth
9. Save metrics + per-sample results as JSON in the checkpoint directory

### 5.2 Eval Scripts Reference

| Script | Dataset | Answer Parsing | Output File | `max_new_tokens` |
|--------|---------|---------------|-------------|------------------|
| `eval_gsm8k.py` | `openai/gsm8k` test (1319 samples) | `math_verify.parse()` + `verify()` | `eval_results.json` | 2048 |
| `eval_math.py` | Local MATH test + `HuggingFaceH4/MATH-500` | `math_verify.parse()` + `verify()` | `eval_results.json` | 2048 |
| `eval_mmlust.py` | `TIGER-Lab/MMLU-STEM` test | `process_mmlu_answer()` (letter A-J) | `eval_results_mmlust.json` | 512 |
| `eval_rag.py` | `../RAG_Eval/` (5 benchmarks) | `process_qa_answer()` (normalized text) | `eval_results_{code}.json` | 512 |
| `eval_arcc.py` | `allenai/ai2_arc` ARC-Challenge test | `process_mmlu_answer()` (letter) | `eval_results_ai2_arc.json` | 512 |

**Notes:**
- `eval_math.py` reports two accuracy metrics: overall MATH accuracy and MATH-500 subset accuracy (checks if problem appears in the MATH-500 dataset).
- `eval_mmlust.py` reports per-subject accuracy breakdown in addition to overall accuracy.
- `eval_rag.py` runs 5 separate evaluations sequentially (NQ, TQ, 2Wiki, HotpotQA, Bamboogle), each producing its own output file.
- `eval_arcc.py` also supports `openbookqa` and `qasc` datasets via code modification.

### 5.3 Eval CLI Arguments

All eval scripts share these arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint_path` | str | None | Path to the trained checkpoint directory (required) |
| `--batch_size` | int | 32 | Evaluation batch size |
| `--no-greedy` | flag | _(off)_ | Disable greedy decoding (default: greedy/`is_inference=True`) |
| `--only_grpo` | flag | _(off)_ | Evaluate a GRPO baseline checkpoint (skip `model.answer_start`, no thinking residual) |

`eval_rag.py` has an additional argument:
- `--eval_examples` (int, default=None): Limit number of evaluation examples per benchmark

**Important:** When evaluating GRPO checkpoints, always pass `--only_grpo`. Without it, the thinking residual modules (which exist in the model architecture but were not trained) would be applied with random weights, corrupting the output.

### 5.4 Automatic Base Model and Temperature Detection

All eval scripts auto-detect configuration from the checkpoint path:

```python
# Auto-detect base model by matching name in path
base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
for model in base_models:
    if model.split('/')[-1] in checkpoint_path:
        base_model = model

# Auto-extract temperature from experiment naming convention
temperature = float(checkpoint_path.split('-temp')[-1].split('/')[0])
```

This works because experiment paths follow the naming convention in Section 3.6.

### 5.5 Running Evaluation Manually

```bash
# GSM8K
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py \
  --checkpoint_path ./experiments/Qwen2.5-3B-Instruct-gsm8k-group4-lora32-rmin0.99-temp0.5/checkpoint-935 \
  --batch_size 128

# MATH
CUDA_VISIBLE_DEVICES=0 python eval_math.py \
  --checkpoint_path ./experiments/Qwen2.5-3B-Instruct-math-group8-lora32-rmin0.99-temp0.5/checkpoint-500 \
  --batch_size 128

# MMLU-STEM
CUDA_VISIBLE_DEVICES=0 python eval_mmlust.py \
  --checkpoint_path ./experiments/Qwen2.5-3B-Instruct-mmlu-group8-lora32-rmin0.99-temp0.5/checkpoint-500 \
  --batch_size 128

# RAG (runs all 5 benchmarks)
CUDA_VISIBLE_DEVICES=0 python eval_rag.py \
  --checkpoint_path ./experiments/Qwen2.5-3B-Instruct-rag-group4-lora32-rmin0.99-temp0.5/checkpoint-500 \
  --batch_size 128

# ARC-Challenge
CUDA_VISIBLE_DEVICES=0 python eval_arcc.py \
  --checkpoint_path ./experiments/Qwen2.5-3B-Instruct-mmlu-group8-lora32-rmin0.99-temp0.5/checkpoint-500 \
  --batch_size 128
```

---

## 6. Unified Pipeline: `run_hrpo_all.sh` & `run_grpo_all.sh`

Two unified scripts orchestrate training and evaluation across all tasks with smart skip logic:

- **`run_hrpo_all.sh`** — HRPO training (with thinking residual)
- **`run_grpo_all.sh`** — GRPO baseline (vanilla GRPO, no thinking residual)

Both scripts accept the same CLI options and share identical hyperparameters (LR, beta, LoRA rank, batch sizes, etc.). The only difference is that `run_grpo_all.sh` passes `--only_grpo` to all training and eval scripts, which disables the thinking-residual-specific components.

### 6.1 CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu ID` | `0` | GPU device ID |
| `--tasks TASKS` | `all` | Comma-separated: `gsm8k,math,mmlu,rag` or `"all"` |
| `--model NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model name |
| `--paper-params` | _(off)_ | Use paper-original batch sizes instead of H200-optimized |
| `--eval-only` | _(off)_ | Skip training, only evaluate existing checkpoints |
| `--skip-eval` | _(off)_ | Skip evaluation after training |
| `--no-wandb` | _(off)_ | Disable WandB logging (`WANDB_DISABLED=true`) |
| `--dry-run` | _(off)_ | Print commands without executing |

### 6.2 H200-Optimized vs. Paper-Original Batch Sizes

The script defaults to H200-optimized settings (maximize BS, set GA=1 to eliminate accumulation overhead). The `--paper-params` flag reverts to paper-original values. **Effective batch size is identical in both modes.**

| Task | H200: BS x GA | Paper: BS x GA | Effective BS | group_size | max_seq_length |
|------|---------------|----------------|--------------|------------|----------------|
| GSM8K | 32 x 1 | 8 x 4 | 32 | 4 | 2048 |
| MATH | 64 x 1 | 16 x 4 | 64 | 8 | 4096 |
| MMLU | 64 x 1 | 16 x 4 | 64 | 8 | 2048 |
| RAG | 64 x 1 | 16 x 4 | 64 | 4 | 3072 |
| Eval | 128 | 32 | -- | -- | -- |

### 6.3 Smart Skip Logic

- **Training:** If `./experiments/{exp_name}/checkpoint-*` already exists, training is skipped for that task.
- **Evaluation:** If `eval_results*.json` already exists in the checkpoint directory, evaluation is skipped for that task.
- This allows safe re-runs -- the script picks up where it left off.

### 6.4 Pipeline Flow

```
1. Parse CLI arguments
2. Activate conda environment ("rot")
3. Validate datasets (check required directories exist)
4. Disable WandB if --no-wandb
5. Training phase (unless --eval-only):
   For each task: train_{task}() -> logs to logs/{task}_train_{timestamp}.log
6. Evaluation phase (unless --skip-eval):
   For each task: eval_{task}() -> logs to logs/{task}_eval_{timestamp}.log
7. Print summary table (checkpoint status + accuracy for each task)
8. Report any failures and exit
```

### 6.5 Usage Examples

```bash
# ---- HRPO (with thinking residual) ----

# Run everything with default settings (H200-optimized)
bash run_hrpo_all.sh

# Train and eval only GSM8K on GPU 1
bash run_hrpo_all.sh --gpu 1 --tasks gsm8k

# Use paper-original batch sizes, no WandB
bash run_hrpo_all.sh --paper-params --no-wandb

# Only evaluate existing checkpoints
bash run_hrpo_all.sh --eval-only --tasks gsm8k,math

# Dry run to see what commands would execute
bash run_hrpo_all.sh --dry-run

# Train with a different model
bash run_hrpo_all.sh --model Qwen/Qwen2.5-3B-Instruct --tasks gsm8k

# ---- GRPO Baseline (no thinking residual) ----

# Run GRPO baseline on all tasks
bash run_grpo_all.sh

# GRPO on GSM8K only
bash run_grpo_all.sh --gpu 1 --tasks gsm8k

# GRPO dry run
bash run_grpo_all.sh --dry-run --tasks gsm8k

# Or run GRPO directly via the training script
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py --only_grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct --group_size 4
```

---

## 7. Reproduction: Bug Fixes and Key Issues

### 7.1 `accelerate` Version Incompatibility

**Problem:** `accelerate>=1.13.0` removed `FP8BackendType` from the exports, but the unsloth-patched `_fast_inner_training_loop` (compiled as `<string>`) references it.

**Error:**
```
NameError: name 'FP8BackendType' is not defined
```

**Fix:** Downgrade accelerate:
```bash
pip install "accelerate==1.5.2"
```

### 7.2 `tokenizers` Version Conflict

**Problem:** The local `transformers/` directory requires `tokenizers>=0.21,<0.22`, but `unsloth` pulls in `tokenizers==0.22.2`.

**Error:**
```
ImportError: tokenizers>=0.21,<0.22 is required
```

**Fix:**
```bash
pip install "tokenizers>=0.21,<0.22"
```

### 7.3 MMLU ClassLabel Type Casting

**Problem:** The MMLU dataset has `answer` column with type `ClassLabel(names=['A','B','C','D'])`. Even though `process_mmlu` returns string answers like `"A"`, HF `datasets` auto-converts them back to int because the column type is ClassLabel.

**Error:**
```
AttributeError: 'int' object has no attribute 'strip'
```

**Fix:** In `hrpo_mmlu.py`, cast the answer column and remove original columns during mapping:

```python
from datasets import Value
dataset = dataset.cast_column("answer", Value("int32"))
processed = dataset.map(process_mmlu, batched=True,
                        batch_size=chunk_size, load_from_cache_file=False,
                        remove_columns=dataset.column_names)
```

### 7.4 MATH Dataset Access (hendrycks/competition_math)

**Problem:** The original `hendrycks/competition_math` dataset is disabled (403 Forbidden) on HuggingFace Hub.

**Fix:** Use the `EleutherAI/hendrycks_math` mirror dataset and reconstruct the expected folder structure.

### 7.5 PyTorch + Unsloth Version Pinning

**Problem:** `unsloth` requires `torch<2.11.0`, so even though CUDA 12.8 has `torch==2.11.0+cu128`, unsloth downgrades to `torch==2.10.0+cu128`.

**Resolution:** This is expected. `torch==2.10.0+cu128` still has full CUDA 12.8 support.

### 7.6 SymPy JSON Serialization in Eval Scripts

**Problem:** `math_verify.parse()` returns SymPy objects (`sympy.Integer`, `sympy.Float`) which are not JSON serializable. When eval scripts (GSM8K, MATH) try to save results via `json.dump`, they crash at the very end after all samples have been processed.

**Error:**
```
TypeError: Object of type Integer is not JSON serializable
```

**Affected files:** `eval_gsm8k.py`, `eval_math.py`

**Fix:** Wrap SymPy objects with `str()` and verify results with `bool()` before storing in the results dict:

```python
result = {
    'question': batch_data['question'][j],
    'true_answer': str(true_answer),
    'generated_answer': str(generated_answer),
    'full_response': response,
    'correct': bool(verify(generated_answer, true_answer)),
}
```

**Note:** `eval_mmlust.py` and `eval_rag.py` are NOT affected -- they use `process_mmlu_answer()` and `process_qa_answer()` which return plain strings.

---

## 8. Environment Setup (Verified Working)

```bash
# Create/use conda environment with Python 3.11
conda activate rot

# 1. Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install unsloth (pulls transformers, peft, accelerate, trl, etc.)
pip install unsloth

# 3. Install remaining dependencies
pip install math-verify wandb

# 4. Fix version conflicts
pip install "accelerate==1.5.2"
pip install "tokenizers>=0.21,<0.22"
```

**Verified versions:**
| Package | Version |
|---------|---------|
| torch | 2.10.0+cu128 |
| unsloth | 2026.3.18 |
| trl | 0.24.0 |
| accelerate | 1.5.2 |
| tokenizers | 0.21.4 |
| datasets | 4.3.0 |
| peft | 0.18.1 |
| bitsandbytes | 0.49.2 |
| math-verify | 0.9.0 |

---

## 9. Supported Models

The codebase supports both Qwen2 and LLaMA model families (architecture modifications exist for both):

| Model | Architecture File |
|-------|-------------------|
| `Qwen/Qwen2.5-1.5B-Instruct` | `transformers/models/qwen2/modeling_qwen2.py` |
| `Qwen/Qwen2.5-3B-Instruct` | `transformers/models/qwen2/modeling_qwen2.py` |
| `meta-llama/Llama-3.2-1B-Instruct` | `transformers/models/llama/modeling_llama.py` |
| `meta-llama/Llama-3.2-3B-Instruct` | `transformers/models/llama/modeling_llama.py` |

Both architecture files contain identical `ThinkingResidualLambda`, `thinking_residual_gate_r/i`, and `thinking_residual()` implementations. The Unsloth fast path (`unsloth/models/llama.py`) handles both families.

**Important limitation:** Eval scripts auto-detect the base model from the checkpoint path. Currently they **only recognize Qwen models** (Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct). If you train with LLaMA and attempt to evaluate, the eval script will fail because `base_model` will be `None`. To add LLaMA support, extend the `base_models` list in each eval script (`eval_gsm8k.py`, `eval_math.py`, `eval_mmlust.py`, `eval_rag.py`, `eval_arcc.py`).

---

## 10. Key Training Metrics to Monitor

From the training logs, monitor these HRPO-specific metrics:

| Metric | Meaning | Expected Range |
|--------|---------|---------------|
| `embeds_ratio` | Average `a_t` value -- how much embedding vs. residual | Starts ~0.997, should decrease over training |
| `hidden_ratio` | Average `sqrt(1-a_t^2)` -- how much hidden state blends in | Starts ~0.074, should increase over training |
| `reward` | Group average reward | Should increase from 0 |
| `completion_length` | Average generated token count | May decrease as model learns conciseness |
| `kl` | KL divergence from reference policy | Should stay moderate |

---

## 11. Running Training (Individual Scripts)

```bash
# GSM8K (simplest -- auto-downloads dataset)
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --group_size 4 \
  --residual_r_min 0.99

# MATH
CUDA_VISIBLE_DEVICES=0 python hrpo_math.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_root ../MATH \
  --group_size 8

# MMLU
CUDA_VISIBLE_DEVICES=0 python hrpo_mmlu.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_root ../MMLU_Train_Merged \
  --group_size 8

# RAG
CUDA_VISIBLE_DEVICES=0 python hrpo_rag.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_root ../RAG_Train_Merged \
  --group_size 4
```

Full argparse arguments (shared by all training scripts):

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `Qwen/Qwen2.5-1.5B-Instruct` | Base model |
| `--lora_rank` | int | 32 | LoRA rank |
| `--lr` | float | 5e-6 | Main learning rate |
| `--beta` | float | 0.005 | GRPO beta |
| `--residual_r_min` | float | 0.99 | Lambda init lower bound |
| `--residual_r_max` | float | 0.999 | Lambda init upper bound |
| `--lr_residual_gate` | float | 1e-4 | LR for gate_r and gate_i |
| `--lr_residual_Lambda` | float | 1e-3 | LR for Lambda |
| `--weight_decay` | float | 0.1 | Weight decay |
| `--warmup_ratio` | float | 0.1 | Warmup ratio |
| `--lr_scheduler_type` | str | `cosine` | LR scheduler |
| `--optimizer` | str | `paged_adamw_8bit` | Optimizer |
| `--max_grad_norm` | float | 0.1 | Gradient clipping |
| `--group_size` | int | 4 or 8 | GRPO group size |
| `--temperature` | float | 0.5 | Sampling temperature |
| `--gradient_accumulation_steps` | int | 4 | GA steps |
| `--per_device_train_batch_size` | int | 8 or 16 | Batch size |
| `--max_prompt_length` | int | 1024 or 2048 | Max prompt tokens |
| `--max_completion_length` | int | 1024 or 2048 | Max completion tokens |
| `--seed` | int | 42 | Random seed |
| `--dataset_root` | str | varies | Dataset path (MATH/MMLU/RAG only) |
| `--only_grpo` | flag | _(off)_ | Run as vanilla GRPO baseline (disables all thinking residual components) |

---

## 12. GRPO Baseline Mode (`--only_grpo`)

All training scripts (`hrpo_*.py`) and eval scripts (`eval_*.py`) support a `--only_grpo` flag that disables the thinking residual mechanism, reducing HRPO to vanilla GRPO. This enables fair ablation comparisons on the same codebase.

### 12.1 What `--only_grpo` Disables

When `--only_grpo` is passed to a training script, the following HRPO-specific components are skipped:

| Component | HRPO (default) | GRPO (`--only_grpo`) |
|-----------|---------------|---------------------|
| `model.answer_start` | Set to `"####"` — enables thinking/answer mode switching during generation | **Not set** — generation runs without thinking residual blending |
| `modules_to_save` | `[thinking_residual_gate_r, gate_i, Lambda]` — these modules are trained and saved | **`None`** — only LoRA adapters are trained/saved |
| `reset_lambda_parameters()` | Lambda initialized with `[r_min, r_max]` | **Skipped** — Lambda stays at random init (irrelevant since not used) |
| `patch_trainer_optimizer()` | 4 param groups with separate LRs for gate and Lambda | **Skipped** — standard optimizer with 2 param groups (decay / no decay) |

### 12.2 Generation Code Guard

The modified `transformers/generation/utils.py` guards the thinking-state computation with:

```python
if getattr(self, 'answer_start', None) is not None:
    # compute is_thinking, last_thinking_states ...
```

When `answer_start` is not set (GRPO mode), `is_thinking` and `last_thinking_states` remain `None` (initialized at the top of the generation loop). They are never passed to the model forward, so no thinking residual blending occurs. This guard is the safety net that prevents crashes and ensures clean GRPO generation.

### 12.3 Running GRPO Baseline

```bash
# Via unified pipeline (recommended)
bash run_grpo_all.sh --tasks gsm8k,math --no-wandb

# Via individual scripts
CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py --only_grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct --group_size 4

# Evaluate GRPO checkpoint (must pass --only_grpo)
CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py --only_grpo \
  --checkpoint_path ./experiments/Qwen2.5-1.5B-Instruct-gsm8k-grpo-group4-lora32-temp0.5/checkpoint-250
```

### 12.4 HRPO vs GRPO Experiment Directories

HRPO and GRPO experiments are saved to different directories, so both can coexist under `./experiments/`:

```
experiments/
|-- Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.99-temp0.5/       # HRPO
|   |-- checkpoint-250/
|   |-- checkpoint-500/
|-- Qwen2.5-1.5B-Instruct-gsm8k-grpo-group4-lora32-temp0.5/           # GRPO
|   |-- checkpoint-250/
|   |-- checkpoint-500/
```

---

## 13. Project File Structure

```
hrpo-trl/
|-- hrpo_gsm8k.py          # Training: GSM8K (auto-downloads from HuggingFace)
|-- hrpo_math.py            # Training: MATH (local filesystem dataset)
|-- hrpo_mmlu.py            # Training: MMLU (pre-saved dataset)
|-- hrpo_rag.py             # Training: RAG/QA (pre-saved dataset)
|-- eval_gsm8k.py           # Eval: GSM8K (1319 test samples)
|-- eval_math.py            # Eval: MATH full test + MATH-500 subset
|-- eval_mmlust.py          # Eval: MMLU-STEM (per-subject breakdown)
|-- eval_rag.py             # Eval: RAG (NQ, TQ, 2Wiki, HotpotQA, Bamboogle)
|-- eval_arcc.py            # Eval: ARC-Challenge (+ OpenBookQA, QASC)
|-- patch.py                # Custom optimizer patching (4 param groups)
|-- utils.py                # Reward functions, answer parsing, data processing
|-- run_hrpo_all.sh         # Unified pipeline: HRPO train + eval all tasks
|-- run_grpo_all.sh         # Unified pipeline: GRPO baseline train + eval all tasks
|-- transformers/           # Modified HF Transformers (ThinkingResidual + generation logic)
|   |-- models/qwen2/modeling_qwen2.py    # ThinkingResidualLambda + gating (Qwen2)
|   |-- models/llama/modeling_llama.py     # ThinkingResidualLambda + gating (LLaMA)
|   |-- generation/utils.py               # is_inference, thinking mask, thinking states
|-- unsloth/                # Modified Unsloth (fast inference with thinking residual)
|   |-- models/llama.py                   # Prefill + decode-time residual blending
|-- trl/                    # Local TRL (GRPO trainer)
|-- experiments/            # Training outputs (checkpoints + eval results)
|-- logs/                   # Training and evaluation logs
```

---

## 14. Summary: What Makes HRPO Different

1. **No explicit CoT required:** HRPO enables latent reasoning without generating reasoning tokens visible to the user
2. **Gated residual connection:** A GRU-inspired gate blends hidden states into embeddings with learnable parameters
3. **Progressive training:** Lambda initialization ensures the model starts with nearly pure embeddings and gradually learns to use hidden states
4. **RL-compatible:** The token sampling in generation introduces stochasticity, making it compatible with GRPO without needing explicit reasoning trajectories
5. **Interpretable:** The `embeds_ratio` / `hidden_ratio` metrics show how much the model relies on latent vs. token reasoning at each step
