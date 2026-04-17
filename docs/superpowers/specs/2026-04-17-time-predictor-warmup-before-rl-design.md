# Time Predictor Warmup Before RL Design

Date: 2026-04-17

## Goal

Before formal RL training begins for time-conditioning modes, run a short predictor-only warmup pass so `thinking_time_predictor` learns a non-trivial rollout signal before the main RL loss starts depending on it.

The warmup must:

- train only `thinking_time_predictor`
- use a seed-stable random subset of the training dataset
- default to `20%` of the dataset, but remain configurable
- make exactly one pass over that subset
- keep the existing reward-weighted auxiliary loss behavior
- leave the subsequent RL phase unchanged and still run on the full dataset

## Confirmed Decisions

The design below locks in the following user-approved choices:

- warmup trains only `thinking_time_predictor`
- warmup samples the subset by shuffling with `seed` and taking a prefix
- warmup fraction defaults to `0.2` and is configurable
- warmup runs for exactly one pass over the sampled subset
- RL still trains on the full dataset after warmup
- warmup keeps the current reward weighting for `thinking_time_aux_loss`
- warmup runs as an explicit independent pass before `trainer.train()`

## Current State

The current codebase already has the pieces needed for predictor supervision during RL replay:

- rollout generation returns `thinking_mask` and `rollout_thinking_time`
- training replay caches predictor inputs in `_train_predictor_hidden_states`
- `compute_thinking_time_aux_loss()` supervises `thinking_time_predictor` against GT thinking-time ramps
- the auxiliary loss currently weights each sample with `compute_thinking_time_aux_weights(rewards)`

However, predictor supervision only happens inside the normal RL training loop. That means the first RL updates start from an untrained predictor and can produce weak or noisy rollout-time estimates.

## Chosen Approach

Add a dedicated predictor warmup phase that runs immediately after trainer construction and immediately before `trainer.train()`.

This phase reuses the existing rollout and replay pipeline instead of inventing a second data path:

1. draw a seed-stable random subset from the already prepared training dataset
2. generate completions with the current model exactly as GRPO rollout already does
3. compute the existing reward-weighted predictor auxiliary loss
4. update only `thinking_time_predictor`
5. exit warmup after one pass through the subset
6. start the normal RL training loop on the full dataset

This keeps the warmup aligned with actual rollout behavior while making the parameter boundary explicit: the warmup teaches the predictor only, and the RL phase remains the first time the policy, KL path, and modulation path are optimized.

## Scope

In scope:

- a configurable predictor warmup fraction
- predictor-only warmup execution before RL
- reuse of the existing reward-weighted auxiliary loss
- seed-stable subset sampling
- tests and logging for warmup behavior

Out of scope:

- warmup for non-time-conditioning modes
- warmup of `reasoning_time_embedding` or `adaln_proj`
- separate warmup checkpoints
- changing formal RL dataset composition
- changing auxiliary reward weighting semantics

## Configuration Surface

Expose one new CLI/config argument in the shared training parser:

- `--time-predictor-warmup-fraction`

Behavior:

- default: `0.2`
- valid range: `0.0 <= fraction <= 1.0`
- `0.0` disables the warmup phase entirely

No separate enable flag is needed. A zero fraction is the off switch.

The flag is parsed for all training entrypoints so the shell wrappers stay uniform, but warmup execution only happens when the selected mode uses time conditioning (`tgrpo` or `thrpo`).

## Dataset Selection

Warmup uses the already prepared RL training dataset, not a separate preprocessing pipeline.

Subset construction:

1. take the prepared dataset passed into `create_training_trainer()`
2. shuffle it with `seed=args.seed`
3. compute `warmup_size = ceil(len(dataset) * fraction)` when `fraction > 0`
4. select the first `warmup_size` rows

Properties:

- deterministic for a fixed dataset and seed
- independent from original dataset ordering
- never mutates the full dataset used by the later RL phase

If `fraction > 0` but the dataset is empty, warmup is skipped with a clear log message. If `fraction == 0`, warmup is skipped silently except for one startup log line noting that it is disabled.

## Warmup Execution Model

Warmup is an independent pass that runs before formal RL starts and after the full `GRPOTrainer` is fully constructed.

The execution order becomes:

1. build dataset
2. create trainer and model exactly as today
3. if eligible, run predictor warmup
4. run `trainer.train(...)` on the full dataset exactly as today

Why trainer construction happens first:

- the trainer already owns the prepared model, tokenizer, accelerator, reward functions, and rollout helper methods
- `GRPOTrainer._prepare_inputs()` already computes rollouts, rewards, `thinking_mask`, and `rollout_thinking_time`
- the warmup can reuse those components without duplicating rollout logic

## Warmup Data Flow

For each warmup batch:

1. use the same raw-example structure as normal GRPO batches
2. call `trainer._prepare_inputs(batch)` to generate rollouts and compute rewards
3. concatenate prompt and completion ids as the replay sequence
4. call `prepare_time_conditioning_training_step(...)` with the batch `thinking_mask` and `rollout_thinking_time`
5. run one replay forward pass to populate `_train_predictor_hidden_states`
6. compute `compute_thinking_time_aux_loss(...)` using the batch rewards
7. backpropagate only that auxiliary loss
8. step a temporary warmup optimizer that only owns predictor parameters
9. clear cached time-conditioning state

There is no policy loss, no KL term, and no RL optimizer step during warmup.

## Parameter Update Boundary

Warmup updates only `thinking_time_predictor`.

Explicitly frozen during warmup:

- base LM parameters
- LoRA parameters
- `reasoning_time_embedding`
- all per-layer `adaln_proj`
- thinking residual modules when present

The implementation must temporarily force `requires_grad=False` for all non-predictor parameters during warmup and restore their original trainability state before RL begins.

This freeze is a correctness guard, not just an optimizer choice. It ensures the warmup phase cannot accidentally leak gradients into other modules if the auxiliary path changes later.

## Optimizer And Scheduling

Warmup uses a temporary predictor-only optimizer built from the same training optimizer family already selected by CLI, but with a parameter list restricted to `thinking_time_predictor` and a learning rate of `args.lr_time_conditioning`.

Deliberate choices:

- no separate warmup LR flag
- no warmup LR scheduler
- no separate checkpoint/state persistence for the warmup optimizer

Rationale:

- `lr_time_conditioning` is already the dedicated LR knob for this module family
- the warmup is exactly one pass, so adding a second scheduler surface is unnecessary
- the warmup optimizer state does not need to survive into formal RL because RL creates and owns its own optimizer state as it already does today

Gradient accumulation uses the same `gradient_accumulation_steps` as formal training so memory behavior stays predictable.

## Reward Weighting

Warmup keeps the current reward-weighted auxiliary loss unchanged.

That means:

- rewards are produced from the same rollout completions already generated in the warmup pass
- `compute_thinking_time_aux_weights(rewards)` continues to define the per-sample weights
- no special warmup-only uniform weighting path is added

This preserves consistency between warmup and formal RL predictor supervision.

## Mode Gating

Warmup runs only when all of the following are true:

- selected mode uses time conditioning
- `time_predictor_warmup_fraction > 0`
- the run is not resuming from an existing RL checkpoint

Expected behavior by mode:

- `tgrpo`: warmup enabled when fraction > 0
- `thrpo`: warmup enabled when fraction > 0
- `grpo`: warmup always skipped
- `hrpo`: warmup always skipped

## Resume Semantics

Resume behavior is intentionally simple:

- if `--resume` resolves to an existing RL checkpoint, skip warmup entirely
- then continue with the normal `trainer.train(resume_from_checkpoint=...)` path

Rationale:

- resumed runs already have a trained or partially trained predictor state in the checkpointed adapter
- replaying warmup on resume would change the meaning of checkpoint continuation
- no warmup-specific checkpoint marker is required

This also means warmup is a start-of-run bootstrap only, not a persistent multi-stage training regime.

## Logging And Metrics

Warmup should produce explicit logs so its behavior is observable and debuggable.

Required startup logs:

- whether warmup is enabled or skipped
- warmup fraction
- sampled warmup subset size
- whether the run is skipping warmup because of resume or mode

Required per-step or aggregated warmup metrics:

- `warmup/thinking_time_aux_loss`
- `warmup/thinking_time_pred_error`
- `warmup/thinking_time_rollout_error` when available
- `warmup/reward`
- `warmup/step`

The metric names are prefixed with `warmup/` so they do not collide with formal RL metrics.

## Proposed File Boundaries

Keep tensor-level time-conditioning logic in `time_conditioning.py` and place warmup orchestration in a new focused module.

New module:

- `time_predictor_warmup.py`

Responsibilities:

- warmup subset sampling
- predictor-only parameter freezing/restoration
- temporary predictor-only optimizer construction
- the independent warmup loop that reuses `GRPOTrainer`

Existing file changes:

- `utils.py`
  - add CLI argument
  - expose a shared helper that task entrypoints call before `trainer.train(...)`
- task entry scripts (`hrpo_gsm8k.py`, `hrpo_math.py`, `hrpo_mmlu.py`, `hrpo_rag.py`)
  - call the shared warmup entrypoint after trainer creation and before `trainer.train(...)`
- `run_grpo_all.sh` and `run_hrpo_all.sh`
  - expose the new flag through shell wrappers for `tgrpo` and `thrpo`

This keeps trainer internals minimally disturbed while avoiding a large `utils.py` grab bag.

## Error Handling

The implementation must fail clearly for:

- `time_predictor_warmup_fraction < 0` or `> 1`
- missing `thinking_time_predictor` on a supposedly time-conditioned model
- missing `thinking_mask` or replay caches in a warmup batch that is expected to contain them
- non-finite warmup loss

Recoverable situations should skip warmup with logs instead of raising:

- fraction is `0`
- dataset length is `0`
- selected mode does not use time conditioning
- run is resuming from an existing checkpoint

## Testing

Add focused tests for:

- parser accepts `--time-predictor-warmup-fraction`
- invalid warmup fractions are rejected
- warmup subset sampling is deterministic for a fixed seed
- warmup subset size uses `ceil(len(dataset) * fraction)`
- warmup is skipped for `grpo` and `hrpo`
- warmup is skipped on resume
- warmup only exposes predictor parameters to the temporary optimizer
- warmup uses the existing reward-weighted auxiliary loss path
- warmup leaves the full dataset untouched for formal RL
- shell wrappers pass the new CLI flag for time-conditioning modes

Where practical, tests should mock rollout execution rather than depending on expensive generation.

## Risks And Tradeoffs

- Because reward weighting is retained, very poor initial rollouts can still reduce warmup effectiveness. This is an accepted tradeoff because preserving supervision semantics was explicitly chosen.
- Reusing `trainer._prepare_inputs()` keeps alignment high but couples warmup to internal trainer behavior. This is acceptable because the alternative would duplicate rollout and reward code paths.
- Skipping warmup on resume means resumed runs cannot re-bootstrap the predictor. This is intentional to preserve checkpoint continuation semantics.

## Success Criteria

The feature is complete when:

- a fresh `tgrpo` or `thrpo` run performs a single predictor-only warmup pass before formal RL
- warmup consumes a deterministic random subset defined by `seed` and `time_predictor_warmup_fraction`
- only `thinking_time_predictor` is updated during warmup
- formal RL still trains on the full dataset
- resumed runs skip warmup
- tests cover parser, subset selection, mode gating, and predictor-only update boundaries
