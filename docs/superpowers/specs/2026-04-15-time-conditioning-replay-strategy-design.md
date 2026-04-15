# Time-Conditioning Replay Strategy Design

Date: 2026-04-15

## Goal

Add an explicit training-time replay strategy switch for time conditioning instead of hard-coding rollout self-prediction as the only source of replayed thinking time.

The new design must support three strategies:

- `self`: keep current behavior and use detached rollout predictor outputs.
- `gt_noise`: use `gt_thinking_time + U(-eps, eps)`, then clamp to `[0, 1]`.
- `mix`: use `alpha * rollout_thinking_time + (1 - alpha) * noisy_gt`, then clamp to `[0, 1]`.

The design must preserve the current inference path. Rollout and inference continue to use online predictor outputs only. The strategy switch applies only to the training-time replay signal that feeds `reasoning_time_embedding` and downstream `adaln_proj` modulation.

## Problem Statement

The current implementation uses rollout-time self-predicted thinking time as the only replay source for training-time AdaLN modulation. This keeps training aligned with inference, but it also means early predictor noise is injected directly into the RL main loss path.

We want a controlled way to decouple or partially decouple the replayed training signal from rollout self-predictions:

- `gt_noise` provides a stable near-GT replay target for the modulation path.
- `mix` provides a curriculum that starts with GT-like replay and gradually returns to rollout-aligned replay.

This change is explicitly not a change to the auxiliary predictor supervision. `compute_thinking_time_aux_loss()` continues to supervise the predictor against GT targets using rollout-consistent hidden traces.

## Constraints

- Inference behavior must remain unchanged.
- Existing `self` behavior must remain the default so current experiments are preserved unless the new flags are enabled.
- `eps` must match the dead-zone margin by default. Current value is `0.01`.
- `mix` alpha schedule is fixed for now:
  - first quarter of training: `alpha = 0`
  - second and third quarter: linearly increase `alpha` from `0` to `1`
  - last quarter: `alpha = 1`
- The strategy must operate on normalized, position-aligned `(B, L)` tensors and keep masking semantics unchanged.
- The replay signal must stay detached from rollout tensors before entering the RL main path.

## Proposed Design

### Configuration Surface

Expose explicit config attributes on the base model config when time conditioning is enabled:

- `thinking_time_replay_strategy`
- `thinking_time_replay_noise_eps`
- `thinking_time_replay_mix_warmup_start_fraction`
- `thinking_time_replay_mix_warmup_end_fraction`

Initial values:

- `thinking_time_replay_strategy = "self"`
- `thinking_time_replay_noise_eps = 0.01`
- `thinking_time_replay_mix_warmup_start_fraction = 0.25`
- `thinking_time_replay_mix_warmup_end_fraction = 0.75`

The last two fractions define the linear ramp region for `mix`.

Trainer-side scheduling state must be passed explicitly into the replay selection helper so the helper can compute `alpha` from training progress without implicitly reading mutable global state.

### Training-Time Replay Selection

Extend `select_training_thinking_time_embedding()` so it no longer assumes rollout self-time is the only valid replay source.

Inputs remain:

- `gt_thinking_time`
- `rollout_thinking_time`
- `thinking_mask`

New inputs:

- `replay_strategy`
- `noise_eps`
- `training_progress`

Selection rules:

1. `self`

Use the current implementation:

- validate `rollout_thinking_time`
- detach and clamp rollout values
- use rollout values directly as `selected_thinking_time`

2. `gt_noise`

- sample `noise ~ U(-eps, eps)` with the same shape and device as `gt_thinking_time`
- compute `noisy_gt = clamp(gt_thinking_time + noise, 0, 1)`
- zero masked positions after masking logic is applied
- use `noisy_gt` as `selected_thinking_time`

3. `mix`

- require valid `rollout_thinking_time`
- build `noisy_gt` exactly as in `gt_noise`
- compute `alpha` from `training_progress`
- compute `selected_thinking_time = clamp(alpha * rollout + (1 - alpha) * noisy_gt, 0, 1)`

Masking remains the last step:

- non-thinking positions are forced to zero
- the selected tensor is then embedded with `reasoning_time_embedding`

### Alpha Schedule

`training_progress` is a scalar in `[0, 1]`.

Define:

- `start = 0.25`
- `end = 0.75`

Then:

- if `progress <= start`, `alpha = 0`
- if `start < progress < end`, `alpha = (progress - start) / (end - start)`
- if `progress >= end`, `alpha = 1`

This gives:

- quarter 1: pure noisy GT replay
- quarter 2 and 3: gradual interpolation toward rollout replay
- quarter 4: pure rollout replay

### Metrics

Keep current source metrics and extend them so runs are diagnosable:

- `thinking_time_replay_strategy_self`
- `thinking_time_replay_strategy_gt_noise`
- `thinking_time_replay_strategy_mix`
- `thinking_time_replay_alpha`
- `thinking_time_replay_noise_eps`

Retain `thinking_time_rollout_error` when rollout is available. For `gt_noise`, rollout error is still useful if rollout exists, but should not be required. When rollout is absent and not needed by the active strategy, report availability as `0`.

### Trainer Integration

`GRPOTrainer.compute_loss()` already resets time-conditioning state and calls `prepare_trainer_time_conditioning()`. Extend this flow so trainer progress is computed once and passed into the replay-preparation path.

Progress source:

- use the current global optimizer step relative to planned max steps
- clamp the resulting scalar to `[0, 1]`

The scheduling input must be deterministic within a run. If trainer progress cannot be resolved to a scalar in `[0, 1]`, then:

- `self` and `gt_noise` may proceed because they do not depend on `alpha`
- `mix` must raise a clear `ValueError` instead of silently falling back to `alpha = 1`

The same integration must be mirrored in the Unsloth replacement path so both trainer implementations preserve identical replay behavior.

### Error Handling

- `self` requires `rollout_thinking_time`; missing rollout remains an error.
- `gt_noise` does not require `rollout_thinking_time`.
- `mix` requires `rollout_thinking_time`; missing rollout is an error.
- invalid strategy values raise a clear `ValueError`.
- negative `eps` values raise a clear `ValueError`.
- invalid fraction ranges such as `start >= end`, `start < 0`, or `end > 1` raise a clear `ValueError`.
- non-finite replay inputs remain an error.

### Testing

Add or update focused tests for:

- `self` preserves current behavior.
- `gt_noise` accepts missing rollout and returns a clamped tensor with masked positions zeroed.
- `mix` interpolates correctly at progress `0.0`, `0.25`, `0.5`, `0.75`, and `1.0`.
- `mix` and `self` reject missing rollout.
- stats reflect the active strategy and correct alpha.
- cached training tensors are still populated with the expected shape.
- invalid strategy, invalid `eps`, and invalid fraction ranges raise clear errors.

Tests should avoid brittle exact-noise assertions by constraining value ranges or mocking the random sample when exact interpolation checks are needed.

### Risks And Tradeoffs

- `gt_noise` intentionally introduces a train-inference mismatch. This is acceptable only when explicitly requested and must remain opt-in.
- `mix` reduces this mismatch by converging back to rollout replay, but it adds scheduler complexity and one more trainer-to-helper dependency.
- If progress computation differs between trainer implementations, results can drift. The scheduling calculation therefore needs to be centralized or replicated carefully.

### Scope

In scope:

- training-time replay strategy switch
- noise epsilon configuration
- mix alpha curriculum
- metrics and tests

Out of scope:

- changes to online rollout/inference predictor behavior
- changes to auxiliary predictor supervision
- changes to the GT target definition
- changes to dead-zone loss behavior beyond reusing the same default epsilon
