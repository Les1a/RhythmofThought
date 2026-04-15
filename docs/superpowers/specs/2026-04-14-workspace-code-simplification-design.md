# Workspace Code Simplification Design

## Goal

Simplify all newly added code in the main workspace while preserving the current external usage surface. The result should reduce indirection, duplication, stale notes, and low-value abstractions without changing how the existing training, evaluation, and shell entrypoints are invoked.

## Scope

This design covers the code currently added or modified in the main workspace relative to `origin/main`, especially:

- `time_conditioning.py`
- `transformers/generation/utils.py`
- `transformers/models/llama/modeling_llama.py`
- `transformers/models/qwen2/modeling_qwen2.py`
- `trl/trainer/grpo_trainer.py`
- `unsloth/models/llama.py`
- `unsloth/models/rl_replacements.py`
- `patch.py`
- `utils.py`
- `hrpo_gsm8k.py`
- `hrpo_math.py`
- `hrpo_mmlu.py`
- `hrpo_rag.py`
- `eval_arcc.py`
- `eval_gsm8k.py`
- `eval_math.py`
- `eval_mmlust.py`
- `eval_rag.py`
- `run_grpo_all.sh`
- `run_hrpo_all.sh`
- `run_thrpo_all.sh`
- `srun.sh`
- tests and newly added supporting notes

## Hard Constraints

The simplification must follow these rules:

1. Preserve current external entrypoints and usage shape.
2. Do not delete `run_thrpo_all.sh`.
3. Do not delete the current `run_grpo_all.sh --time-cond` style usage.
4. Do not force `generate(..., return_thinking_embeds=True)` callers onto an incompatible external return contract.
5. Do not delete `srun.sh`.
6. If a proposed deletion could change training or evaluation behavior rather than just structure or readability, stop and ask the user before applying it.

## Non-Goals

- No redesign of the training modes or experiment naming scheme.
- No workflow change that forces the user onto new CLI entrypoints.
- No speculative algorithm change under the label of cleanup.
- No broad refactor of unrelated legacy code outside the newly added surface.

## Current Problems

### 1. Time-conditioning logic is over-layered

`time_conditioning.py` currently mixes:

- model components
- rollout-time state transitions
- training-time embedding selection
- auxiliary loss logic
- config resolution helpers
- runtime state cache management
- adapter detection and module installation

The file works as a feature hub, but too many helpers exist only to wrap one read or one call site. This increases navigation cost without improving separation.

### 2. Trainer-side wiring is duplicated

`trl/trainer/grpo_trainer.py` and `unsloth/models/rl_replacements.py` both carry nearly identical logic for:

- extracting time-conditioning tensors
- validating shapes
- preparing training-time embeddings
- running auxiliary loss
- logging related metrics

This duplication makes it easy for the two paths to drift.

### 3. Model/generation integration is harder to read than needed

The time-conditioning path across generation and model forward files contains repeated state explanations, repeated condition checks, and return-path handling that obscures the actual control flow.

### 4. Script layer contains large copy-pasted sections

The `hrpo_*`, `eval_*`, and `run_*.sh` scripts repeat the same setup and orchestration patterns across tasks. The current code keeps task-specific defaults together with large amounts of shared plumbing.

### 5. Notes and support files contain stale or low-value text

Some comments, docstrings, and notes are descriptive without constraining behavior. Some generated cache files under `test/__pycache__` should not remain in the workspace.

## Design Principles

### Keep entrypoints stable, move complexity inward

The user-facing filenames, flags, and common command invocations stay intact. Internal logic is simplified by extracting shared behavior and deleting low-value wrappers, not by removing familiar entrypoints.

### Prefer fewer layers over thinner wrappers

When a helper:

- is called once
- only forwards parameters
- only reads one config field
- or does not protect an invariant

it should be inlined or merged into a more meaningful function.

### Keep semantics close to where they matter

- Pure thinking-time math belongs in `time_conditioning.py`.
- Trainer orchestration should call helpers, not rebuild them.
- Model files should only decide whether modulation is present, then apply it or bypass it.
- Task scripts should primarily describe task differences.

### Delete explanation that does not constrain behavior

Docstrings and notes should stay only when they define invariants, integration contracts, or easy-to-miss assumptions. Explanatory prose that merely narrates obvious code should be removed.

## Proposed Structure

### A. Simplify `time_conditioning.py` into a small set of responsibility blocks

The file should remain the canonical home for the feature, but its contents should be grouped and shortened.

Keep these blocks:

1. Core modules
   - `ThinkingTimePredictor`
   - `ReasoningTimeEmbedding`
   - `AdaLNProjection`

2. Pure training math
   - thinking-time target construction
   - auxiliary weight construction
   - auxiliary loss computation

3. Rollout-time helpers
   - online conditioning preparation
   - hidden-cache update

4. Runtime integration helpers
   - base-model resolution
   - state reset
   - enable/detect hooks

5. Minimal mask utilities

Reduce or remove:

- one-off config resolver helpers where direct reads are clearer
- wrappers that exist only to rename a single downstream call
- repetitive comments that restate the code
- stats packaging that does not affect behavior or downstream decisions

### B. Move repeated trainer wiring behind unified helpers

Create a small helper surface in `time_conditioning.py` for trainer-side usage so both trainer implementations can follow the same flow:

1. prepare training-time cache from `thinking_mask` and rollout tensors
2. compute optional auxiliary loss from rollout trace inputs
3. return metrics in one place

Then reduce `trl/trainer/grpo_trainer.py` and `unsloth/models/rl_replacements.py` to:

- fetch inputs
- call the shared helper(s)
- set model cache
- add aux loss
- log metrics
- clear state

This keeps the two trainer paths behaviorally aligned without keeping duplicate implementations.

### C. Keep generation/model integration thin

`transformers/generation/utils.py` should keep only the minimum logic required to:

- reset time-conditioning state
- let generation produce rollout-time outputs
- pass back the existing expected tuple shape for callers that use `return_thinking_embeds=True`

The model forward files should only perform:

- detect conditioning tensors
- if present, compute modulation
- otherwise run the normal path

Repeated local narration of time-conditioning semantics should be removed from these integration files.

### D. Convert training and eval scripts into thin task entrypoints

Keep each task file and CLI surface, but move shared logic into reusable helpers.

For training scripts, shared logic should cover:

- mode resolution
- experiment naming
- resume/skip checks
- base model and tokenizer setup
- PEFT wrapping
- trainer construction
- common optimizer patching
- common time-conditioning toggles

Each `hrpo_<task>.py` file should mainly contain:

- dataset loading and preprocessing
- task-specific defaults
- task-specific reward selection

For evaluation scripts, shared logic should cover:

- base model detection
- adapter loading
- mode toggles
- generation loop scaffolding
- result saving

Each `eval_<task>.py` file should mainly contain:

- dataset-specific loading
- answer extraction/parsing differences
- task-specific scoring

### E. Simplify shell scripts without changing entrypoints

Keep `run_grpo_all.sh`, `run_hrpo_all.sh`, and `run_thrpo_all.sh`, but reduce duplication by:

- introducing short shared utility functions inside each script where repetition is excessive
- normalizing argument assembly
- avoiding repeated resume/skip/eval boilerplate blocks where a function can express them directly

This is an internal cleanup only. The scripts keep their current names and command-line behavior.

### F. Remove low-value additions

Safe cleanup includes:

- deleting `test/__pycache__`
- removing stale or redundant notes that do not define invariants
- deleting docstrings/comments that merely narrate obvious code
- shrinking long prose comments into short invariant-focused comments where needed

If any note-like file appears to capture an active design decision rather than redundant explanation, pause and confirm before deleting it.

## Behavioral Safety Rules

The following changes are considered structure-preserving and can proceed without extra approval:

- inlining one-use helpers
- consolidating duplicated code into shared functions
- removing redundant docstrings and comments
- removing generated cache files
- shortening metrics plumbing when metric names and behavior stay the same
- simplifying optimizer group construction without changing parameter assignment

The following changes require user confirmation before execution:

- removing any branch that may alter training-time conditioning behavior
- deleting any rollout/trust or predictor logic whose effect on learning dynamics is uncertain
- removing any script path that may still be intentionally available despite overlap
- changing returned tensor semantics even if the external call signature stays similar

## Testing and Verification Requirements

After implementation, verify at least:

1. relevant unit tests for `time_conditioning.py`
2. unit tests for training-mode utilities and script helpers
3. `py_compile` on touched Python files
4. at least one parameter-level smoke check for the training script path

The cleanup is successful when:

- the same entrypoints remain available
- the time-conditioning path reads linearly instead of as a state puzzle
- trainer and unsloth paths no longer duplicate core logic
- task scripts are visibly thinner
- comments and notes describe only real constraints

## File-Level Change Plan

### `time_conditioning.py`

- Reorder into clear sections
- Inline low-value wrappers
- Remove dead or purely explanatory text
- Keep only invariant-bearing comments
- Centralize shared training-time helper logic

### `trl/trainer/grpo_trainer.py`

- Replace local duplicated prep logic with shared helper calls
- Keep only trainer-specific orchestration

### `unsloth/models/rl_replacements.py`

- Mirror the same simplification as GRPO trainer
- Keep wrapper-specific control flow only

### `transformers/generation/utils.py`

- Shorten the time-conditioning branch
- Preserve current return compatibility

### `transformers/models/llama/modeling_llama.py`
### `transformers/models/qwen2/modeling_qwen2.py`
### `unsloth/models/llama.py`

- Keep modulation injection minimal and local
- Remove repeated explanatory noise

### `patch.py`

- Simplify optimizer group construction
- Keep parameter grouping behavior intact

### `utils.py`

- Keep shared mode and naming helpers
- Move or remove anything that is not truly cross-task

### `hrpo_*.py`

- Convert to thin task entrypoints over shared training scaffolding

### `eval_*.py`

- Convert to thin task entrypoints over shared evaluation scaffolding

### `run_*.sh`

- Reduce duplication while preserving current commands and flags

### `test/`

- Remove generated cache files
- Keep or add tests that lock in behavior for the shared helpers introduced by the simplification

## Risks

### Risk 1: cleanup accidentally changes training behavior

Mitigation:

- treat algorithm-affecting paths as confirmation-gated
- prefer refactors that preserve outputs before deleting anything behavior-adjacent

### Risk 2: trainer and unsloth paths drift during cleanup

Mitigation:

- move shared logic behind common helpers first
- keep duplicated local logic only when wrapper constraints truly differ

### Risk 3: task scripts remain too similar after partial cleanup

Mitigation:

- extract common setup early
- leave task files responsible only for dataset and task-specific configuration

## Decision Summary

The recommended implementation is a structural simplification, not a workflow redesign:

- preserve external usage
- collapse low-value abstraction layers
- unify duplicated trainer wiring
- thin out task scripts
- remove stale textual noise
- confirm separately before deleting any path that might change learning behavior
