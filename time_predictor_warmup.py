"""Predictor-only warmup helpers for time-conditioning training modes.

Warmup reuses the standard GRPO trainer path on a deterministic subset, freezes
all non-predictor parameters, disables checkpoint/report side effects, and is
skipped for non-time-conditioned modes or resumed runs.
"""

import copy
import gc
import math
import os
from contextlib import contextmanager

import torch
from transformers import TrainerCallback

from time_conditioning import (
    get_time_conditioning_base_model,
    has_time_conditioning,
    reset_time_conditioning_state,
)


def validate_time_predictor_warmup_fraction(value):
    """Normalize and validate the configured dataset fraction for warmup."""
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"time_predictor_warmup_fraction must be a float, got {value!r}"
        ) from exc
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(
            "time_predictor_warmup_fraction must be between 0.0 and 1.0 inclusive, "
            f"got {fraction}"
        )
    return fraction


def build_time_predictor_warmup_dataset(dataset, fraction, seed):
    """Return a deterministic shuffled subset sized with ceil(len(dataset) * fraction)."""
    fraction = validate_time_predictor_warmup_fraction(fraction)
    if fraction == 0.0 or len(dataset) == 0:
        return None, 0

    warmup_size = math.ceil(len(dataset) * fraction)
    shuffled = dataset.shuffle(seed=int(seed))
    return shuffled.select(range(warmup_size)), warmup_size


def collect_time_predictor_parameters(model):
    """Collect predictor parameters from the wrapped training model."""
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "thinking_time_predictor" in name
    ]


@contextmanager
def predictor_only_grad_scope(model):
    """Temporarily freeze every parameter except the time predictor."""
    original_states = []
    for name, parameter in model.named_parameters():
        original_states.append((parameter, parameter.requires_grad))
        parameter.requires_grad_("thinking_time_predictor" in name)
    try:
        yield
    finally:
        for parameter, requires_grad in original_states:
            parameter.requires_grad_(requires_grad)


class _PredictorWarmupLogCallback(TrainerCallback):
    """Forward warmup metrics from the standard trainer log stream."""

    def __init__(self, task, log_metrics_fn):
        self.task = task
        self.log_metrics_fn = log_metrics_fn

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.log_metrics_fn is None or not logs:
            return

        warmup_metrics = {}
        for source_name, target_name in (
            ("reward", "warmup/reward"),
            ("thinking_time_aux_loss", "warmup/thinking_time_aux_loss"),
            ("thinking_time_pred_error", "warmup/thinking_time_pred_error"),
            ("thinking_time_rollout_error", "warmup/thinking_time_rollout_error"),
        ):
            if source_name in logs:
                warmup_metrics[target_name] = logs[source_name]
        if not warmup_metrics:
            return

        warmup_metrics["warmup/step"] = float(getattr(state, "global_step", 0))
        self.log_metrics_fn(self.task, getattr(state, "global_step", 0), warmup_metrics)


def build_time_predictor_warmup_training_args(trainer):
    """Clone trainer args for a one-pass warmup run with isolated side effects."""
    warmup_args = copy.deepcopy(trainer.args)
    warmup_output_dir = os.path.join(trainer.args.output_dir, "predictor_warmup")
    warmup_args.output_dir = warmup_output_dir
    warmup_args.run_name = warmup_output_dir
    warmup_args.report_to = "none"
    warmup_args.save_strategy = "no"
    warmup_args.max_steps = -1
    warmup_args.num_train_epochs = 1
    return warmup_args


def build_time_predictor_warmup_trainer(
    trainer,
    warmup_dataset,
    *,
    args,
    mode,
    task,
    log_metrics_fn=None,
):
    """Build a temporary standard trainer configured for predictor-only warmup."""
    from patch import patch_trainer_optimizer

    warmup_trainer = trainer.__class__(
        model=trainer.model,
        processing_class=trainer.processing_class,
        reward_funcs=trainer.reward_funcs,
        args=build_time_predictor_warmup_training_args(trainer),
        train_dataset=warmup_dataset,
    )
    warmup_trainer.thinking_time_loss_weight = 1.0
    patch_trainer_optimizer(
        warmup_trainer,
        lr_thinking_residual_gate=(
            getattr(args, "lr_residual_gate", None) if mode in {"hrpo", "thrpo"} else None
        ),
        thinking_residual_Lambda=(
            getattr(args, "lr_residual_Lambda", None) if mode in {"hrpo", "thrpo"} else None
        ),
        lr_time_conditioning=getattr(args, "lr_time_conditioning", None),
    )
    if log_metrics_fn is not None:
        warmup_trainer.add_callback(_PredictorWarmupLogCallback(task, log_metrics_fn))
    return warmup_trainer


def run_time_predictor_warmup(trainer, warmup_dataset, *, args, mode, task, log_metrics_fn=None):
    """Run predictor warmup through the standard trainer path with frozen non-predictor params."""
    if not collect_time_predictor_parameters(trainer.model):
        raise ValueError("time predictor warmup requires thinking_time_predictor parameters")

    warmup_trainer = build_time_predictor_warmup_trainer(
        trainer,
        warmup_dataset,
        args=args,
        mode=mode,
        task=task,
        log_metrics_fn=log_metrics_fn,
    )
    unwrapped_model = warmup_trainer.accelerator.unwrap_model(warmup_trainer.model)
    base_model = get_time_conditioning_base_model(unwrapped_model)
    try:
        with predictor_only_grad_scope(unwrapped_model):
            warmup_trainer.train()
    finally:
        reset_time_conditioning_state(base_model)
        del warmup_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def maybe_run_time_predictor_warmup(
    trainer,
    *,
    args,
    mode,
    resume_from_checkpoint,
    task,
    log_metrics_fn=None,
):
    """Gate and launch predictor warmup for fresh TGRPO/THRPO runs."""
    fraction = validate_time_predictor_warmup_fraction(
        getattr(args, "time_predictor_warmup_fraction", 0.0)
    )
    if mode not in {"tgrpo", "thrpo"} or fraction == 0.0 or resume_from_checkpoint is not None:
        return

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    base_model = get_time_conditioning_base_model(unwrapped_model)
    if not has_time_conditioning(base_model):
        raise ValueError("time predictor warmup expected a time-conditioned model")

    warmup_dataset, warmup_size = build_time_predictor_warmup_dataset(
        trainer.train_dataset,
        fraction=fraction,
        seed=getattr(args, "seed", 0),
    )
    if warmup_dataset is None or warmup_size == 0:
        return

    run_time_predictor_warmup(
        trainer,
        warmup_dataset,
        args=args,
        mode=mode,
        task=task,
        log_metrics_fn=log_metrics_fn,
    )
