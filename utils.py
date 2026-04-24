"""Shared CLI, mode, metadata, reward, and answer-normalization helpers.

Top-level training scripts stay task-specific for dataset loading and rewards,
while this module owns the common GRPO-family contract: explicit mode parsing,
experiment naming, resume resolution, adapter metadata, model restoration, and
the parser options shared by direct Python entrypoints and shell launchers.
"""

import argparse
import json
import os
import re
import string
import types
from math_verify import parse, verify

from time_predictor_warmup import (
    maybe_run_time_predictor_warmup,
    validate_time_predictor_warmup_fraction,
)


ANSWER_START = "####"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_METADATA_KEY = "rot_metadata"
ADAPTER_METADATA_SCHEMA_VERSION = 2
TRAINING_MODES = frozenset({"grpo", "tgrpo", "thrpo", "hrpo"})


def normalize_exp_suffix(suffix: str | None) -> str:
    """Sanitize an optional experiment suffix for safe directory names."""
    if suffix is None:
        return ""
    suffix = suffix.strip()
    if not suffix:
        return ""
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix)
    return suffix.strip("-.")


def is_wandb_disabled() -> bool:
    """Return whether WandB logging is disabled through environment variables."""
    return (
        os.environ.get("WANDB_DISABLED", "").lower() == "true"
        or os.environ.get("WANDB_MODE", "").lower() == "disabled"
    )


def _require_training_mode(mode: str) -> str:
    """Validate a training mode string and return the normalized value."""
    if mode not in TRAINING_MODES:
        raise ValueError(f"unknown training mode: {mode}")
    return mode


def mode_uses_time_conditioning(mode: str) -> bool:
    """Return whether a mode enables the time-conditioning stack."""
    return _require_training_mode(mode) in {"tgrpo", "thrpo"}


def mode_uses_thinking_residual(mode: str) -> bool:
    """Return whether a mode enables the hybrid residual reasoning path."""
    return _require_training_mode(mode) in {"hrpo", "thrpo"}


def _coerce_positive_int(value, *, field_name: str) -> int:
    """Convert a user-facing numeric value into a validated positive integer."""
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive, got {normalized}")
    return normalized


class _StoreWithExplicitFlag(argparse.Action):
    """Argparse action that records whether a value was set explicitly."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def build_training_exp_name(
    model_name,
    task,
    mode,
    group_size,
    lora_rank,
    temperature,
    residual_r_min=None,
    thinking_time_predictor_num_hidden_states=None,
    exp_suffix="",
):
    """Build the canonical experiment directory name for a task, mode, and runtime knobs."""
    mode = _require_training_mode(mode)

    parts = [
        model_name.split("/")[-1],
        task,
        mode,
        f"group{group_size}",
        f"lora{lora_rank}",
    ]
    if mode in {"thrpo", "hrpo"}:
        if residual_r_min is None:
            raise ValueError(f"residual_r_min is required for mode={mode}")
        parts.append(f"rmin{residual_r_min}")
    parts.append(f"temp{temperature}")
    if mode_uses_time_conditioning(mode):
        if thinking_time_predictor_num_hidden_states is None:
            raise ValueError(
                f"thinking_time_predictor_num_hidden_states is required for mode={mode}"
            )
        predictor_hidden_states = _coerce_positive_int(
            thinking_time_predictor_num_hidden_states,
            field_name="thinking_time_predictor_num_hidden_states",
        )
        parts.append(f"last{predictor_hidden_states}")

    exp_suffix = normalize_exp_suffix(exp_suffix)
    if exp_suffix:
        parts.append(exp_suffix)

    return "./experiments/" + "-".join(parts)


def resolve_resume_from_checkpoint(exp_name: str, resume: bool) -> str | None:
    """Resolve the latest checkpoint path when `--resume` is requested."""
    if resume:
        if not os.path.exists(exp_name):
            raise ValueError(f"--resume specified but {exp_name} does not exist.")
        from transformers.trainer_utils import get_last_checkpoint

        checkpoint = get_last_checkpoint(exp_name)
        if checkpoint is None:
            raise ValueError(f"--resume specified but no checkpoint-* dirs found in {exp_name}.")
        return checkpoint

    if os.path.exists(exp_name) and os.listdir(exp_name):
        raise ValueError(f"Experiment {exp_name} already exists.")
    return None


def build_adapter_metadata(args, task: str, mode: str) -> dict:
    """Serialize the runtime configuration that eval and resume treat as authoritative."""
    mode = _require_training_mode(mode)
    use_time_conditioning = mode_uses_time_conditioning(mode)
    use_thinking_residual = mode_uses_thinking_residual(mode)
    return {
        "schema_version": ADAPTER_METADATA_SCHEMA_VERSION,
        "task": task,
        "mode": mode,
        "base_model": args.model_name,
        "temperature": float(args.temperature),
        "group_size": int(args.group_size),
        "lora_rank": int(args.lora_rank),
        "max_prompt_length": int(args.max_prompt_length),
        "max_completion_length": int(args.max_completion_length),
        "use_thinking_residual": use_thinking_residual,
        "use_time_conditioning": use_time_conditioning,
        "residual_r_min": float(args.residual_r_min),
        "residual_r_max": float(args.residual_r_max),
        "thinking_time_loss_weight": float(args.thinking_time_loss_weight) if use_time_conditioning else 0.0,
        "thinking_time_predictor_num_hidden_states": (
            _coerce_positive_int(
                args.thinking_time_predictor_num_hidden_states,
                field_name="thinking_time_predictor_num_hidden_states",
            )
            if use_time_conditioning
            else None
        ),
        "lr": float(args.lr),
        "lr_residual_gate": float(args.lr_residual_gate) if use_thinking_residual else None,
        "lr_residual_lambda": float(args.lr_residual_Lambda) if use_thinking_residual else None,
        "lr_time_conditioning": float(args.lr_time_conditioning) if use_time_conditioning else None,
    }


def attach_adapter_metadata(model, metadata: dict) -> None:
    """Inject repository metadata into the PEFT config's serialized payload."""
    peft_config = getattr(model, "peft_config", None)
    if peft_config is None:
        raise ValueError("model does not expose peft_config for adapter metadata attachment")

    metadata = dict(metadata)
    configs = peft_config.values() if isinstance(peft_config, dict) else [peft_config]
    for config in configs:
        if getattr(config, "_rot_original_to_dict", None) is None:
            config._rot_original_to_dict = config.to_dict

            def _to_dict_with_metadata(self):
                output = self._rot_original_to_dict()
                output[ADAPTER_METADATA_KEY] = dict(self._rot_metadata)
                return output

            config.to_dict = types.MethodType(_to_dict_with_metadata, config)
        config._rot_metadata = metadata


def load_adapter_metadata(adapter_path: str) -> dict:
    """Load and validate repository-specific adapter metadata from disk."""
    config_path = os.path.join(adapter_path, ADAPTER_CONFIG_NAME)
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist")

    with open(config_path) as handle:
        adapter_config = json.load(handle)

    metadata = adapter_config.get(ADAPTER_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError(f"{config_path} does not contain {ADAPTER_METADATA_KEY}")

    schema_version = int(metadata.get("schema_version", -1))
    if schema_version != ADAPTER_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported adapter metadata schema_version={schema_version}; "
            f"expected {ADAPTER_METADATA_SCHEMA_VERSION}"
        )

    normalized = dict(metadata)
    normalized["mode"] = _require_training_mode(normalized.get("mode"))
    if not normalized.get("base_model"):
        raise ValueError(f"{config_path} is missing {ADAPTER_METADATA_KEY}.base_model")
    if mode_uses_time_conditioning(normalized["mode"]):
        if "thinking_time_predictor_num_hidden_states" not in normalized:
            raise ValueError(
                f"{config_path} is missing {ADAPTER_METADATA_KEY}.thinking_time_predictor_num_hidden_states"
            )
        normalized["thinking_time_predictor_num_hidden_states"] = _coerce_positive_int(
            normalized["thinking_time_predictor_num_hidden_states"],
            field_name="thinking_time_predictor_num_hidden_states",
        )
    normalized["temperature"] = float(normalized.get("temperature", 0.5))
    return normalized


def resolve_time_conditioning_predictor_num_hidden_states(args, mode: str, checkpoint_path: str | None = None) -> int | None:
    """Resolve predictor hidden-state count from CLI defaults or checkpoint metadata."""
    mode = _require_training_mode(mode)
    if not mode_uses_time_conditioning(mode):
        return None

    resolved = _coerce_positive_int(
        getattr(args, "thinking_time_predictor_num_hidden_states", 3),
        field_name="thinking_time_predictor_num_hidden_states",
    )
    if checkpoint_path is None:
        args.thinking_time_predictor_num_hidden_states = resolved
        return resolved

    metadata = load_adapter_metadata(checkpoint_path)
    checkpoint_value = metadata.get("thinking_time_predictor_num_hidden_states")
    if checkpoint_value is None:
        raise ValueError(
            f"{checkpoint_path} is missing thinking_time_predictor_num_hidden_states in saved metadata"
        )
    checkpoint_value = _coerce_positive_int(
        checkpoint_value,
        field_name="thinking_time_predictor_num_hidden_states",
    )
    is_explicit = bool(getattr(args, "_thinking_time_predictor_num_hidden_states_explicit", False))
    if is_explicit and resolved != checkpoint_value:
        raise ValueError(
            "thinking_time_predictor_num_hidden_states does not match checkpoint metadata: "
            f"cli={resolved} checkpoint={checkpoint_value}"
        )

    args.thinking_time_predictor_num_hidden_states = checkpoint_value
    return checkpoint_value


def get_modules_to_save_for_mode(mode: str):
    """Return the PEFT `modules_to_save` list for the selected training mode."""
    mode = _require_training_mode(mode)
    if mode == "grpo":
        return None

    from time_conditioning import THINKING_RESIDUAL_MODULE_NAMES, TIME_CONDITIONING_MODULE_NAMES

    modules_to_save = []
    if mode_uses_thinking_residual(mode):
        modules_to_save.extend(THINKING_RESIDUAL_MODULE_NAMES)
    if mode_uses_time_conditioning(mode):
        modules_to_save.extend(TIME_CONDITIONING_MODULE_NAMES)
    return list(modules_to_save)


def configure_model_for_training_mode(
    model,
    mode: str,
    *,
    max_completion_length: int | None = None,
    thinking_time_predictor_num_hidden_states: int | None = None,
):
    """Apply mode-specific runtime toggles before PEFT wrapping or adapter load."""
    mode = _require_training_mode(mode)
    if mode == "grpo":
        if hasattr(model, "answer_start"):
            delattr(model, "answer_start")
    else:
        model.answer_start = ANSWER_START

    from time_conditioning import (
        enable_time_conditioning,
        get_time_conditioning_base_model,
        set_thinking_residual_disabled,
    )

    inner = get_time_conditioning_base_model(model)
    if max_completion_length is not None:
        max_completion_length = int(max_completion_length)
        if max_completion_length <= 0:
            raise ValueError(f"max_completion_length must be positive, got {max_completion_length}")
        inner.config.max_completion_length = max_completion_length
    if mode_uses_time_conditioning(mode) and thinking_time_predictor_num_hidden_states is not None:
        inner.config.thinking_time_predictor_num_hidden_states = _coerce_positive_int(
            thinking_time_predictor_num_hidden_states,
            field_name="thinking_time_predictor_num_hidden_states",
        )

    if mode_uses_thinking_residual(mode):
        set_thinking_residual_disabled(model, False)
    elif mode_uses_time_conditioning(mode):
        set_thinking_residual_disabled(model, True)

    if mode_uses_time_conditioning(mode):
        enable_time_conditioning(model)
    return model


def reset_residual_lambda_for_mode(model, mode: str, residual_r_min: float, residual_r_max: float) -> None:
    """Reset the thinking-residual radius only for modes that train that path."""
    mode = _require_training_mode(mode)
    if mode_uses_thinking_residual(mode):
        model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
            r_min=residual_r_min,
            r_max=residual_r_max,
        )


def load_training_model_and_tokenizer(args, mode: str):
    """Load the base model/tokenizer pair and apply the requested runtime mode."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
        load_in_4bit=False,
        load_in_8bit=False,
        fast_inference=False,
    )
    configure_model_for_training_mode(
        model,
        mode,
        max_completion_length=args.max_completion_length,
        thinking_time_predictor_num_hidden_states=getattr(
            args,
            "thinking_time_predictor_num_hidden_states",
            None,
        ),
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_training_model(model, args, mode: str):
    """Attach LoRA adapters and reset mode-specific trainable state."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=get_modules_to_save_for_mode(mode),
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    reset_residual_lambda_for_mode(
        model,
        mode,
        residual_r_min=args.residual_r_min,
        residual_r_max=args.residual_r_max,
    )
    return model


def build_training_config(args, output_dir: str):
    """Build the shared GRPOConfig used by all task-specific training scripts."""
    from trl import GRPOConfig
    from unsloth import is_bfloat16_supported

    save_steps = 250
    if args.max_steps is not None and args.max_steps > 0:
        save_steps = min(save_steps, int(args.max_steps))

    return GRPOConfig(
        use_vllm=False,
        learning_rate=args.lr,
        beta=args.beta,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        temperature=args.temperature,
        num_generations=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1,
        max_steps=args.max_steps,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none" if is_wandb_disabled() else "wandb",
        output_dir=output_dir,
    )


def configure_trainer_for_mode(trainer, args, mode: str):
    """Apply mode-specific trainer state such as optimizer patching and aux loss."""
    use_time_conditioning = mode_uses_time_conditioning(mode)
    use_thinking_residual = mode_uses_thinking_residual(mode)
    if use_time_conditioning:
        trainer.thinking_time_loss_weight = args.thinking_time_loss_weight
    if mode != "grpo":
        from patch import patch_trainer_optimizer

        patch_trainer_optimizer(
            trainer,
            lr_thinking_residual_gate=args.lr_residual_gate if use_thinking_residual else None,
            thinking_residual_Lambda=args.lr_residual_Lambda if use_thinking_residual else None,
            lr_time_conditioning=args.lr_time_conditioning if use_time_conditioning else None,
        )
    return trainer


def create_training_trainer(args, task: str, dataset, reward_funcs):
    """Create the shared trainer and return it with resolved mode, resume path, and output dir."""
    from trl import GRPOTrainer

    mode = _require_training_mode(args.mode)
    args.time_predictor_warmup_fraction = validate_time_predictor_warmup_fraction(
        getattr(args, "time_predictor_warmup_fraction", 0.2)
    )
    predictor_hidden_states = resolve_time_conditioning_predictor_num_hidden_states(args, mode)
    exp_name = build_training_exp_name(
        model_name=args.model_name,
        task=task,
        mode=mode,
        group_size=args.group_size,
        lora_rank=args.lora_rank,
        temperature=args.temperature,
        residual_r_min=args.residual_r_min,
        thinking_time_predictor_num_hidden_states=predictor_hidden_states,
        exp_suffix=args.exp_suffix,
    )
    resume_from_checkpoint = resolve_resume_from_checkpoint(exp_name, resume=args.resume)
    resolve_time_conditioning_predictor_num_hidden_states(
        args,
        mode,
        checkpoint_path=resume_from_checkpoint,
    )
    model, tokenizer = load_training_model_and_tokenizer(args, mode)
    model = prepare_training_model(model, args, mode)
    attach_adapter_metadata(model, build_adapter_metadata(args, task, mode))
    print(
        f"[train-config] task={task} mode={mode} "
        f"dataset_size={len(dataset)} max_steps={args.max_steps} "
        f"output_dir={exp_name}"
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=build_training_config(args, exp_name),
        train_dataset=dataset,
    )
    configure_trainer_for_mode(trainer, args, mode)
    return trainer, resume_from_checkpoint, mode, exp_name


def run_training_with_optional_time_predictor_warmup(
    trainer,
    *,
    args,
    task: str,
    mode: str,
    resume_from_checkpoint: str | None,
):
    """Run optional predictor-only warmup, then start the main RL trainer."""
    maybe_run_time_predictor_warmup(
        trainer,
        args=args,
        mode=mode,
        resume_from_checkpoint=resume_from_checkpoint,
        task=task,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def create_training_parser(
    *,
    description: str | None = None,
    group_size: int,
    per_device_train_batch_size: int,
    max_prompt_length: int,
    max_completion_length: int,
    dataset_root_default: str | None = None,
):
    """Build the shared CLI used by all task-specific training entrypoints."""
    parser = argparse.ArgumentParser(
        description=description or (
            "Train a single task with the shared GRPO/TGRPO/HRPO/THRPO entrypoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank for trainable adapters.")

    parser.add_argument("--lr", type=float, default=5e-6, help="Base learning rate for LoRA parameters.")
    parser.add_argument("--beta", type=float, default=0.005, help="KL beta used by GRPO.")
    parser.add_argument("--residual_r_min", type=float, default=0.99, help="Minimum residual-radius initialization.")
    parser.add_argument("--residual_r_max", type=float, default=0.999, help="Maximum residual-radius initialization.")
    parser.add_argument("--lr_residual_gate", type=float, default=1e-4, help="Learning rate for residual gate parameters.")
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3, help="Learning rate for residual-radius parameters.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay applied to decayed parameter groups.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for the shared LR scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Scheduler type passed into GRPOConfig.")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", help="Optimizer name passed into GRPOConfig.")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Gradient clipping threshold.")

    parser.add_argument("--group_size", type=int, default=group_size, help="Number of sampled completions per prompt.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature used during rollout.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=per_device_train_batch_size,
        help="Per-device batch size before gradient accumulation.",
    )
    parser.add_argument("--max_steps", type=int, default=-1, help="Override trainer max_steps for smoke runs.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit the dataset before preprocessing.")
    parser.add_argument("--max_prompt_length", type=int, default=max_prompt_length, help="Maximum prompt token length.")
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=max_completion_length,
        help="Maximum generated completion token length.",
    )

    if dataset_root_default is not None:
        parser.add_argument(
            "--dataset_root",
            type=str,
            default=dataset_root_default,
            help="Dataset root for tasks that read local merged data.",
        )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or local path for direct Python entrypoints.",
    )
    parser.add_argument("--mode", choices=sorted(TRAINING_MODES), required=True, help="Training mode to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling and training.")
    parser.add_argument(
        "--thinking_time_loss_weight",
        type=float,
        default=0.1,
        help="Auxiliary loss weight for time-conditioning modes.",
    )
    parser.add_argument(
        "--lr_time_conditioning",
        type=float,
        default=5e-6,
        help="Learning rate for time-conditioning modules.",
    )
    parser.add_argument(
        "--time_predictor_warmup_fraction",
        "--time-predictor-warmup-fraction",
        dest="time_predictor_warmup_fraction",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for one predictor warmup pass before RL.",
    )
    parser.set_defaults(_thinking_time_predictor_num_hidden_states_explicit=False)
    parser.add_argument(
        "--thinking_time_predictor_num_hidden_states",
        "--thinking-time-predictor-num-hidden-states",
        dest="thinking_time_predictor_num_hidden_states",
        action=_StoreWithExplicitFlag,
        type=int,
        default=3,
        help="Number of final hidden states concatenated into the time predictor.",
    )
    parser.add_argument(
        "--exp-suffix",
        "--exp_suffix",
        dest="exp_suffix",
        type=str,
        default="",
        help="Optional suffix appended to the computed experiment directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the latest checkpoint in the computed experiment directory.",
    )
    return parser


def limit_dataset_samples(dataset, max_train_samples: int | None):
    """Truncate a dataset deterministically when a smoke-test limit is requested."""
    if max_train_samples is None:
        return dataset

    max_train_samples = int(max_train_samples)
    if max_train_samples <= 0:
        raise ValueError(f"max_train_samples must be positive, got {max_train_samples}")

    return dataset.select(range(min(len(dataset), max_train_samples)))


def load_eval_model_and_tokenizer(
    adapter_path: str,
    *,
    max_seq_length: int,
):
    """Restore the base model, tokenizer, and runtime mode from adapter metadata."""
    from unsloth import FastLanguageModel
    metadata = load_adapter_metadata(adapter_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=metadata["base_model"],
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
    )
    configure_model_for_training_mode(
        model,
        metadata["mode"],
        max_completion_length=metadata.get("max_completion_length"),
        thinking_time_predictor_num_hidden_states=metadata.get("thinking_time_predictor_num_hidden_states"),
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer, metadata


def build_generation_config(**kwargs):
    """Create GenerationConfig lazily after Unsloth import ordering is established."""
    # Import Unsloth first so eval scripts do not trigger its transformers-order warning.
    import unsloth  # noqa: F401
    from transformers import GenerationConfig

    return GenerationConfig(**kwargs)


def create_eval_parser(description: str | None = None):
    """Build the shared CLI used by the evaluation entrypoints."""
    parser = argparse.ArgumentParser(
        description=description or "Evaluate a saved adapter checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(greedy=True)
    parser.add_argument(
        "--greedy",
        dest="greedy",
        action="store_true",
        help="Use greedy decoding during evaluation.",
    )
    parser.add_argument(
        "--no-greedy",
        dest="greedy",
        action="store_false",
        help="Disable greedy decoding and let generation follow the saved sampling setup.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory containing adapter_config.json.",
    )
    return parser

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, i.e., "
    "{reasoning process} " + ANSWER_START + " {answer}."
)


def extract_from_response(text: str) -> str:
    """Extract the final answer segment following the `####` answer marker."""
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""


def extract_hash_answer(text: str) -> str | None:
    """Extract a GSM8K answer that follows the canonical `####` marker."""
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the final boxed expression from a MATH-style solution string."""
    try:  # wrap in boxed for process_math_answer
        return "\\boxed{" + find_box(text).strip() + "}"
    except IndexError:
        return None


def reward_func_math(completions, answer, **kwargs) -> list[float]:
    """Reward math completions that both format and solve the problem correctly."""
    responses = [completion[0]["content"] for completion in completions]

    ans = [parse(a) for a in answer]
    extracted = [extract_from_response(r) for r in responses]
    predictions = [parse(r) for r in extracted]
    accuracy = [verify(r, a) for r, a in zip(predictions, ans)]

    escaped_answer_start = re.escape(ANSWER_START)
    pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

    rewards = [1.0 if a and m else 0.0 for a, m in zip(accuracy, matches)]

    print(
        "=" * 50,
        f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
        f"\n1/{len(completions)} responses (answer: {ans[0]}):\n{responses[0]}",
        "\n" + "=" * 50,
    )
    return rewards


def reward_func_rag(completions, answer, **kwargs) -> list[float]:
    """Reward RAG completions that match any normalized golden answer."""
    responses = [completion[0]["content"] for completion in completions]

    extracted = [extract_from_response(r) for r in responses]
    predictions = [process_qa_answer(r) for r in extracted]
    # accuracy = [True if r == a else False for r, a in zip(predictions, ans)]
    accuracy = []
    for normalized_pred, answer_list in zip(predictions, answer):
        cur_accuracy = False
        for golden_answer in answer_list:
            normalized_answer = process_qa_answer(golden_answer)
            if normalized_answer == normalized_pred:
                cur_accuracy = True
        accuracy.append(cur_accuracy)

    escaped_answer_start = re.escape(ANSWER_START)
    pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

    rewards = [1.0 if a and m else 0.0 for a, m in zip(accuracy, matches)]

    print(
        "=" * 50,
        f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
        f"\n1/{len(completions)} responses (answer: {answer[0]}):\n{responses[0]}",
        "\n" + "=" * 50,
    )
    return rewards


def reward_func_mmlu(completions, answer, **kwargs) -> list[float]:
    """Reward multiple-choice completions that emit the correct answer letter."""
    responses = [completion[0]["content"] for completion in completions]

    ans = [process_mmlu_answer(a) for a in answer]
    extracted = [extract_from_response(r) for r in responses]
    predictions = [process_mmlu_answer(r) for r in extracted]
    accuracy = [True if r == a else False for r, a in zip(predictions, ans)]

    escaped_answer_start = re.escape(ANSWER_START)
    pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

    rewards = [1.0 if a and m else 0.0 for a, m in zip(accuracy, matches)]

    print(
        "=" * 50,
        f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
        f"\n1/{len(completions)} responses (answer: {ans[0]}):\n{responses[0]}",
        "\n" + "=" * 50,
    )
    return rewards


def delete_extra_zero(n):
    """Normalize numeric strings by trimming redundant trailing zeroes."""
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            print("Conversion to floating number fails: {}".format(n))
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)
        n=str(n)
        return n


def find_box(pred_str: str):
    """Extract the content of the final `boxed{...}` expression."""
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if (ans[0] == "{"):
        stack = 1
        a = ""
        for c in ans[1:]:
            if (c == "{"):
                stack += 1
                a += c
            elif (c == "}"):
                stack -= 1
                if (stack == 0): break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def find_latex(pred_str: str):
    """Return the final LaTeX span enclosed by standard math delimiters."""
    pattern = re.compile(
        r"""
        (\\\[(?P<display>[\s\S]+?)\\\])                        # \[ ... \]
        |(\\\((?P<inline>[\s\S]+?)\\\))                        # \( ... \)
        |((?P<dollar>\$\$?)(?P<dcontent>[\s\S]+?)(?P=dollar))  # $...$ or $$...$$
        """,
        re.VERBOSE
    )
    matches = list(pattern.finditer(pred_str))
    if not matches: return ""
    return (matches[-1].group("display") or matches[-1].group("inline") 
            or matches[-1].group("dcontent")).strip()


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr:
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.strip("$")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def process_gsm8k_answer(pred: str) -> str:
    """Normalize a GSM8K answer string into its final numeric form."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    pred = [delete_extra_zero(s.replace(",", "")) 
            for s in re.findall(r"-?\d+/?\.?\d*", pred)]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def process_math_answer(pred: str) -> str:
    """Normalize a MATH answer into a comparable canonical string."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    if "boxed" in pred:
        pred = find_box(pred)
    elif find_latex(pred):
        pred = find_latex(pred)
    else:
        preds = re.findall(r"-?\d*\.?\d+", pred)
        if(len(preds) >= 1):
            pred = preds[-1]
        else:
            pred = ""

    pred = _strip_string(pred).rstrip(".").rstrip("/")
    pred = re.sub(r"\\text\{([^}]*)\}", r"\1", pred).lower()
    return pred


def process_mmlu_answer(pred: str) -> str:
    """Normalize a multiple-choice response into a single uppercase answer label."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    tmp = re.findall(r"\b(A|B|C|D|E|F|G|H|I|J)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def process_qa_answer(pred: str) -> str:
    """Normalize open-domain QA answers with lowercasing and punctuation stripping."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(pred))))


def process_gsm8k(batch):
    """Convert a GSM8K batch into the shared prompt/answer training schema."""
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["question"]]

    return {
        "prompt": prompts,
        "answer": [extract_hash_answer(a) for a in batch["answer"]]
    }


def process_math(batch):
    """Convert a MATH batch into the shared prompt/answer training schema."""
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["problem"]]

    return {
        "prompt": prompts,
        "answer": [extract_boxed_answer(a) for a in batch["solution"]]
    }


def process_mmlu(batch):
    """Convert an MMLU batch into the shared prompt/answer training schema."""
    def get_prompt(question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt

    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(q, c).strip()},
    ] for q, c in zip(batch["question"], batch["choices"])]

    return {
        "prompt": prompts,
        "answer": [f"{chr(65 + a)}" for a in batch["answer"]]
    }


def process_rag(batch, topk=3):
    """Convert a RAG batch into the shared prompt/answer training schema."""
    def get_prompt(question, contexts):
        prompt = "Context (which may or may not be relevant):\n"
        for context in contexts[:topk]:
            cur_context = context.split("\n")
            cur_context[0] = cur_context[0].strip('"')
            prompt += "::::".join(cur_context) + "\n"
        prompt += f"\nQuestion: {question}"
        return prompt

    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(q, c).strip()},
    ] for q, c in zip(batch["question"], batch["contexts"])]

    return {
        "prompt": prompts,
        "answer": batch["golden_answers"],
    }
