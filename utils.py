import argparse
import json
import os
import re
import string
import types
from math_verify import parse, verify


ANSWER_START = "####"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_METADATA_KEY = "rot_metadata"
ADAPTER_METADATA_SCHEMA_VERSION = 1
TRAINING_MODES = frozenset({"grpo", "tgrpo", "thrpo", "hrpo"})


def normalize_exp_suffix(suffix: str | None) -> str:
    if suffix is None:
        return ""
    suffix = suffix.strip()
    if not suffix:
        return ""
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix)
    return suffix.strip("-.")


def is_wandb_disabled() -> bool:
    return (
        os.environ.get("WANDB_DISABLED", "").lower() == "true"
        or os.environ.get("WANDB_MODE", "").lower() == "disabled"
    )


def _require_training_mode(mode: str) -> str:
    if mode not in TRAINING_MODES:
        raise ValueError(f"unknown training mode: {mode}")
    return mode


def mode_uses_time_conditioning(mode: str) -> bool:
    return _require_training_mode(mode) in {"tgrpo", "thrpo"}


def mode_uses_thinking_residual(mode: str) -> bool:
    return _require_training_mode(mode) in {"hrpo", "thrpo"}


def build_training_exp_name(
    model_name,
    task,
    mode,
    group_size,
    lora_rank,
    temperature,
    residual_r_min=None,
    exp_suffix="",
):
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

    exp_suffix = normalize_exp_suffix(exp_suffix)
    if exp_suffix:
        parts.append(exp_suffix)

    return "./experiments/" + "-".join(parts)


def resolve_resume_from_checkpoint(exp_name: str, resume: bool) -> str | None:
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
        "lr": float(args.lr),
        "lr_residual_gate": float(args.lr_residual_gate) if use_thinking_residual else None,
        "lr_residual_lambda": float(args.lr_residual_Lambda) if use_thinking_residual else None,
        "lr_time_conditioning": float(args.lr_time_conditioning) if use_time_conditioning else None,
    }


def attach_adapter_metadata(model, metadata: dict) -> None:
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
    normalized["temperature"] = float(normalized.get("temperature", 0.5))
    return normalized


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


def configure_model_for_training_mode(model, mode: str, *, max_completion_length: int | None = None):
    """Apply runtime toggles needed by a training mode before PEFT wrapping or adapter load."""
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
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
        load_in_4bit=False,
        load_in_8bit=False,
        fast_inference=False,
    )
    configure_model_for_training_mode(model, mode, max_completion_length=args.max_completion_length)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_training_model(model, args, mode: str):
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
    from trl import GRPOTrainer

    mode = _require_training_mode(args.mode)
    exp_name = build_training_exp_name(
        model_name=args.model_name,
        task=task,
        mode=mode,
        group_size=args.group_size,
        lora_rank=args.lora_rank,
        temperature=args.temperature,
        residual_r_min=args.residual_r_min,
        exp_suffix=args.exp_suffix,
    )
    resume_from_checkpoint = resolve_resume_from_checkpoint(exp_name, resume=args.resume)
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


def create_training_parser(
    *,
    group_size: int,
    per_device_train_batch_size: int,
    max_prompt_length: int,
    max_completion_length: int,
    dataset_root_default: str | None = None,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--residual_r_min", type=float, default=0.99)
    parser.add_argument("--residual_r_max", type=float, default=0.999)
    parser.add_argument("--lr_residual_gate", type=float, default=1e-4)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    parser.add_argument("--group_size", type=int, default=group_size)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=per_device_train_batch_size,
    )
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=max_prompt_length)
    parser.add_argument("--max_completion_length", type=int, default=max_completion_length)

    if dataset_root_default is not None:
        parser.add_argument("--dataset_root", type=str, default=dataset_root_default)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", choices=sorted(TRAINING_MODES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking_time_loss_weight", type=float, default=0.1)
    parser.add_argument("--lr_time_conditioning", type=float, default=1e-4)
    parser.add_argument("--exp-suffix", "--exp_suffix", dest="exp_suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)
    return parser


def limit_dataset_samples(dataset, max_train_samples: int | None):
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
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer, metadata


def build_generation_config(**kwargs):
    # Import Unsloth first so eval scripts do not trigger its transformers-order warning.
    import unsloth  # noqa: F401
    from transformers import GenerationConfig

    return GenerationConfig(**kwargs)


def create_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-greedy", dest="greedy", action="store_false", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    return parser

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, i.e., "
    "{reasoning process} " + ANSWER_START + " {answer}."
)


def extract_from_response(text: str) -> str:
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""


def extract_hash_answer(text: str) -> str | None:
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None


def extract_boxed_answer(text: str) -> str | None:
    try:  # wrap in boxed for process_math_answer
        return "\\boxed{" + find_box(text).strip() + "}"
    except IndexError:
        return None


def reward_func_math(completions, answer, **kwargs) -> list[float]:
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
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    pred = [delete_extra_zero(s.replace(",", "")) 
            for s in re.findall(r"-?\d+/?\.?\d*", pred)]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def process_math_answer(pred: str) -> str:
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
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["question"]]

    return {
        "prompt": prompts,
        "answer": [extract_hash_answer(a) for a in batch["answer"]]
    }


def process_math(batch):
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["problem"]]

    return {
        "prompt": prompts,
        "answer": [extract_boxed_answer(a) for a in batch["solution"]]
    }


def process_mmlu(batch):
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
