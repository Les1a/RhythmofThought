import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
import json
import argparse
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_math(split="train", chunk_size=1000, root='../MATH') -> Dataset:
    problems, solutions = [], []
    for folder in os.listdir(os.path.join(root, split)):
        for file in os.listdir(os.path.join(root, split, folder)):
            if file.endswith('.json'):
                with open(os.path.join(root, split, folder, file), 'r') as f:
                    entry = json.load(f)
                problems.append(entry['problem'])
                solutions.append(entry['solution'])
    
    dataset = Dataset.from_dict({
        'problem': problems,
        'solution': solutions,
    })
    return dataset.map(process_math, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    if args.only_grpo and args.tgrpo:
        raise ValueError("--only_grpo and --tgrpo are mutually exclusive")
    if args.tgrpo:
        args.time_conditioning = True
    is_grpo = args.only_grpo
    is_tgrpo = args.tgrpo
    is_thrpo = (not is_grpo) and (not is_tgrpo) and args.time_conditioning
    is_hrpo = (not is_grpo) and (not is_tgrpo) and (not args.time_conditioning)

    if is_grpo:
        exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-math-grpo-group{args.group_size}"
                    f"-lora{args.lora_rank}-temp{args.temperature}")
    elif is_tgrpo:
        exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-math-tgrpo-group{args.group_size}"
                    f"-lora{args.lora_rank}-temp{args.temperature}")
    elif is_thrpo:
        exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-math-thrpo-group{args.group_size}"
                    f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}")
    else:
        exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-math-hrpo-group{args.group_size}"
                    f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}")
    if is_thrpo:
        exp_name += "-tcond"
    resume_from_checkpoint = None
    if args.resume:
        if not os.path.exists(exp_name):
            print(f"--resume specified but {exp_name} does not exist. Exiting...")
            exit()
        resume_from_checkpoint = get_last_checkpoint(exp_name)
        if resume_from_checkpoint is None:
            print(f"--resume specified but no checkpoint-* dirs found in {exp_name}. Exiting...")
            exit()
        print(f"Resuming from {resume_from_checkpoint}")
    elif os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = False,
    )
    if not is_grpo:
        model.answer_start = ANSWER_START
    if is_tgrpo:
        model.disable_thinking_residual = True
        model.model.disable_thinking_residual = True
    if is_tgrpo or is_thrpo:
        from time_conditioning import enable_time_conditioning
        enable_time_conditioning(model)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if is_grpo:
        modules_to_save = None
    elif is_tgrpo:
        modules_to_save = []
    else:
        modules_to_save = [
            "thinking_residual_gate_r",
            "thinking_residual_gate_i",
            "thinking_residual_Lambda",
        ]
    if (is_tgrpo or is_thrpo) and modules_to_save is not None:
        modules_to_save.extend([
            "time_progress_predictor", "sinusoidal_time_embedding",
            "adaln_proj",
        ])
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save = modules_to_save,
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    if is_hrpo or is_thrpo:
        model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
            r_min = args.residual_r_min, r_max = args.residual_r_max,
        )
    training_args = GRPOConfig(
        use_vllm = False,
        learning_rate = args.lr,
        beta = args.beta,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optimizer,
        max_grad_norm = args.max_grad_norm,
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        temperature = args.temperature,
        num_generations = args.group_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = 1,
        save_steps = 250,
        save_total_limit = 3,
        report_to = "none" if os.environ.get("WANDB_DISABLED", "").lower() == "true" else "wandb",
        output_dir = exp_name,
    )

    dataset = preprocess_math('train', chunk_size=500, root=args.dataset_root)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reward_func_math,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    if is_tgrpo or is_thrpo:
        trainer.time_loss_weight = args.time_loss_weight
    if not is_grpo:
        patch_trainer_optimizer(
            trainer,
            lr_thinking_residual_gate=None if is_tgrpo else args.lr_residual_gate,
            thinking_residual_Lambda=None if is_tgrpo else args.lr_residual_Lambda,
            lr_time_conditioning=args.lr_time_conditioning if (is_tgrpo or is_thrpo) else None,
        )
    if (is_thrpo
            and resume_from_checkpoint is None and args.pretrain_time_predicator):
        from time_conditioning import pretrain_time_predictor
        print("Pretraining time predictor...")
        pretrain_time_predictor(
            model, tokenizer, dataset,
            num_samples=args.pretrain_time_samples,
            num_epochs=args.pretrain_time_epochs,
            lr=args.lr_time_conditioning,
            temperature=args.temperature,
            max_completion_length=args.max_completion_length,
            batch_size=128,
        )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
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

    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=2048)

    parser.add_argument("--dataset_root", type=str, default="../MATH")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_conditioning", action="store_true", default=False)
    parser.add_argument("--time_loss_weight", type=float, default=0.1)
    parser.add_argument("--lr_time_conditioning", type=float, default=1e-4)
    parser.add_argument("--pretrain-time-predicator", action="store_true", default=False)
    parser.add_argument("--pretrain_time_samples", type=int, default=1024)
    parser.add_argument("--pretrain_time_epochs", type=int, default=3)
    parser.add_argument("--only_grpo", action="store_true", default=False)
    parser.add_argument("--tgrpo", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"

    main(args)
