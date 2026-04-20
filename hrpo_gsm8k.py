import unsloth
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
from datasets import load_dataset, Dataset
from utils import (
    create_training_parser,
    create_training_trainer,
    limit_dataset_samples,
    process_gsm8k,
    reward_func_math,
    run_training_with_optional_time_predictor_warmup,
)

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_gsm8k(split="train", chunk_size=1000, max_train_samples=None) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    dataset = limit_dataset_samples(dataset, max_train_samples)
    return dataset.map(process_gsm8k, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    dataset = preprocess_gsm8k('train', chunk_size=500, max_train_samples=args.max_train_samples)
    trainer, resume_from_checkpoint, mode, _ = create_training_trainer(
        args,
        task="gsm8k",
        dataset=dataset,
        reward_funcs=[reward_func_math],
    )
    run_training_with_optional_time_predictor_warmup(
        trainer,
        args=args,
        task="gsm8k",
        mode=mode,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    parser = create_training_parser(
        group_size=4,
        per_device_train_batch_size=8,
        max_prompt_length=1024,
        max_completion_length=1024,
    )
    args = parser.parse_args()
    main(args)
