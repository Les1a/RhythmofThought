import unsloth
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import json
import os
from datasets import Dataset
from utils import (
    create_training_parser,
    create_training_trainer,
    limit_dataset_samples,
    process_math,
    reward_func_math,
    run_training_with_optional_time_predictor_warmup,
)

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_math(split="train", chunk_size=1000, root='../MATH', max_train_samples=None) -> Dataset:
    problems, solutions = [], []
    for folder in os.listdir(os.path.join(root, split)):
        for file in os.listdir(os.path.join(root, split, folder)):
            if file.endswith('.json'):
                with open(os.path.join(root, split, folder, file), 'r') as f:
                    entry = json.load(f)
                problems.append(entry['problem'])
                solutions.append(entry['solution'])
                if max_train_samples is not None and len(problems) >= int(max_train_samples):
                    break
        if max_train_samples is not None and len(problems) >= int(max_train_samples):
            break
    
    dataset = Dataset.from_dict({
        'problem': problems,
        'solution': solutions,
    })
    dataset = limit_dataset_samples(dataset, max_train_samples)
    return dataset.map(process_math, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    dataset = preprocess_math(
        'train',
        chunk_size=500,
        root=args.dataset_root,
        max_train_samples=args.max_train_samples,
    )
    trainer, resume_from_checkpoint, mode, _ = create_training_trainer(
        args,
        task="math",
        dataset=dataset,
        reward_funcs=[reward_func_math],
    )
    run_training_with_optional_time_predictor_warmup(
        trainer,
        args=args,
        task="math",
        mode=mode,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    parser = create_training_parser(
        group_size=8,
        per_device_train_batch_size=16,
        max_prompt_length=2048,
        max_completion_length=2048,
        dataset_root_default="../MATH",
    )
    args = parser.parse_args()
    main(args)
