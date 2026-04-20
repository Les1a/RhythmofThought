import unsloth
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from datasets import Dataset
import os

from utils import (
    create_training_parser,
    create_training_trainer,
    limit_dataset_samples,
    process_mmlu,
    reward_func_mmlu,
    run_training_with_optional_time_predictor_warmup,
)

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_mmlu(chunk_size=1000, root='../MMLU_Train_Merged', max_train_samples=None) -> Dataset:
    dataset = Dataset.load_from_disk(root)
    dataset = limit_dataset_samples(dataset, max_train_samples)
    # Cast ClassLabel answer to string to avoid auto-conversion back to int
    from datasets import Value
    dataset = dataset.cast_column("answer", Value("int32"))
    processed = dataset.map(process_mmlu, batched=True,
                            batch_size=chunk_size, load_from_cache_file=False,
                            remove_columns=dataset.column_names)
    return processed


def main(args):
    dataset = preprocess_mmlu(
        chunk_size=500,
        root=args.dataset_root,
        max_train_samples=args.max_train_samples,
    )
    trainer, resume_from_checkpoint, mode, _ = create_training_trainer(
        args,
        task="mmlu",
        dataset=dataset,
        reward_funcs=[reward_func_mmlu],
    )
    run_training_with_optional_time_predictor_warmup(
        trainer,
        args=args,
        task="mmlu",
        mode=mode,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    parser = create_training_parser(
        group_size=8,
        per_device_train_batch_size=16,
        max_prompt_length=1024,
        max_completion_length=1024,
        dataset_root_default="../MMLU_Train_Merged",
    )
    args = parser.parse_args()
    main(args)
