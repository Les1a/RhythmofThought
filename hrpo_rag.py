"""Train a RAG checkpoint through the shared GRPO/TGRPO/HRPO/THRPO stack."""

import unsloth
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
from datasets import Dataset
from utils import (
    create_training_parser,
    create_training_trainer,
    limit_dataset_samples,
    process_rag,
    reward_func_rag,
    run_training_with_optional_time_predictor_warmup,
)

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_rag(chunk_size=1000, root='../RAG_Train_Merged', max_train_samples=None) -> Dataset:
    """Load the merged RAG dataset and normalize it into the shared train schema."""
    dataset = Dataset.load_from_disk(root)
    dataset = limit_dataset_samples(dataset, max_train_samples)
    processed = dataset.map(process_rag, batched=True,
                            batch_size=chunk_size, load_from_cache_file=False)
    return processed


def main(args):
    """Create the trainer for RAG and launch optional warmup plus RL training."""
    dataset = preprocess_rag(
        chunk_size=500,
        root=args.dataset_root,
        max_train_samples=args.max_train_samples,
    )
    trainer, resume_from_checkpoint, mode, _ = create_training_trainer(
        args,
        task="rag",
        dataset=dataset,
        reward_funcs=[reward_func_rag],
    )
    run_training_with_optional_time_predictor_warmup(
        trainer,
        args=args,
        task="rag",
        mode=mode,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    parser = create_training_parser(
        description=(
            "Train RAG with the shared GRPO/TGRPO/HRPO/THRPO entrypoint. "
            "Use --mode to select the training behavior."
        ),
        group_size=4,
        per_device_train_batch_size=16,
        max_prompt_length=2048,
        max_completion_length=1024,
        dataset_root_default="../RAG_Train_Merged",
    )
    args = parser.parse_args()
    main(args)
