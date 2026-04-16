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
)

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_rag(chunk_size=1000, root='../RAG_Train_Merged', max_train_samples=None) -> Dataset:
    dataset = Dataset.load_from_disk(root)
    dataset = limit_dataset_samples(dataset, max_train_samples)
    processed = dataset.map(process_rag, batched=True,
                            batch_size=chunk_size, load_from_cache_file=False)
    return processed


def main(args):
    dataset = preprocess_rag(
        chunk_size=500,
        root=args.dataset_root,
        max_train_samples=args.max_train_samples,
    )
    trainer, resume_from_checkpoint, _, _ = create_training_trainer(
        args,
        task="rag",
        dataset=dataset,
        reward_funcs=[reward_func_rag],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = create_training_parser(
        group_size=4,
        per_device_train_batch_size=16,
        max_prompt_length=2048,
        max_completion_length=1024,
        dataset_root_default="../RAG_Train_Merged",
    )
    args = parser.parse_args()
    main(args)
