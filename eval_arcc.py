"""Evaluate a saved checkpoint on ARC-style multiple-choice benchmarks."""

import os
import json
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

from utils import (
    SYSTEM_PROMPT,
    build_generation_config,
    create_eval_parser,
    extract_from_response,
    load_eval_model_and_tokenizer,
    process_mmlu_answer,
)


def evaluate_model(
    adapter_path: str,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    dataset_code: str = 'ai2_arc',
):
    """Run batched evaluation on ARC-C, OpenBookQA, or QASC and save results."""
    def get_prompt(question, choices):
        """Format a multiple-choice question with lettered options."""
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt

    model, tokenizer, metadata = load_eval_model_and_tokenizer(
        adapter_path,
        max_seq_length=2048,
    )
    temperature = metadata["temperature"]

    if dataset_code == 'ai2_arc':
        dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge')['test']
    elif dataset_code == 'openbookqa':
        dataset = load_dataset('allenai/openbookqa', 'main')['test']
    elif dataset_code == 'qasc':
        dataset = load_dataset('allenai/qasc', 'default')['validation']
    
    question_key = 'question_stem' if 'openbookqa' in dataset_code else 'question'
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data[question_key])

        # Prepare prompts using the same format as training
        prompts = [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_prompt(q, c['text']).strip()},
        ] for q, c in zip(batch_data[question_key], batch_data["choices"])]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            generation_config=build_generation_config(
                do_sample=True,  # for temperature, top-k, etc.
                temperature=temperature,
                max_new_tokens=512,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
        )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(
                tokenizer.special_tokens_map['eos_token']
            )[0]

            # Extract the generated answer using XML tags
            extracted = extract_from_response(response)
            generated_answer = process_mmlu_answer(extracted)
            true_answer = batch_data['answerKey'][j]
            true_answer = process_mmlu_answer(true_answer)
            print(generated_answer, true_answer, generated_answer == true_answer)

            # Store the result
            result = {
                'question': batch_data[question_key][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': generated_answer == true_answer
            }
            results.append(result)

            if generated_answer == true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': adapter_path,
        'timestamp': datetime.now().isoformat()
    }

    if save_results:
        save_path = adapter_path + f"/eval_results_{dataset_code}.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    args = create_eval_parser(
        description=(
            "Evaluate a checkpoint on ARC-style benchmarks. The evaluator restores "
            "the base model, mode, and temperature from adapter metadata."
        )
    ).parse_args()
    checkpoint_path = args.checkpoint_path
    print(f"Starting ARC-C evaluation metadata load from {checkpoint_path}")

    if not os.path.exists(os.path.join(checkpoint_path, 'eval_results_ai2_arc.json')):
        print(f"Starting ARC-C evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            adapter_path=checkpoint_path,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=None,
            save_results=True,
            dataset_code='ai2_arc',
        )
