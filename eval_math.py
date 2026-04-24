"""Evaluate a saved checkpoint on MATH and report the MATH-500 subset score."""

import os
import json
from datetime import datetime
from datasets import load_dataset, Dataset
from math_verify import parse, verify
from tqdm import tqdm

from utils import (
    SYSTEM_PROMPT,
    build_generation_config,
    create_eval_parser,
    extract_boxed_answer,
    extract_from_response,
    load_eval_model_and_tokenizer,
    process_math,
)


def preprocess_math(split="train", chunk_size=1000, root='../MATH') -> Dataset:
    """Load the local MATH directory and normalize it into the shared eval schema."""
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


def evaluate_model(
    adapter_path: str,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
):
    """Run batched MATH evaluation and track both full and MATH-500 accuracy."""
    model, tokenizer, metadata = load_eval_model_and_tokenizer(
        adapter_path,
        max_seq_length=4096,
    )
    temperature = metadata["temperature"]

    dataset = preprocess_math('test', chunk_size=500)
    math500 = load_dataset('HuggingFaceH4/MATH-500')['test']
    math500_problems = set(math500['problem'])

    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0
    correct_math500 = 0
    total_math500 = 0

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
        current_batch_size = len(batch_data['problem'])

        # Prepare prompts using the same format as training
        problems = batch_data['problem']
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.strip()},
            ]
            for q in batch_data['problem']
        ]

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
                max_new_tokens=2048,
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
            generated_answer = parse(extracted)
            true_answer = extract_boxed_answer(batch_data['solution'][j])
            true_answer = parse(true_answer)
            is_correct = bool(verify(generated_answer, true_answer))
            print(generated_answer, true_answer, is_correct)

            if problems[j] in math500_problems:
                total_math500 += 1
                if is_correct:
                    correct_math500 += 1

            # Store the result
            result = {
                'question': batch_data['problem'][j],
                'true_answer': str(true_answer),
                'generated_answer': str(generated_answer),
                'full_response': response,
                'correct': is_correct,
            }
            results.append(result)

            if is_correct:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    accuracy_math500 = correct_math500 / total_math500 if total_math500 > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'accuracy_math500': accuracy_math500,
        'correct_math500': correct_math500,
        'total_math500': total_math500,
        'model_path': adapter_path,
        'timestamp': datetime.now().isoformat()
    }

    if save_results:
        save_path = adapter_path + "/eval_results.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    args = create_eval_parser(
        description=(
            "Evaluate a checkpoint on MATH. The evaluator restores the base "
            "model, mode, and temperature from adapter metadata."
        )
    ).parse_args()
    checkpoint_path = args.checkpoint_path
    print(f"Starting MATH evaluation metadata load from {checkpoint_path}")

    if not os.path.exists(os.path.join(checkpoint_path, 'eval_results.json')):
        print(f"Starting MATH evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            adapter_path=checkpoint_path,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=None,
            save_results=True,
        )
