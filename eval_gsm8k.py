import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def evaluate_model(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
    only_grpo: bool = False,
    tgrpo: bool = False,
):
    if only_grpo and tgrpo:
        raise ValueError("--only_grpo and --tgrpo are mutually exclusive")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = False,
        fast_inference = False,
    )
    if not only_grpo:
        model.answer_start = ANSWER_START
    if tgrpo:
        model.disable_thinking_residual = True
        model.model.disable_thinking_residual = True
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    from time_conditioning import detect_time_conditioning
    detect_time_conditioning(model, adapter_path)
    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    dataset = load_dataset('openai/gsm8k', 'main')['test']
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
        current_batch_size = len(batch_data['question'])

        # Prepare prompts using the same format as training
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.strip()},
            ]
            for q in batch_data['question']
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
            generation_config=GenerationConfig(
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
            true_answer = extract_hash_answer(batch_data['answer'][j])
            true_answer = parse(true_answer)
            is_correct = bool(verify(generated_answer, true_answer))
            print(generated_answer, true_answer, is_correct)

            # Store the result
            result = {
                'question': batch_data['question'][j],
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
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
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
    from utils import detect_base_model, detect_temperature, create_eval_parser

    args = create_eval_parser().parse_args()
    checkpoint_path = args.checkpoint_path
    base_model = detect_base_model(checkpoint_path)
    temperature = detect_temperature(checkpoint_path)
    print(checkpoint_path, base_model, temperature)

    if not os.path.exists(os.path.join(checkpoint_path, 'eval_results.json')):
        print(f"Starting GSM8k evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=None,
            save_results=True,
            only_grpo=args.only_grpo,
            tgrpo=args.tgrpo,
        )
