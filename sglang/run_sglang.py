from datasets import load_dataset
import sglang as sgl
import asyncio
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference with a specific model path.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs.jsonl",
    )
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files="alpaca_eval.json",
        split="train",
    )
    model_path = args.model_path

    # TODO: initialize sglang egnine here
    # you may want to explore different args we can pass here to make the inference faster
    # e.g. dp_size, mem_fraction_static
    llm = sgl.Engine(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=True,
        mem_fraction_static=0.75,
        cuda_graph_max_bs=64,
        attention_backend="dual_chunk_flash_attn",
    )

    prompts = []

    for i in dataset:
        prompts.append(i['instruction'])

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 1024}

    outputs = []

    # TODO: you may want to explore different batch_size
    batch_size = min(32, len(prompts))

    from tqdm import tqdm
    for i in tqdm(range(0, len(prompts), batch_size)):
        # TODO: prepare the batched prompts and use llm.generate
        # save the output in outputs
        batch_prompts = prompts[i:i + batch_size]
        batch_outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params,
        )
        # Extract the generated text from the output objects
        outputs.extend([out["text"] for out in batch_outputs])

    with open(args.output_file, "w") as f:
        for i in range(0, len(outputs), 10):
            instruction = prompts[i]
            output = outputs[i]
            f.write(json.dumps({
                "output": output,
                "instruction": instruction
            }) + "\n")

if __name__ == "__main__":
    main()
