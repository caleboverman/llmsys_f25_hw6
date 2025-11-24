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
    llm = sgl.Runtime(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=True,
        mem_fraction_static=0.85,
        attention_backend="dual_chunk_flash_attn",
    )

    prompts = []

    for i in dataset:
        prompts.append(i['instruction'])

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8192}

    outputs = []

    # TODO: you may want to explore different batch_size
    batch_size = min(8, len(prompts)) 

    from tqdm import tqdm
    for i in tqdm(range(0, len(prompts), batch_size)):
        # TODO: prepare the batched prompts and use llm.generate
        # save the output in outputs
        if i == 0:
            @sgl.function
            def alpaca_eval_program(s, prompt: str):
                s += prompt
                s += sgl.gen(
                    "output",
                    max_new_tokens=sampling_params["max_new_tokens"],
                )
                return {"output": s["output"]}

        batch_prompts = prompts[i:i + batch_size]
        payload = [{"prompt": prompt} for prompt in batch_prompts]
        batch_outputs = llm.generate(
            alpaca_eval_program,
            payload,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_new_tokens=sampling_params["max_new_tokens"],
        )
        outputs.extend([sample["output"] for sample in batch_outputs])

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
