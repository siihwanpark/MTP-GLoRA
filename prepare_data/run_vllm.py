import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser("vLLM JSONL Inference")

    # Model / tokenizer
    parser.add_argument("--model", type=str, required=True,
                   help="HuggingFace model name or local path (e.g., Qwen/Qwen3-8B).")
    parser.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer name/path. Defaults to --model.")
    parser.add_argument("--tp_size", type=int, default=8,
                   help="vLLM tensor_parallel_size. Set to number of GPUs to use.")
    parser.add_argument("--gpu_mem_util", type=float, default=0.9,
                   help="vLLM gpu_memory_utilization (0~1).")

    # Data
    parser.add_argument("--dataset", type=str, default=None,
                   help="Dataset name from HuggingFace.")
    parser.add_argument("--input_jsonl", type=str, default=None,
                   help="JSONL file with lines like {'prompt': '...'}")
    parser.add_argument("--num_samples", type=int, default=None,
                   help="Limit number of prompts (None = all).")
    parser.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling `num_samples`.")

    # Prompt shaping
    parser.add_argument("--use_chat_template", action="store_true",
                   help="Use tokenizer.apply_chat_template with role='user'.")
    parser.add_argument("--enable_thinking", action="store_true",
                   help="If --use_chat_template, pass enable_thinking=True (Qwen3 supports this).")
    parser.add_argument("--system_prompt", type=str, default=None,
                   help="Optional system prompt when using chat template.")
    parser.add_argument("--prefix_len", type=int, default=None,
                   help="If set, truncate prompt to this many tokens before generation.")

    # Generation
    parser.add_argument("--max_model_len", type=int, default=34816)
    parser.add_argument("--max_gen_len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)

    # Output
    parser.add_argument("--output_dir", type=str, default="vllm_results")
    parser.add_argument("--outfile_prefix", type=str, default="vllm_jsonl")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    return args


def sample_prompts(
    prompts: List[Dict[str, Any]],
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if num_samples is None:
        return prompts
    
    rng = random.Random(seed) if seed is not None else random
    return rng.sample(prompts, num_samples)


def read_prompts_from_dataset(
    dataset_name: str,
    split: str = "train",
    prompt_key: str = "question",
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    dataset = load_dataset(dataset_name, split=split)
    records = []
    for item in dataset:
        records.append({"prompt": item[prompt_key]})

    return sample_prompts(records, num_samples, seed)


def read_prompts_from_jsonl(
    jsonl_path: str,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:

    def _normalize(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "prompt" not in obj:
            if "input" in obj and isinstance(obj["input"], str):
                obj["prompt"] = obj["input"]
            else:
                return None
        return obj
    
    prompts: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj = _normalize(obj)
            if obj is None:
                continue
            prompts.append(obj)

    return sample_prompts(prompts, num_samples, seed)


def apply_chat_template(tokenizer, text: str, system_prompt: Optional[str], enable_thinking: bool) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    # Qwen3 supports enable_thinking flag; others will just ignore extra kw.
    # Ref: Qwen docs about apply_chat_template & thinking mode.
    try:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Some tokenizers don't accept enable_thinking kwarg
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return templated


def truncate_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class VLLMRunner:
    def __init__(self, args):
        self.args = args
        print(f"[Init] Loading tokenizer: {args.tokenizer}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="right", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[Init] Loading vLLM model on {args.tp_size} GPUs: {args.model}")
        self.llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            trust_remote_code=True,
        )

        self.sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_gen_len,
        )


    def build_inputs(self, records: List[Dict[str, Any]]) -> List[str]:
        inputs = []
        for rec in records:
            prompt = rec["prompt"]
            if self.args.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                prompt = apply_chat_template(
                    self.tokenizer, prompt, self.args.system_prompt, self.args.enable_thinking
                )
            if self.args.prefix_len:
                prompt = truncate_by_tokens(self.tokenizer, prompt, self.args.prefix_len)
            inputs.append(prompt)
        return inputs


    def run(self):
        t0 = time.time()
        if self.args.dataset is not None:
            records = read_prompts_from_dataset(self.args.dataset, split="train", prompt_key="question", num_samples=self.args.num_samples, seed=self.args.seed)
        elif self.args.input_jsonl is not None:
            records = read_prompts_from_jsonl(self.args.input_jsonl, num_samples=self.args.num_samples, seed=self.args.seed)
        else:
            raise ValueError("Either --dataset or --input_jsonl must be provided")
        
        print(f"[Data] Loaded {len(records)} prompts from {self.args.dataset if self.args.dataset is not None else self.args.input_jsonl}")

        inputs = self.build_inputs(records)
        print(f"[Gen] Generating with vLLM... (batching handled internally)")

        # vLLM generate with continuous batching
        outputs = self.llm.generate(inputs, self.sampling)

        results = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        for i, out in enumerate(tqdm(outputs, desc="Collecting outputs")):
            # vLLM returns one or more candidates per prompt; we take the first
            cand = out.outputs[0]
            input_tokens = len(out.prompt_token_ids)
            output_tokens = len(cand.token_ids)
            total_prompt_tokens += input_tokens
            total_output_tokens += output_tokens

            item = {
                "input": inputs[i],
                "output": cand.text,
            }
            results.append(item)

        t1 = time.time()
        self.save(results, t1 - t0, total_prompt_tokens, total_output_tokens)


    def save(self, results: List[Dict[str, Any]], elapsed: float, ptoks: int, otoks: int):
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        outpath = Path(self.args.output_dir) / f"{self.args.outfile_prefix}_{ts}.json"

        payload = {
            "config": {
                "model": self.args.model,
                "tokenizer": self.args.tokenizer,
                "tp_size": self.args.tp_size,
                "gpu_memory_util": self.args.gpu_mem_util,
                "max_model_len": self.args.max_model_len,
                "prefix_len": self.args.prefix_len,
                "max_gen_len": self.args.max_gen_len,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "use_chat_template": self.args.use_chat_template,
                "enable_thinking": self.args.enable_thinking,
                "system_prompt": self.args.system_prompt,
                "input_jsonl": self.args.input_jsonl,
                "num_samples": self.args.num_samples,
            },
            "runtime": {
                "elapsed_sec": elapsed,
                "num_samples": len(results),
                "prompt_tokens": ptoks,
                "output_tokens": otoks,
                "total_tokens": ptoks + otoks,
                "samples_per_sec": (len(results) / elapsed) if elapsed > 0 else None,
                "tok_per_sec": ((ptoks + otoks) / elapsed) if elapsed > 0 else None,
            },
            "results": results,
        }

        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        rt = payload["runtime"]
        print("\n=== Summary ===")
        print(f"Saved: {outpath}")
        print(f"Samples: {rt['num_samples']}, time: {rt['elapsed_sec']:.2f}s")
        print(f"Tokens: {rt['total_tokens']} (prompt {rt['prompt_tokens']}, output {rt['output_tokens']})")
        print(f"Throughput: {rt['samples_per_sec']:.2f} samples/s, {rt['tok_per_sec']:.2f} tok/s")


def main():
    args = parse_args()
    runner = VLLMRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
