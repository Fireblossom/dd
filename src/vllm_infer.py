#!/usr/bin/env python3
"""
Standalone vLLM Inference Script - No LLaMA-Factory Dependencies

Supports:
- Loading data from JSONL files
- Optional LoRA adapter
- Batch inference
"""
import argparse
import gc
import json
import os
from typing import Optional, List, Dict

from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    raise ImportError("Please install vllm: pip install vllm")

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_dataset_from_info(dataset_name: str, dataset_dir: str) -> List[Dict]:
    """Load dataset via dataset_info.json."""
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if dataset_name in info:
            file_name = info[dataset_name]["file_name"]
            return load_jsonl(os.path.join(dataset_dir, file_name))
    
    # Fallback: try loading dataset_name.jsonl directly
    jsonl_path = os.path.join(dataset_dir, f"{dataset_name}.jsonl")
    if os.path.exists(jsonl_path):
        return load_jsonl(jsonl_path)
    
    raise FileNotFoundError(f"Dataset not found: {dataset_name} in {dataset_dir}")


def build_prompt(sample: Dict, tokenizer, template: str = "qwen") -> str:
    """Build inference prompt.
    
    Supported templates: qwen, yi, gemma, llama, default
    """
    system = sample.get("system", "")
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    
    # Combine instruction and input
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n{input_text}" if instruction else input_text
    
    # Build prompt based on template
    if template == "qwen":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    elif template == "yi":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    elif template == "gemma":
        messages = []
        if system:
            user_content = f"{system}\n\n{user_content}"
        messages.append({"role": "user", "content": user_content})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    elif template == "llama":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    else:
        # Default: simple concatenation
        if system:
            return f"{system}\n\n{user_content}\n\n"
        return f"{user_content}\n\n"


def vllm_infer(
    model_name_or_path: str,
    dataset: str,
    dataset_dir: str,
    save_name: str = "generated_predictions.jsonl",
    adapter_name_or_path: Optional[str] = None,
    template: str = "qwen",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    batch_size: int = 1024,
    gpu_memory_utilization: float = 0.92,
    tensor_parallel_size: Optional[int] = None,
    max_model_len: int = 4096,
    seed: Optional[int] = None,
):
    """Batch inference using vLLM."""
    print(f"Loading model: {model_name_or_path}")
    print(f"Template: {template}")
    if adapter_name_or_path:
        print(f"Loading LoRA adapter: {adapter_name_or_path}")
    
    # Auto-detect GPU count (default to all available)
    if tensor_parallel_size is None:
        import torch
        tensor_parallel_size = torch.cuda.device_count() or 1
    print(f"Using {tensor_parallel_size} GPU(s) with tensor parallelism")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Initialize vLLM engine
    engine_args = {
        "model": model_name_or_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "disable_log_stats": True,
        "enable_lora": adapter_name_or_path is not None,
    }
    
    llm = LLM(**engine_args)
    
    # Load dataset
    print(f"Loading dataset: {dataset} from {dataset_dir}")
    samples = load_dataset_from_info(dataset, dataset_dir)
    print(f"Total samples: {len(samples)}")
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    
    # Set LoRA request
    lora_request = None
    if adapter_name_or_path:
        lora_request = LoRARequest("default", 1, adapter_name_or_path)
    
    # Batch inference
    all_results = []
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Inferencing"):
        batch = samples[i:i + batch_size]
        prompts = [build_prompt(s, tokenizer, template) for s in batch]
        labels = [s.get("output", "") for s in batch]
        
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        predictions = [out.outputs[0].text for out in outputs]
        
        for prompt, pred, label in zip(prompts, predictions, labels):
            all_results.append({
                "prompt": prompt,
                "predict": pred,
                "label": label,
            })
        
        gc.collect()
    
    # Save results
    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)
    with open(save_name, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print("*" * 70)
    print(f"Generated {len(all_results)} predictions, saved to {save_name}")
    print("*" * 70)


def main():
    parser = argparse.ArgumentParser(description="vLLM Batch Inference (Standalone)")
    parser.add_argument("--model_name_or_path", required=True, help="Model path")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory")
    parser.add_argument("--save_name", default="generated_predictions.jsonl", help="Output file")
    parser.add_argument("--adapter_name_or_path", default=None, help="LoRA adapter path")
    parser.add_argument("--template", default="qwen", choices=["qwen", "yi", "gemma", "llama", "default"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    
    vllm_infer(
        model_name_or_path=args.model_name_or_path,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        save_name=args.save_name,
        adapter_name_or_path=args.adapter_name_or_path,
        template=args.template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
