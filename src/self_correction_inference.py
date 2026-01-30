#!/usr/bin/env python3
"""
实验二：Self-Correction基线 - 两阶段自修正推理

流程:
1. 阶段1: 标准推理获取initial_pred
2. 阶段2: 拼接反思prompt，获取final_pred
3. 对比两阶段在Openness Tendency任务上的性能
"""
import argparse
import json
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# For vLLM inference
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Warning: vLLM not available. Using mock inference for testing.")


@dataclass
class SelfCorrectionConfig:
    """Configuration for self-correction inference."""
    model_path: str
    max_tokens_stage1: int = 256
    max_tokens_stage2: int = 128
    temperature: float = 0.1
    top_p: float = 0.9


# 反思Prompt
REFLECTION_PROMPT = """请反思刚才的判断：

你刚才的输出是：{initial_output}

请检查：
1. 你的"开放倾向"判断是否受到了文本中情绪化词汇的影响？
2. 该判断是否与文本的主要语义一致？
3. 是否存在"只要/就"等宽容表达，或"只有/才/必须"等排斥表达？

如果需要修正"开放倾向"的判断，请输出完整的修正后JSON；否则直接输出"无需修正"。"""


def load_test_data(jsonl_path: str) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_tone_from_output(output: str) -> Optional[str]:
    """Extract tone from model output."""
    if not output:
        return None
    
    # Try JSON parsing
    try:
        # Find JSON in output
        match = re.search(r'\{[^}]+\}', output)
        if match:
            obj = json.loads(match.group())
            tone = obj.get("开放倾向", obj.get("tone", None))
            if tone:
                return normalize_tone(tone)
    except:
        pass
    
    # Fallback: keyword matching
    output_clean = re.sub(r'\s+', '', output)
    if any(k in output_clean for k in ["宽容开放", "宽容", "开放", "包容"]):
        return "宽容开放"
    if any(k in output_clean for k in ["紧缩排斥", "排斥", "严格", "排外"]):
        return "紧缩排斥"
    if any(k in output_clean for k in ["中性", "中立"]):
        return "中性"
    
    return None


def normalize_tone(tone: str) -> str:
    """Normalize tone value."""
    tone = re.sub(r'\s+', '', tone)
    if any(k in tone for k in ["宽容", "开放", "包容"]):
        return "宽容开放"
    if any(k in tone for k in ["严格", "排斥", "紧缩", "排外"]):
        return "紧缩排斥"
    if any(k in tone for k in ["中性", "中立"]):
        return "中性"
    return tone


def build_stage1_messages(sample: Dict) -> List[Dict]:
    """Build messages for stage 1 (initial inference)."""
    return [
        {"role": "system", "content": sample.get("system", "你是一个标注助手。")},
        {"role": "user", "content": sample.get("instruction", "") + "\n" + sample.get("input", "")}
    ]


def build_stage2_messages(sample: Dict, initial_output: str) -> List[Dict]:
    """Build messages for stage 2 (reflection)."""
    stage1_messages = build_stage1_messages(sample)
    
    return stage1_messages + [
        {"role": "assistant", "content": initial_output},
        {"role": "user", "content": REFLECTION_PROMPT.format(initial_output=initial_output)}
    ]


class MockLLM:
    """Mock LLM for testing when vLLM is not available."""
    def generate(self, prompts, sampling_params):
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text})()]
        
        return [MockOutput('{"本地人判定标准": "测试", "开放倾向": "中性"}') for _ in prompts]


def run_inference_vllm(
    llm,
    samples: List[Dict],
    config: SelfCorrectionConfig
) -> List[Dict]:
    """Run two-stage inference with vLLM."""
    results = []
    
    # Stage 1: Initial inference
    print("Stage 1: Initial inference...")
    stage1_prompts = []
    for sample in samples:
        msgs = build_stage1_messages(sample)
        # Format as single prompt (simplified - real implementation should use chat template)
        prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
        stage1_prompts.append(prompt)
    
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens_stage1
    )
    
    stage1_outputs = llm.generate(stage1_prompts, sampling_params)
    initial_preds = [out.outputs[0].text for out in stage1_outputs]
    
    # Stage 2: Reflection
    print("Stage 2: Reflection...")
    stage2_prompts = []
    for sample, initial in zip(samples, initial_preds):
        msgs = build_stage2_messages(sample, initial)
        # Format as conversation
        prompt = ""
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            prompt += f"[{role}]: {content}\n\n"
        stage2_prompts.append(prompt)
    
    sampling_params_s2 = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens_stage2
    )
    
    stage2_outputs = llm.generate(stage2_prompts, sampling_params_s2)
    final_preds = [out.outputs[0].text for out in stage2_outputs]
    
    # Combine results
    for i, sample in enumerate(samples):
        initial = initial_preds[i]
        final = final_preds[i]
        
        # Check if correction was made
        needs_correction = "无需修正" not in final
        
        # Extract tones
        initial_tone = extract_tone_from_output(initial)
        if needs_correction:
            final_tone = extract_tone_from_output(final)
        else:
            final_tone = initial_tone
        
        results.append({
            "input": sample.get("input", ""),
            "gold_output": sample.get("output", ""),
            "initial_pred": initial,
            "final_pred": final if needs_correction else initial,
            "initial_tone": initial_tone,
            "final_tone": final_tone,
            "corrected": needs_correction,
            "_meta": sample.get("_meta", {})
        })
    
    return results


def evaluate_tones(results: List[Dict]) -> Dict:
    """Evaluate tone accuracy for initial and final predictions."""
    gold_tones = []
    initial_tones = []
    final_tones = []
    
    for r in results:
        # Extract gold tone
        gold_output = r.get("gold_output", "")
        try:
            obj = json.loads(gold_output)
            gold = normalize_tone(obj.get("开放倾向", "中性"))
        except:
            gold = "中性"
        
        gold_tones.append(gold)
        initial_tones.append(r.get("initial_tone") or "中性")
        final_tones.append(r.get("final_tone") or "中性")
    
    # Calculate accuracy
    initial_correct = sum(1 for g, p in zip(gold_tones, initial_tones) if g == p)
    final_correct = sum(1 for g, p in zip(gold_tones, final_tones) if g == p)
    
    n = len(results)
    correction_count = sum(1 for r in results if r.get("corrected"))
    
    # Track correction impact
    improved = 0
    degraded = 0
    for r, g in zip(results, gold_tones):
        if r.get("corrected"):
            initial_correct_flag = (r.get("initial_tone") == g)
            final_correct_flag = (r.get("final_tone") == g)
            if not initial_correct_flag and final_correct_flag:
                improved += 1
            elif initial_correct_flag and not final_correct_flag:
                degraded += 1
    
    return {
        "num_samples": n,
        "initial_accuracy": initial_correct / n if n > 0 else 0,
        "final_accuracy": final_correct / n if n > 0 else 0,
        "correction_count": correction_count,
        "correction_rate": correction_count / n if n > 0 else 0,
        "improved_by_correction": improved,
        "degraded_by_correction": degraded,
        "net_improvement": improved - degraded,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-correction inference for Openness Tendency")
    parser.add_argument("--test_file", required=True, help="Test JSONL file")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Model path")
    parser.add_argument("--out", default="outputs/self_correction_results.jsonl", help="Output file")
    parser.add_argument("--out_metrics", default="outputs/self_correction_metrics.json", help="Metrics output")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--mock", action="store_true", help="Use mock inference for testing")
    args = parser.parse_args()
    
    print(f"Loading test data from {args.test_file}...")
    samples = load_test_data(args.test_file)
    print(f"Loaded {len(samples)} samples")
    
    config = SelfCorrectionConfig(model_path=args.model)
    
    # Initialize LLM
    if args.mock or not HAS_VLLM:
        print("Using mock inference...")
        llm = MockLLM()
    else:
        print(f"Loading model {args.model}...")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    
    # Run inference
    print("Running two-stage inference...")
    results = run_inference_vllm(llm, samples, config)
    
    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to: {args.out}")
    
    # Evaluate
    metrics = evaluate_tones(results)
    print("\n=== Self-Correction Results ===")
    print(f"Initial Tone Accuracy: {metrics['initial_accuracy']:.3f}")
    print(f"Final Tone Accuracy:   {metrics['final_accuracy']:.3f}")
    print(f"Improvement:           {metrics['final_accuracy'] - metrics['initial_accuracy']:+.3f}")
    print(f"Correction Rate:       {metrics['correction_rate']:.1%}")
    print(f"Net Improvement:       {metrics['net_improvement']} samples")
    
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {args.out_metrics}")


if __name__ == "__main__":
    main()
