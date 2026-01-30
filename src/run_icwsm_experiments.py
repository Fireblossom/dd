#!/usr/bin/env python3
"""
ICWSM 2025 Supplementary Experiments - Unified Runner

Automatically runs all experiments and generates results.
Uses in-process vLLM inference with model reuse.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
from typing import Optional, List, Dict, Any

# Project path configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_INFER_SCRIPT = os.path.join(PROJECT_ROOT, "src", "vllm_infer.py")

# Default model configuration
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_TEMPLATE = "qwen"

# Global vLLM engine singleton for model reuse
_VLLM_ENGINE: Optional[Any] = None
_VLLM_TOKENIZER: Optional[Any] = None
_VLLM_MODEL_PATH: Optional[str] = None
_VLLM_ADAPTER_PATH: Optional[str] = None


def get_vllm_engine(model: str, adapter_path: Optional[str] = None):
    """Get or create the global vLLM engine (singleton pattern for model reuse)."""
    global _VLLM_ENGINE, _VLLM_TOKENIZER, _VLLM_MODEL_PATH, _VLLM_ADAPTER_PATH
    
    # Check if we need to reload (different model or adapter)
    need_reload = (
        _VLLM_ENGINE is None or
        _VLLM_MODEL_PATH != model or
        _VLLM_ADAPTER_PATH != adapter_path
    )
    
    if need_reload:
        # Clean up old engine if exists
        if _VLLM_ENGINE is not None:
            print("Unloading previous model...")
            del _VLLM_ENGINE
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        print(f"\n{'='*60}")
        print(f"Loading model: {model}")
        if adapter_path:
            print(f"Loading LoRA adapter: {adapter_path}")
        print(f"{'='*60}\n")
        
        # Import vLLM
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import torch
        
        # Load tokenizer
        _VLLM_TOKENIZER = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        # Auto-detect GPU count
        tensor_parallel_size = torch.cuda.device_count() or 1
        print(f"Using {tensor_parallel_size} GPU(s) with tensor parallelism")
        
        # Initialize vLLM engine
        engine_args = {
            "model": model,
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": 0.92,
            "max_model_len": 4096,
            "disable_log_stats": True,
            "enable_lora": adapter_path is not None,
        }
        
        _VLLM_ENGINE = LLM(**engine_args)
        _VLLM_MODEL_PATH = model
        _VLLM_ADAPTER_PATH = adapter_path
        print("Model loaded successfully!")
    else:
        print(f"Reusing loaded model: {model}")
    
    return _VLLM_ENGINE, _VLLM_TOKENIZER


def load_dataset_from_info(dataset_name: str, dataset_dir: str) -> List[Dict]:
    """Load dataset via dataset_info.json."""
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if dataset_name in info:
            file_name = info[dataset_name]["file_name"]
            jsonl_path = os.path.join(dataset_dir, file_name)
            data = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
    
    # Fallback: try loading dataset_name.jsonl directly
    jsonl_path = os.path.join(dataset_dir, f"{dataset_name}.jsonl")
    if os.path.exists(jsonl_path):
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    raise FileNotFoundError(f"Dataset not found: {dataset_name} in {dataset_dir}")


def build_prompt(sample: Dict, tokenizer, template: str = "qwen") -> str:
    """Build inference prompt."""
    system = sample.get("system", "")
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    
    # Combine instruction and input
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n{input_text}" if instruction else input_text
    
    # Build prompt based on template
    if template in ["qwen", "yi", "llama"]:
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
    else:
        # Default: simple concatenation
        if system:
            return f"{system}\n\n{user_content}\n\n"
        return f"{user_content}\n\n"


def run_vllm_inference(
    dataset: str,
    dataset_dir: str,
    save_name: str,
    model: str = DEFAULT_MODEL,
    template: str = DEFAULT_TEMPLATE,
    adapter_path: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> bool:
    """Run vLLM inference using the global engine (model reused across calls)."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Output: {save_name}")
    print(f"{'='*60}\n")
    
    try:
        # Get or create engine (reuses if same model)
        llm, tokenizer = get_vllm_engine(model, adapter_path)
        
        # Load dataset
        print(f"Loading dataset: {dataset} from {dataset_dir}")
        samples = load_dataset_from_info(dataset, dataset_dir)
        print(f"Total samples: {len(samples)}")
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            max_tokens=max_new_tokens,
            repetition_penalty=1.0,
        )
        
        # Set LoRA request
        lora_request = None
        if adapter_path:
            lora_request = LoRARequest("default", 1, adapter_path)
        
        # Batch inference
        all_results = []
        batch_size = 4096
        
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
        return True
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_experiment_1_sensitivity(args):
    """Experiment 1: Prompt Sensitivity Testing."""
    print("\n" + "="*70)
    print("Experiment 1: Prompt Sensitivity Testing")
    print("="*70)
    
    # Step 1: Generate variant datasets
    print("\n[Step 1] Generating prompt variant datasets...")
    sensitivity_dir = os.path.join(args.lf_data_dir, "sensitivity")
    cmd = [
        "python3", os.path.join(PROJECT_ROOT, "src", "prompt_sensitivity.py"),
        "--splits_dir", os.path.join(PROJECT_ROOT, "dataset", "splits"),
        "--outdir", sensitivity_dir,
        "--split", args.split,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    
    # Step 2: Run inference (3 variants)
    results_dir = os.path.join(args.output_dir, "sensitivity")
    os.makedirs(results_dir, exist_ok=True)
    
    for variant in ["A", "B", "C"]:
        print(f"\n[Step 2.{variant}] Inferencing variant {variant}...")
        success = run_vllm_inference(
            dataset=f"sensitivity_{variant}_only_test",
            dataset_dir=sensitivity_dir,
            save_name=os.path.join(results_dir, f"variant_{variant}_preds.jsonl"),
            model=args.model,
            template=args.template,
            adapter_path=args.adapter,
        )
        if not success:
            print(f"Warning: Variant {variant} inference failed")
    
    # Step 3: Analyze results
    print("\n[Step 3] Analyzing sensitivity results...")
    data_file = os.path.join(sensitivity_dir, "sensitivity_A_only_test.jsonl")
    cmd = [
        "python3", os.path.join(PROJECT_ROOT, "src", "analyze_sensitivity.py"),
        "--results_dir", results_dir,
        "--data_file", data_file,
        "--out", os.path.join(results_dir, "sensitivity_report.md"),
        "--out_json", os.path.join(results_dir, "sensitivity_metrics.json"),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    print(f"\n✓ Experiment 1 complete! Results: {results_dir}/sensitivity_report.md")


def run_experiment_2_self_correction(args):
    """Experiment 2: Self-Correction Baseline."""
    print("\n" + "="*70)
    print("Experiment 2: Self-Correction Baseline")
    print("="*70)
    
    results_dir = os.path.join(args.output_dir, "self_correction")
    os.makedirs(results_dir, exist_ok=True)
    
    test_file = os.path.join(args.lf_data_dir, "labels_only_test.jsonl")
    
    # Stage 1: Standard inference
    print("\n[Stage 1] Standard inference...")
    stage1_output = os.path.join(results_dir, "stage1_preds.jsonl")
    run_vllm_inference(
        dataset="labels_only_test",
        dataset_dir=args.lf_data_dir,
        save_name=stage1_output,
        model=args.model,
        template=args.template,
        adapter_path=args.adapter,
    )
    
    # Stage 2: Generate reflection data and inference
    print("\n[Stage 2] Generating reflection data...")
    stage2_dataset = generate_reflection_dataset(stage1_output, test_file, results_dir)
    
    print("\n[Stage 2] Reflection inference...")
    stage2_output = os.path.join(results_dir, "stage2_preds.jsonl")
    run_vllm_inference(
        dataset="self_correction_stage2",
        dataset_dir=results_dir,
        save_name=stage2_output,
        model=args.model,
        template=args.template,
        adapter_path=args.adapter,
        max_new_tokens=128,
    )
    
    # Analyze results
    print("\n[Step 3] Analyzing self-correction effect...")
    analyze_self_correction(stage1_output, stage2_output, test_file, results_dir)
    
    print(f"\n✓ Experiment 2 complete! Results: {results_dir}/")


def generate_reflection_dataset(stage1_preds: str, gold_file: str, output_dir: str) -> str:
    """Generate Stage 2 reflection dataset."""
    
    # Chinese reflection prompt (content kept in Chinese)
    REFLECTION_PROMPT = """请反思刚才的判断：

你刚才的输出是：{initial_output}

请检查：
1. 你的"开放倾向"判断是否受到了文本中情绪化词汇的影响？
2. 该判断是否与文本的主要语义一致？
3. 是否存在"只要/就"等宽容表达，或"只有/才/必须"等排斥表达？

如果需要修正"开放倾向"的判断，请输出修正后的JSON；否则直接输出"无需修正"。"""

    # Load stage1 predictions and gold data
    with open(stage1_preds, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    
    with open(gold_file, "r", encoding="utf-8") as f:
        golds = [json.loads(line) for line in f if line.strip()]
    
    # Generate stage2 dataset
    stage2_data = []
    for pred, gold in zip(preds, golds):
        initial_output = pred.get("predict", "")
        
        # Build reflection prompt
        entry = {
            "system": gold.get("system", "你是一个标注助手。"),
            "instruction": REFLECTION_PROMPT.format(initial_output=initial_output),
            "input": gold.get("input", ""),
            "output": "",  # No gold for reflection
            "_meta": {"initial_pred": initial_output}
        }
        stage2_data.append(entry)
    
    # Write stage2 dataset
    stage2_file = os.path.join(output_dir, "self_correction_stage2.jsonl")
    with open(stage2_file, "w", encoding="utf-8") as f:
        for entry in stage2_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Write dataset_info.json
    dataset_info = {
        "self_correction_stage2": {
            "file_name": "self_correction_stage2.jsonl",
            "columns": {"system": "system", "prompt": "instruction", "query": "input", "response": "output"}
        }
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"Generated: {stage2_file} ({len(stage2_data)} samples)")
    return stage2_file


def analyze_self_correction(stage1_file: str, stage2_file: str, gold_file: str, output_dir: str):
    """Analyze self-correction effect."""
    import re
    
    def extract_tone(text):
        if not text:
            return "中性"
        try:
            match = re.search(r'\{[^}]+\}', text)
            if match:
                obj = json.loads(match.group())
                return obj.get("开放倾向", "中性")
        except:
            pass
        if any(k in text for k in ["宽容", "开放"]):
            return "宽容开放"
        if any(k in text for k in ["排斥", "严格", "紧缩"]):
            return "紧缩排斥"
        return "中性"
    
    def extract_gold_tone(sample):
        try:
            obj = json.loads(sample.get("output", "{}"))
            return obj.get("开放倾向", "中性")
        except:
            return "中性"
    
    with open(stage1_file, "r", encoding="utf-8") as f:
        stage1 = [json.loads(line) for line in f if line.strip()]
    with open(stage2_file, "r", encoding="utf-8") as f:
        stage2 = [json.loads(line) for line in f if line.strip()]
    with open(gold_file, "r", encoding="utf-8") as f:
        golds = [json.loads(line) for line in f if line.strip()]
    
    n = min(len(stage1), len(stage2), len(golds))
    
    stage1_correct = 0
    stage2_correct = 0
    corrections = 0
    
    for i in range(n):
        gold_tone = extract_gold_tone(golds[i])
        stage1_tone = extract_tone(stage1[i].get("predict", ""))
        stage2_pred = stage2[i].get("predict", "")
        
        # Check if correction was made
        if "无需修正" in stage2_pred:
            stage2_tone = stage1_tone
        else:
            stage2_tone = extract_tone(stage2_pred)
            if stage2_tone != stage1_tone:
                corrections += 1
        
        if stage1_tone == gold_tone:
            stage1_correct += 1
        if stage2_tone == gold_tone:
            stage2_correct += 1
    
    metrics = {
        "num_samples": n,
        "stage1_accuracy": stage1_correct / n if n > 0 else 0,
        "stage2_accuracy": stage2_correct / n if n > 0 else 0,
        "improvement": (stage2_correct - stage1_correct) / n if n > 0 else 0,
        "correction_count": corrections,
        "correction_rate": corrections / n if n > 0 else 0,
    }
    
    # Save metrics
    with open(os.path.join(output_dir, "self_correction_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\n=== Self-Correction Results ===")
    print(f"Stage 1 Accuracy: {metrics['stage1_accuracy']:.3f}")
    print(f"Stage 2 Accuracy: {metrics['stage2_accuracy']:.3f}")
    print(f"Improvement: {metrics['improvement']:+.3f}")
    print(f"Corrections Made: {metrics['correction_count']} ({metrics['correction_rate']:.1%})")


def run_experiment_3_external_validity(args):
    """Experiment 3: External Validity Verification."""
    print("\n" + "="*70)
    print("Experiment 3: External Validity Verification")
    print("="*70)
    
    stats_file = os.path.join(PROJECT_ROOT, "dataset", "city_statistics.csv")
    if not os.path.exists(stats_file):
        print(f"Error: Please fill in {stats_file} first")
        return
    
    # Check if stats file has data
    with open(stats_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1 or not any("," in line and len(line.split(",")) > 1 and line.split(",")[1].strip() for line in lines[1:]):
        print(f"Error: {stats_file} has no data. Please fill in city statistics first")
        return
    
    results_dir = os.path.join(args.output_dir, "external_validity")
    os.makedirs(results_dir, exist_ok=True)
    
    # Use existing predictions if available
    pred_file = args.pred_file
    if not pred_file:
        # Look for existing predictions
        for candidate in [
            os.path.join(args.output_dir, "sensitivity", "variant_A_preds.jsonl"),
            os.path.join(PROJECT_ROOT, "outputs", "ft_clean_fixed", "qwen_labels_only", "generated_predictions.jsonl"),
        ]:
            if os.path.exists(candidate):
                pred_file = candidate
                break
    
    if not pred_file or not os.path.exists(pred_file):
        print("Error: Please run Experiment 1 first or provide --pred_file")
        return
    
    cmd = [
        "python3", os.path.join(PROJECT_ROOT, "src", "external_validity.py"),
        "--pred_file", pred_file,
        "--stats_file", stats_file,
        "--out_dir", results_dir,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    print(f"\n✓ Experiment 3 complete! Results: {results_dir}/")


def run_experiment_4_bias_audit(args):
    """Experiment 4: Regional Bias Audit."""
    print("\n" + "="*70)
    print("Experiment 4: Regional Bias Audit")
    print("="*70)
    
    results_dir = os.path.join(args.output_dir, "bias_audit")
    os.makedirs(results_dir, exist_ok=True)
    
    test_file = os.path.join(args.lf_data_dir, "labels_only_test.jsonl")
    
    # Step 1: Identify candidate samples and generate counterfactual data
    print("\n[Step 1] Identifying candidate samples...")
    from bias_audit import BiasAuditPipeline, load_jsonl, compute_aggregate_metrics, generate_report
    
    samples = load_jsonl(test_file)
    pipeline = BiasAuditPipeline()
    candidates = pipeline.identify_candidates(samples)
    
    if not candidates:
        print("No samples containing regional terms found")
        return
    
    print(f"Found {len(candidates)} candidate samples")
    
    # Step 2: Generate original and counterfactual datasets
    original_data = []
    counterfactual_data = []
    
    for c in candidates:
        sample = c["sample"]
        original_data.append(sample)
        
        cf_sample = sample.copy()
        cf_sample["input"] = c["counterfactual_text"]
        counterfactual_data.append(cf_sample)
    
    # Write datasets
    for name, data in [("original", original_data), ("counterfactual", counterfactual_data)]:
        path = os.path.join(results_dir, f"bias_{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Write dataset_info
    dataset_info = {
        "bias_original": {"file_name": "bias_original.jsonl", "columns": {"system": "system", "prompt": "instruction", "query": "input", "response": "output"}},
        "bias_counterfactual": {"file_name": "bias_counterfactual.jsonl", "columns": {"system": "system", "prompt": "instruction", "query": "input", "response": "output"}},
    }
    with open(os.path.join(results_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # Step 3: Run inference
    print("\n[Step 2] Inferencing original samples...")
    run_vllm_inference(
        dataset="bias_original",
        dataset_dir=results_dir,
        save_name=os.path.join(results_dir, "original_preds.jsonl"),
        model=args.model,
        template=args.template,
    )
    
    print("\n[Step 3] Inferencing counterfactual samples...")
    run_vllm_inference(
        dataset="bias_counterfactual",
        dataset_dir=results_dir,
        save_name=os.path.join(results_dir, "counterfactual_preds.jsonl"),
        model=args.model,
        template=args.template,
    )
    
    # Step 4: Analyze consistency
    print("\n[Step 4] Analyzing consistency...")
    analyze_bias_audit(candidates, results_dir)
    
    print(f"\n✓ Experiment 4 complete! Results: {results_dir}/")


def analyze_bias_audit(candidates, results_dir):
    """Analyze bias audit results."""
    import re
    from collections import defaultdict
    
    def extract_prediction(text):
        result = {"labels": [], "tone": "中性"}
        try:
            match = re.search(r'\{[^}]+\}', text or "")
            if match:
                obj = json.loads(match.group())
                labels = obj.get("本地人判定标准", "")
                if isinstance(labels, str):
                    labels = [l.strip() for l in labels.split(";") if l.strip()]
                result["labels"] = labels
                result["tone"] = obj.get("开放倾向", "中性")
        except:
            pass
        return result
    
    orig_file = os.path.join(results_dir, "original_preds.jsonl")
    cf_file = os.path.join(results_dir, "counterfactual_preds.jsonl")
    
    with open(orig_file, "r", encoding="utf-8") as f:
        orig_preds = [json.loads(line) for line in f if line.strip()]
    with open(cf_file, "r", encoding="utf-8") as f:
        cf_preds = [json.loads(line) for line in f if line.strip()]
    
    n = min(len(orig_preds), len(cf_preds), len(candidates))
    tone_matches = 0
    label_jaccards = []
    
    results = []
    for i in range(n):
        orig = extract_prediction(orig_preds[i].get("predict", ""))
        cf = extract_prediction(cf_preds[i].get("predict", ""))
        
        tone_match = orig["tone"] == cf["tone"]
        if tone_match:
            tone_matches += 1
        
        # Jaccard for labels
        s1, s2 = set(orig["labels"]), set(cf["labels"])
        if s1 or s2:
            jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 1.0
        else:
            jaccard = 1.0
        label_jaccards.append(jaccard)
        
        results.append({
            "city": candidates[i]["city"],
            "original_prediction": orig,
            "counterfactual_prediction": cf,
            "tone_match": tone_match,
            "label_jaccard": jaccard,
        })
    
    metrics = {
        "total_samples": n,
        "overall_tone_consistency": tone_matches / n if n > 0 else 0,
        "overall_label_consistency": sum(label_jaccards) / n if n > 0 else 0,
        "tone_changes": n - tone_matches,
    }
    
    # Save
    with open(os.path.join(results_dir, "bias_audit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(results_dir, "bias_audit_results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print("\n=== Bias Audit Results ===")
    print(f"Total samples: {n}")
    print(f"Tone consistency: {metrics['overall_tone_consistency']:.2%}")
    print(f"Label consistency: {metrics['overall_label_consistency']:.2%}")
    
    if metrics['overall_tone_consistency'] > 0.95:
        print("✓ Model shows high regional unbiasedness")
    else:
        print("△ Model shows some regional bias")


def main():
    parser = argparse.ArgumentParser(description="ICWSM 2025 Supplementary Experiments")
    parser.add_argument("--experiment", "-e", type=int, choices=[1, 2, 3, 4], 
                        help="Run specific experiment (1-4), or all if not specified")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Template name")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--output_dir", default="outputs/icwsm", help="Output directory")
    parser.add_argument("--lf_data_dir", default="dataset/lf_data", help="Data directory")
    parser.add_argument("--pred_file", default=None, help="Existing prediction file (for Exp 3)")
    parser.add_argument("--split", default="test", choices=["test", "all"],
                        help="Data scope: test (default ~200) or all (~1978)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    experiments = {
        1: ("Prompt Sensitivity", run_experiment_1_sensitivity),
        2: ("Self-Correction", run_experiment_2_self_correction),
        3: ("External Validity", run_experiment_3_external_validity),
        4: ("Bias Audit", run_experiment_4_bias_audit),
    }
    
    if args.experiment:
        name, func = experiments[args.experiment]
        print(f"\nRunning Experiment {args.experiment}: {name}")
        func(args)
    else:
        print("\nRunning all 4 experiments...")
        for exp_id, (name, func) in experiments.items():
            try:
                func(args)
            except Exception as e:
                print(f"Experiment {exp_id} failed: {e}")
    
    print("\n" + "="*70)
    print("All experiments complete!")
    print(f"Results directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
