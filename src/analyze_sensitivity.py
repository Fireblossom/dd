#!/usr/bin/env python3
"""
实验一：Prompt敏感性测试 - 分析3种Prompt变体的结果

分析内容:
1. 整体F1均值/标准差
2. 每城市核心发现一致性
3. 可视化输出
"""
import argparse
import json
import os
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

# Import evaluation functions from existing module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_multilabel import (
    load_jsonl, extract_gold_labels, extract_pred_labels, 
    build_label_space, to_multihot, extract_gold_tone, extract_pred_tone
)


TARGET_CITIES = ["beijing", "shanghai", "guangzhou", "shenzhen"]
VARIANTS = ["A", "B", "C"]

# 城市特征假设 (用于检验一致性)
CITY_CHARACTERISTICS = {
    "shanghai": {"expected_high": "社会文化与心理标准|个人的语言能力"},
    "shenzhen": {"expected_high": "客观标准|个人或家族在当地的资产与权益"},
    "guangzhou": {"expected_high": "社会文化与心理标准|个人的语言能力"},
    "beijing": {"expected_high": "客观标准|家族在当地的历史传承"},
}


def evaluate_variant(pred_file: str, data_file: str) -> Dict:
    """Evaluate a single variant's predictions."""
    preds = load_jsonl(pred_file)
    golds = load_jsonl(data_file)
    
    if len(preds) != len(golds):
        raise ValueError(f"Length mismatch: {len(preds)} vs {len(golds)}")
    
    gold_label_lists = [extract_gold_labels(g) for g in golds]
    label_space = build_label_space(gold_label_lists + [["未提及任何标准"]])
    label_to_id = {l: i for i, l in enumerate(label_space)}
    
    Y_true, Y_pred = [], []
    city_preds = defaultdict(list)
    city_golds = defaultdict(list)
    
    for i, (pred, gold) in enumerate(zip(preds, golds)):
        pred_text = pred.get("predict", "")
        gold_labels = gold_label_lists[i]
        pred_labels = extract_pred_labels(pred_text, label_space)
        
        y_t = to_multihot(gold_labels, label_to_id)
        y_p = to_multihot(pred_labels, label_to_id)
        Y_true.append(y_t)
        Y_pred.append(y_p)
        
        # Track by city
        meta = pred.get("_meta", {})
        city = meta.get("city", gold.get("_meta", {}).get("city", "unknown"))
        if not city or city == "unknown":
            sample_id = gold.get("input", "").split("_")[0] if "input" in gold else "unknown"
            city = sample_id.split("_")[0] if "_" in sample_id else "unknown"
        
        city_preds[city].append(pred_labels)
        city_golds[city].append(gold_labels)
    
    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)
    
    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    
    # Tone metrics
    gold_tones = [extract_gold_tone(g) for g in golds]
    pred_tones = [extract_pred_tone(p.get("predict", "")) for p in preds]
    valid_idx = [i for i, t in enumerate(gold_tones) if t]
    
    tone_acc = 0.0
    if valid_idx:
        matches = sum(1 for i in valid_idx if gold_tones[i] == pred_tones[i])
        tone_acc = matches / len(valid_idx)
    
    # Per-city label distribution
    city_label_dist = {}
    for city in TARGET_CITIES:
        if city not in city_preds:
            continue
        label_counts = defaultdict(int)
        total = 0
        for labels in city_preds[city]:
            for l in labels:
                if "未提及" not in l:
                    label_counts[l] += 1
                    total += 1
        if total > 0:
            city_label_dist[city] = {l: c / total for l, c in label_counts.items()}
        else:
            city_label_dist[city] = {}
    
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "tone_accuracy": float(tone_acc),
        "num_samples": len(preds),
        "city_label_distribution": city_label_dist,
        "label_space": label_space,
    }


def compute_city_characteristic_consistency(results: Dict[str, Dict]) -> Dict:
    """Check if city characteristics are consistent across variants."""
    consistency = {}
    
    for city, expected in CITY_CHARACTERISTICS.items():
        expected_label = expected["expected_high"]
        rankings = []
        
        for variant, res in results.items():
            dist = res.get("city_label_distribution", {}).get(city, {})
            if not dist:
                continue
            
            # Get rank of expected label
            sorted_labels = sorted(dist.items(), key=lambda x: -x[1])
            rank = next((i for i, (l, _) in enumerate(sorted_labels) if l == expected_label), -1)
            rankings.append((variant, rank, dist.get(expected_label, 0)))
        
        consistency[city] = {
            "expected_label": expected_label,
            "rankings": rankings,
            "is_consistent": all(r[1] <= 2 for r in rankings if r[1] >= 0),  # Top 3
        }
    
    return consistency


def generate_summary_table(results: Dict[str, Dict]) -> str:
    """Generate markdown summary table."""
    lines = ["## Prompt Sensitivity Results\n"]
    lines.append("| Variant | Micro-F1 | Macro-F1 | Tone Acc | Samples |")
    lines.append("|---------|----------|----------|----------|---------|")
    
    f1_values = []
    for variant in VARIANTS:
        if variant not in results:
            continue
        r = results[variant]
        lines.append(f"| {variant} | {r['micro_f1']:.3f} | {r['macro_f1']:.3f} | {r['tone_accuracy']:.3f} | {r['num_samples']} |")
        f1_values.append(r["micro_f1"])
    
    if len(f1_values) >= 2:
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        lines.append(f"\n**Mean Micro-F1**: {mean_f1:.3f} ± {std_f1:.3f}")
        lines.append(f"**Variance**: {std_f1 / mean_f1 * 100:.1f}%")
    
    return "\n".join(lines)


def generate_city_consistency_report(consistency: Dict) -> str:
    """Generate city characteristic consistency report."""
    lines = ["\n## City Characteristic Consistency\n"]
    
    for city, data in consistency.items():
        status = "✓ Consistent" if data["is_consistent"] else "✗ Inconsistent"
        lines.append(f"### {city.title()} ({status})")
        lines.append(f"Expected high: `{data['expected_label']}`")
        lines.append("| Variant | Rank | Proportion |")
        lines.append("|---------|------|------------|")
        for variant, rank, prop in data["rankings"]:
            rank_str = f"#{rank+1}" if rank >= 0 else "N/A"
            lines.append(f"| {variant} | {rank_str} | {prop:.2%} |")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt sensitivity results")
    parser.add_argument("--results_dir", required=True, help="Directory with variant prediction files")
    parser.add_argument("--data_file", required=True, help="Gold data JSONL file")
    parser.add_argument("--out", default="outputs/sensitivity_analysis.md", help="Output report path")
    parser.add_argument("--out_json", default="outputs/sensitivity_analysis.json", help="Output JSON path")
    args = parser.parse_args()
    
    results = {}
    
    for variant in VARIANTS:
        pred_file = os.path.join(args.results_dir, f"sensitivity_{variant}_only_test_preds.jsonl")
        if not os.path.exists(pred_file):
            # Try alternative naming
            pred_file = os.path.join(args.results_dir, f"variant_{variant}_preds.jsonl")
        
        if os.path.exists(pred_file):
            print(f"Evaluating variant {variant}...")
            results[variant] = evaluate_variant(pred_file, args.data_file)
        else:
            print(f"Warning: Prediction file not found for variant {variant}")
    
    if not results:
        print("No results to analyze. Please run inference first.")
        return
    
    # Compute consistency
    consistency = compute_city_characteristic_consistency(results)
    
    # Generate reports
    summary = generate_summary_table(results)
    consistency_report = generate_city_consistency_report(consistency)
    
    full_report = summary + consistency_report
    
    # Write outputs
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Report saved to: {args.out}")
    
    # Save JSON
    output_data = {
        "variants": results,
        "consistency": consistency,
        "summary": {
            "mean_micro_f1": float(np.mean([r["micro_f1"] for r in results.values()])),
            "std_micro_f1": float(np.std([r["micro_f1"] for r in results.values()])),
        }
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"JSON saved to: {args.out_json}")
    
    print("\n" + full_report)


if __name__ == "__main__":
    main()
