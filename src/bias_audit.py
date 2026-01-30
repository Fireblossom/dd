#!/usr/bin/env python3
"""
Experiment 4: Regional Bias Audit - Counterfactual Testing

Method:
1. Identify samples containing regional names
2. Replace with counterpart regions in the same city
3. Compare prediction consistency between original/replaced
"""
import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 城市内区域替换配对
REGION_PAIRS = {
    "shanghai": [
        ("浦东", "闵行"), ("静安", "普陀"), ("徐汇", "杨浦"),
        ("黄浦", "虹口"), ("长宁", "宝山"), ("浦西", "浦东"),
    ],
    "beijing": [
        ("海淀", "朝阳"), ("东城", "丰台"), ("西城", "石景山"),
        ("通州", "大兴"), ("昌平", "顺义"), ("城里", "郊区"),
    ],
    "guangzhou": [
        ("天河", "白云"), ("越秀", "荔湾"), ("海珠", "番禺"),
        ("老城区", "新城区"), ("老三区", "新区"),
    ],
    "shenzhen": [
        ("南山", "龙岗"), ("福田", "宝安"), ("罗湖", "龙华"),
        ("关内", "关外"), ("特区内", "特区外"),
    ],
}

# Build reverse mapping for easier lookup
REGION_TO_CITY = {}
for city, pairs in REGION_PAIRS.items():
    for r1, r2 in pairs:
        REGION_TO_CITY[r1] = city
        REGION_TO_CITY[r2] = city


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def infer_city(sample_id: str) -> str:
    """Infer city from sample ID."""
    parts = sample_id.lower().split("_")
    if parts:
        return parts[0]
    return "unknown"


def find_regions_in_text(text: str, city: str) -> List[Tuple[str, str]]:
    """Find region names in text and return (region, replacement) pairs."""
    if city not in REGION_PAIRS:
        return []
    
    found = []
    for r1, r2 in REGION_PAIRS[city]:
        if r1 in text:
            found.append((r1, r2))
        elif r2 in text:
            found.append((r2, r1))
    
    return found


def create_counterfactual(text: str, region_pairs: List[Tuple[str, str]]) -> str:
    """Create counterfactual version of text by replacing regions."""
    cf_text = text
    for original, replacement in region_pairs:
        cf_text = cf_text.replace(original, replacement)
    return cf_text


def extract_prediction(pred_text: str) -> Dict:
    """Extract labels and tone from prediction text."""
    result = {"labels": [], "tone": "中性"}
    
    if not pred_text:
        return result
    
    # Try JSON parsing
    try:
        match = re.search(r'\{[^}]+\}', pred_text)
        if match:
            obj = json.loads(match.group())
            labels = obj.get("本地人判定标准", obj.get("labels", ""))
            if isinstance(labels, str):
                labels = [l.strip() for l in labels.split(";") if l.strip()]
            result["labels"] = labels if isinstance(labels, list) else []
            result["tone"] = obj.get("开放倾向", obj.get("tone", "中性"))
    except:
        pass
    
    return result


def compute_consistency(pred1: Dict, pred2: Dict) -> Dict:
    """Compute consistency between two predictions."""
    # Label consistency: Jaccard similarity
    labels1 = set(pred1.get("labels", []))
    labels2 = set(pred2.get("labels", []))
    
    if not labels1 and not labels2:
        label_jaccard = 1.0
    elif not labels1 or not labels2:
        label_jaccard = 0.0
    else:
        intersection = len(labels1 & labels2)
        union = len(labels1 | labels2)
        label_jaccard = intersection / union if union > 0 else 0.0
    
    # Tone consistency: exact match
    tone_match = pred1.get("tone") == pred2.get("tone")
    
    return {
        "label_jaccard": label_jaccard,
        "tone_match": tone_match,
        "labels_changed": labels1 != labels2,
        "tone_changed": not tone_match,
    }


class BiasAuditPipeline:
    """Pipeline for running bias audit experiments."""
    
    def __init__(self, inference_fn=None):
        """
        Args:
            inference_fn: Function that takes a list of texts and returns predictions.
                         If None, uses mock inference for testing.
        """
        self.inference_fn = inference_fn or self._mock_inference
    
    def _mock_inference(self, samples: List[Dict]) -> List[str]:
        """Mock inference for testing."""
        return ['{"本地人判定标准": "测试", "开放倾向": "中性"}'] * len(samples)
    
    def identify_candidates(self, samples: List[Dict]) -> List[Dict]:
        """Identify samples containing regional terms."""
        candidates = []
        
        for sample in samples:
            sample_id = sample.get("_meta", {}).get("id", "")
            if not sample_id:
                # Try to get from other fields
                input_text = sample.get("input", "")
                match = re.search(r'(\w+)_\d+_\d+', input_text)
                if match:
                    sample_id = match.group(0)
            
            city = infer_city(sample_id)
            if city not in REGION_PAIRS:
                continue
            
            text = sample.get("input", "")
            regions = find_regions_in_text(text, city)
            
            if regions:
                candidates.append({
                    "sample": sample,
                    "city": city,
                    "regions": regions,
                    "original_text": text,
                    "counterfactual_text": create_counterfactual(text, regions),
                })
        
        return candidates
    
    def run_audit(self, candidates: List[Dict]) -> List[Dict]:
        """Run bias audit on candidate samples."""
        if not candidates:
            return []
        
        # Prepare original and counterfactual samples
        original_samples = []
        counterfactual_samples = []
        
        for c in candidates:
            sample = c["sample"].copy()
            original_samples.append(sample)
            
            cf_sample = sample.copy()
            cf_sample["input"] = c["counterfactual_text"]
            counterfactual_samples.append(cf_sample)
        
        # Run inference on both
        print(f"Running inference on {len(candidates)} original samples...")
        original_preds = self.inference_fn(original_samples)
        
        print(f"Running inference on {len(candidates)} counterfactual samples...")
        cf_preds = self.inference_fn(counterfactual_samples)
        
        # Compare results
        results = []
        for i, c in enumerate(candidates):
            orig_pred = extract_prediction(original_preds[i])
            cf_pred = extract_prediction(cf_preds[i])
            consistency = compute_consistency(orig_pred, cf_pred)
            
            results.append({
                "city": c["city"],
                "original_text": c["original_text"],
                "counterfactual_text": c["counterfactual_text"],
                "regions_replaced": c["regions"],
                "original_prediction": orig_pred,
                "counterfactual_prediction": cf_pred,
                "consistency": consistency,
            })
        
        return results


def compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate bias audit metrics."""
    if not results:
        return {"error": "No results to analyze"}
    
    n = len(results)
    
    # Overall consistency
    label_jaccards = [r["consistency"]["label_jaccard"] for r in results]
    tone_matches = [r["consistency"]["tone_match"] for r in results]
    
    # By city
    city_metrics = defaultdict(lambda: {"count": 0, "label_jaccard_sum": 0, "tone_match_count": 0})
    for r in results:
        city = r["city"]
        city_metrics[city]["count"] += 1
        city_metrics[city]["label_jaccard_sum"] += r["consistency"]["label_jaccard"]
        if r["consistency"]["tone_match"]:
            city_metrics[city]["tone_match_count"] += 1
    
    city_summary = {}
    for city, m in city_metrics.items():
        city_summary[city] = {
            "count": m["count"],
            "label_consistency": m["label_jaccard_sum"] / m["count"],
            "tone_consistency": m["tone_match_count"] / m["count"],
        }
    
    return {
        "total_samples": n,
        "overall_label_consistency": sum(label_jaccards) / n,
        "overall_tone_consistency": sum(tone_matches) / n,
        "samples_with_label_change": sum(1 for r in results if r["consistency"]["labels_changed"]),
        "samples_with_tone_change": sum(1 for r in results if r["consistency"]["tone_changed"]),
        "by_city": city_summary,
    }


def generate_report(metrics: Dict, results: List[Dict], output_path: str):
    """Generate markdown report."""
    lines = ["# Regional Bias Audit Report\n"]
    
    # Overall metrics
    lines.append("## Overall Results\n")
    lines.append(f"- **Total Samples**: {metrics['total_samples']}")
    lines.append(f"- **Label Consistency**: {metrics['overall_label_consistency']:.2%}")
    lines.append(f"- **Tone Consistency**: {metrics['overall_tone_consistency']:.2%}")
    lines.append(f"- **Samples with Label Change**: {metrics['samples_with_label_change']}")
    lines.append(f"- **Samples with Tone Change**: {metrics['samples_with_tone_change']}")
    
    # By city
    lines.append("\n## By City\n")
    lines.append("| City | Samples | Label Consistency | Tone Consistency |")
    lines.append("|------|---------|-------------------|------------------|")
    
    for city, m in metrics.get("by_city", {}).items():
        lines.append(f"| {city} | {m['count']} | {m['label_consistency']:.2%} | {m['tone_consistency']:.2%} |")
    
    # Examples of inconsistency
    inconsistent = [r for r in results if r["consistency"]["tone_changed"]]
    if inconsistent:
        lines.append("\n## Inconsistent Examples (Tone Changed)\n")
        for r in inconsistent[:5]:  # Show max 5 examples
            lines.append(f"### {r['city']}")
            lines.append(f"- **Original**: {r['original_text'][:100]}...")
            lines.append(f"- **Counterfactual**: {r['counterfactual_text'][:100]}...")
            lines.append(f"- **Original Tone**: {r['original_prediction']['tone']}")
            lines.append(f"- **Counterfactual Tone**: {r['counterfactual_prediction']['tone']}")
            lines.append("")
    
    # Conclusion
    lines.append("\n## Conclusion\n")
    if metrics['overall_tone_consistency'] > 0.95:
        lines.append("> ✓ Model shows high regional unbiasedness. Tone judgments are not affected by regional labels.")
    elif metrics['overall_tone_consistency'] > 0.85:
        lines.append("> △ Model is mostly regionally unbiased, but some judgments are influenced by regional labels.")
    else:
        lines.append("> ✗ Model shows significant regional bias. Further analysis and improvement needed.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Regional bias audit")
    parser.add_argument("--test_file", required=True, help="Test JSONL file")
    parser.add_argument("--out_dir", default="outputs/bias_audit", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock inference")
    args = parser.parse_args()
    
    print(f"Loading test data from {args.test_file}...")
    samples = load_jsonl(args.test_file)
    print(f"Loaded {len(samples)} samples")
    
    # Initialize pipeline
    pipeline = BiasAuditPipeline()
    
    # Identify candidates
    print("Identifying samples with regional terms...")
    candidates = pipeline.identify_candidates(samples)
    print(f"Found {len(candidates)} candidates with regional terms")
    
    if not candidates:
        print("No candidates found for bias audit. Check if regional terms exist in data.")
        return
    
    # Show candidate distribution
    city_counts = defaultdict(int)
    for c in candidates:
        city_counts[c["city"]] += 1
    print("Candidates by city:", dict(city_counts))
    
    # Run audit
    print("\nRunning bias audit...")
    results = pipeline.run_audit(candidates)
    
    # Compute metrics
    metrics = compute_aggregate_metrics(results)
    
    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(args.out_dir, "bias_audit_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to: {results_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.out_dir, "bias_audit_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate report
    report_path = os.path.join(args.out_dir, "bias_audit_report.md")
    generate_report(metrics, results, report_path)
    
    # Print summary
    print("\n=== Bias Audit Summary ===")
    print(f"Total samples tested: {metrics['total_samples']}")
    print(f"Label consistency: {metrics['overall_label_consistency']:.2%}")
    print(f"Tone consistency: {metrics['overall_tone_consistency']:.2%}")


if __name__ == "__main__":
    main()
