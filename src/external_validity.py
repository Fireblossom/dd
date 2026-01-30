#!/usr/bin/env python3
"""
Experiment 3: External Validity Verification - City metrics correlation analysis

Analysis:
1. Compute city-level metrics from model predictions
2. Calculate correlation with real socioeconomic data
3. Generate scatter plot visualizations
"""
import argparse
import json
import os
import re
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using numpy for correlation.")


TARGET_CITIES = ["beijing", "shanghai", "guangzhou", "shenzhen"]
CITY_NAMES_CN = {
    "beijing": "北京",
    "shanghai": "上海",
    "guangzhou": "广州",
    "shenzhen": "深圳",
}

# 标签类别映射
LABEL_CATEGORIES = {
    "资产导向": ["客观标准|个人或家族在当地的资产与权益"],
    "语言导向": ["社会文化与心理标准|个人的语言能力"],
    "历史导向": ["客观标准|家族在当地的历史传承"],
    "行政导向": ["客观标准|个人的法律与行政认定"],
    "居住导向": ["客观标准|个人在当地的当前居住状态"],
    "文化导向": ["社会文化与心理标准|个人的文化实践与知识"],
}


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_city_statistics(csv_path: str) -> Dict[str, Dict]:
    """Load city statistics from CSV."""
    import csv
    stats = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city = row["city"].lower()
            stats[city] = {
                k: float(v) if v and v.replace(".", "").replace("-", "").isdigit() else v
                for k, v in row.items()
                if k != "city"
            }
    return stats


def infer_city(sample_id_or_input: str) -> str:
    """Infer city from sample ID or input text."""
    text = sample_id_or_input.lower()
    for city in TARGET_CITIES:
        if city in text:
            return city
    # Try to extract from sample ID format like "beijing_123_0"
    parts = text.split("_")
    if parts and parts[0] in TARGET_CITIES:
        return parts[0]
    return "unknown"


def extract_labels_from_pred(pred_text: str) -> List[str]:
    """Extract labels from prediction text."""
    if not pred_text:
        return []
    
    # Try JSON parsing
    try:
        match = re.search(r'\{[^}]+\}', pred_text)
        if match:
            obj = json.loads(match.group())
            labels = obj.get("本地人判定标准", obj.get("labels", ""))
            if isinstance(labels, list):
                return labels
            if isinstance(labels, str):
                return [l.strip() for l in labels.split(";") if l.strip()]
    except:
        pass
    
    # Fallback: split by semicolon
    return [l.strip() for l in pred_text.split(";") if l.strip()]


def extract_tone(pred_text: str) -> str:
    """Extract tone from prediction."""
    if not pred_text:
        return "中性"
    
    try:
        match = re.search(r'\{[^}]+\}', pred_text)
        if match:
            obj = json.loads(match.group())
            tone = obj.get("开放倾向", obj.get("tone", "中性"))
            return tone
    except:
        pass
    
    # Keyword matching
    if any(k in pred_text for k in ["宽容", "开放"]):
        return "宽容开放"
    if any(k in pred_text for k in ["排斥", "严格", "紧缩"]):
        return "紧缩排斥"
    return "中性"


def compute_city_metrics(predictions: List[Dict]) -> Dict[str, Dict]:
    """Compute city-level metrics from predictions."""
    city_data = defaultdict(lambda: {
        "total": 0,
        "label_counts": defaultdict(int),
        "tone_counts": defaultdict(int),
    })
    
    for pred in predictions:
        # Infer city
        sample_id = pred.get("_meta", {}).get("id", "")
        if not sample_id:
            sample_id = pred.get("input", "")
        city = infer_city(sample_id)
        
        if city not in TARGET_CITIES:
            continue
        
        city_data[city]["total"] += 1
        
        # Extract labels
        pred_text = pred.get("predict", pred.get("final_pred", ""))
        labels = extract_labels_from_pred(pred_text)
        for label in labels:
            city_data[city]["label_counts"][label] += 1
        
        # Extract tone
        tone = extract_tone(pred_text)
        city_data[city]["tone_counts"][tone] += 1
    
    # Compute ratios
    city_metrics = {}
    for city, data in city_data.items():
        total = data["total"]
        if total == 0:
            continue
        
        metrics = {
            "sample_count": total,
            "排外指数": data["tone_counts"].get("紧缩排斥", 0) / total,
            "开放指数": data["tone_counts"].get("宽容开放", 0) / total,
        }
        
        # Add category metrics
        for cat_name, cat_labels in LABEL_CATEGORIES.items():
            count = sum(data["label_counts"].get(l, 0) for l in cat_labels)
            metrics[cat_name] = count / total
        
        city_metrics[city] = metrics
    
    return city_metrics


def compute_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Pearson correlation and p-value."""
    if len(x) < 3:
        # Not enough data points for meaningful correlation
        r = np.corrcoef(x, y)[0, 1] if len(x) >= 2 else 0
        return r, 1.0  # p=1 indicates not significant
    
    if HAS_SCIPY:
        r, p = stats.pearsonr(x, y)
        return r, p
    else:
        r = np.corrcoef(x, y)[0, 1]
        return r, None


def generate_scatter_plots(
    city_metrics: Dict[str, Dict],
    city_stats: Dict[str, Dict],
    output_dir: str
):
    """Generate scatter plots for correlations."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define correlation pairs to plot
    correlation_pairs = [
        ("资产导向", "house_price_ratio", "房价收入比"),
        ("语言导向", "dialect_usage", "方言使用率"),
        ("排外指数", "migrant_ratio", "外来人口比例"),
        ("开放指数", "migrant_ratio", "外来人口比例"),
    ]
    
    results = []
    
    for model_metric, stat_key, stat_label in correlation_pairs:
        cities = []
        x_vals = []
        y_vals = []
        
        for city in TARGET_CITIES:
            if city not in city_metrics or city not in city_stats:
                continue
            
            model_val = city_metrics[city].get(model_metric)
            stat_val = city_stats[city].get(stat_key)
            
            if model_val is not None and stat_val is not None:
                cities.append(city)
                x_vals.append(stat_val)
                y_vals.append(model_val)
        
        if len(x_vals) < 2:
            print(f"Warning: Not enough data for {model_metric} vs {stat_key}")
            continue
        
        r, p = compute_correlation(x_vals, y_vals)
        
        results.append({
            "model_metric": model_metric,
            "stat_metric": stat_key,
            "pearson_r": r,
            "p_value": p,
            "n_cities": len(cities),
        })
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_vals, y_vals, s=150, alpha=0.7, c='steelblue')
        
        # Add city labels
        for city, x, y in zip(cities, x_vals, y_vals):
            ax.annotate(
                CITY_NAMES_CN.get(city, city),
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=12,
            )
        
        # Add trend line
        if len(x_vals) >= 2:
            z = np.polyfit(x_vals, y_vals, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(min(x_vals), max(x_vals), 100)
            ax.plot(x_line, p_line(x_line), "r--", alpha=0.5, label=f"r={r:.2f}")
        
        ax.set_xlabel(stat_label, fontsize=12)
        ax.set_ylabel(model_metric, fontsize=12)
        ax.set_title(f"{model_metric} vs {stat_label}\n(Pearson r={r:.2f})", fontsize=14)
        ax.legend()
        
        # Save plot
        filename = f"correlation_{model_metric}_{stat_key}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    return results


def generate_report(
    city_metrics: Dict[str, Dict],
    correlation_results: List[Dict],
    output_path: str
):
    """Generate markdown report."""
    lines = ["# External Validity Report\n"]
    
    # City metrics table
    lines.append("## City Model Metrics\n")
    lines.append("| City | Samples | Exclusivity | Openness | Asset-Oriented | Language-Oriented | History-Oriented |")
    lines.append("|------|---------|-------------|----------|----------------|-------------------|------------------|")
    
    for city in TARGET_CITIES:
        if city not in city_metrics:
            continue
        m = city_metrics[city]
        lines.append(
            f"| {CITY_NAMES_CN.get(city, city)} | {m['sample_count']} | "
            f"{m['排外指数']:.2%} | {m['开放指数']:.2%} | "
            f"{m.get('资产导向', 0):.2%} | {m.get('语言导向', 0):.2%} | "
            f"{m.get('历史导向', 0):.2%} |"
        )
    
    # Correlation results
    lines.append("\n## Correlation Analysis\n")
    lines.append("| Model Metric | Stat Metric | Pearson r | p-value | Cities |")
    lines.append("|--------------|-------------|-----------|---------|--------|")
    
    for cr in correlation_results:
        p_str = f"{cr['p_value']:.3f}" if cr['p_value'] is not None else "N/A"
        lines.append(
            f"| {cr['model_metric']} | {cr['stat_metric']} | "
            f"{cr['pearson_r']:.3f} | {p_str} | {cr['n_cities']} |"
        )
    
    # Key findings
    significant = [cr for cr in correlation_results if cr.get('p_value') and cr['p_value'] < 0.1]
    if significant:
        lines.append("\n## Key Findings\n")
        for cr in significant:
            direction = "positive" if cr['pearson_r'] > 0 else "negative"
            lines.append(f"- {cr['model_metric']} shows {direction} correlation with {cr['stat_metric']} (r={cr['pearson_r']:.2f})")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="External validity analysis")
    parser.add_argument("--pred_file", required=True, help="Prediction JSONL file")
    parser.add_argument("--stats_file", required=True, help="City statistics CSV")
    parser.add_argument("--out_dir", default="outputs/external_validity", help="Output directory")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.pred_file}...")
    predictions = load_jsonl(args.pred_file)
    print(f"Loaded {len(predictions)} predictions")
    
    print(f"Loading city statistics from {args.stats_file}...")
    city_stats = load_city_statistics(args.stats_file)
    print(f"Loaded statistics for {len(city_stats)} cities")
    
    # Compute city metrics
    print("Computing city-level metrics...")
    city_metrics = compute_city_metrics(predictions)
    
    for city, metrics in city_metrics.items():
        print(f"  {city}: {metrics['sample_count']} samples, 排外={metrics['排外指数']:.2%}")
    
    # Generate plots and compute correlations
    print("Generating correlation plots...")
    os.makedirs(args.out_dir, exist_ok=True)
    correlation_results = generate_scatter_plots(city_metrics, city_stats, args.out_dir)
    
    # Generate report
    report_path = os.path.join(args.out_dir, "external_validity_report.md")
    generate_report(city_metrics, correlation_results or [], report_path)
    
    # Save raw metrics
    metrics_path = os.path.join(args.out_dir, "city_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(city_metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\nDone! Check output directory for results.")


if __name__ == "__main__":
    main()
