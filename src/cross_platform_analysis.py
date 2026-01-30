#!/usr/bin/env python3
"""
Cross-Platform Urban Identity Analysis

Compares identity construction patterns across:
- Xiaohongshu (RedNote)
- Douyin (TikTok China)
- WeChat Channels

Includes statistical tests (chi-square, IAA) for rigor.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Statistical tests will be skipped.")

try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Target cities (4 megacities)
TARGET_CITIES = ["北京", "上海", "广州", "深圳"]
CITY_EN = {"北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen"}

# Map Chinese labels to English (corrected to match actual data)
CRITERIA_MAP = {
    # Objective criteria (actual names from data)
    "当地居住状态": "Residence",
    "当地出生和成长": "Birth/Upbringing", 
    "获得法律认定": "Legal/Admin Status",
    "家族历史传承": "Family Heritage",
    "拥有当地资产": "Property/Assets",
    "城市内部地理片区归属": "Intra-city Region",
    # Subjective criteria (actual names from data)
    "个人认定": "Self-Identification",
    "归属感": "Sense of Belonging",
    "语言能力": "Language Ability",
    "文化实践": "Cultural Practice",
    "被社群接纳": "Community Acceptance",
    # Legacy names (for backwards compatibility)
    "当前居住状态": "Residence",
    "法律与行政认定": "Legal/Admin Status",
    "文化实践与知识": "Cultural Practice",
    "无": "None Mentioned",
}

OPENNESS_MAP = {
    "开放": "Inclusive",
    "中性": "Neutral", 
    "保守": "Restrictive",
}

# Xiaohongshu-specific label mappings (JSONL format)
XHS_CRITERIA_MAP = {
    "家族在当地的历史传承": "Family Heritage",
    "个人的法律与行政认定": "Legal/Admin Status",
    "当地的出生和成长": "Birth/Upbringing",
    "拥有当地的资产": "Property/Assets",
    "当前的居住状态": "Residence",
    "城市内部地理片区归属": "Intra-city Region",
    "个人认定": "Self-Identification",
    "个人的归属感": "Sense of Belonging",
    "个人的语言能力": "Language Ability",
    "个人的文化实践与知识": "Cultural Practice",
    "被社群接纳": "Community Acceptance",
}

XHS_OPENNESS_MAP = {
    "紧缩排斥": "Restrictive",
    "中性": "Neutral",
    "宽容开放": "Inclusive",
}

ALL_CRITERIA = list(set(CRITERIA_MAP.values()) - {"None Mentioned"})


def load_xiaohongshu(lf_data_dir: str) -> pd.DataFrame:
    """Load Xiaohongshu data from JSONL files (4 target cities only)."""
    records = []
    
    for split in ["train", "dev", "test"]:
        jsonl_path = os.path.join(lf_data_dir, f"labels_withphrase_{split}.jsonl")
        if not os.path.exists(jsonl_path):
            continue
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                
                # Parse output JSON
                try:
                    output = json.loads(d.get("output", "{}"))
                except:
                    continue
                
                # Extract city from input text
                input_text = d.get("input", "")
                city = None
                for cn in TARGET_CITIES:
                    if cn in input_text:
                        city = cn
                        break
                
                if not city:
                    continue
                
                # Parse criteria
                criteria_str = output.get("本地人判定标准", "")
                obj_criteria = []
                subj_criteria = []
                for part in criteria_str.split(";"):
                    part = part.strip()
                    for cn, en in XHS_CRITERIA_MAP.items():
                        if cn in part:
                            # Categorize as objective or subjective
                            if en in ["Family Heritage", "Legal/Admin Status", "Birth/Upbringing",
                                      "Property/Assets", "Residence", "Intra-city Region"]:
                                obj_criteria.append(cn)
                            else:
                                subj_criteria.append(cn)
                            break
                
                # Parse openness
                openness_cn = output.get("开放倾向", "")
                openness = "保守" if openness_cn == "紧缩排斥" else ("开放" if openness_cn == "宽容开放" else "中性")
                
                records.append({
                    "platform": "小红书",
                    "city": city,
                    "text": input_text,
                    "text_length": len(input_text),
                    "objective_criteria_1": ", ".join(obj_criteria),
                    "subjective_criteria_1": ", ".join(subj_criteria),
                    "openness_1": openness,
                    "objective_criteria_2": "",  # No second annotator for XHS
                    "subjective_criteria_2": "",
                    "openness_2": "",
                })
    
    return pd.DataFrame(records)


def load_platform_xlsx(filepath: str, platform_name: str) -> pd.DataFrame:
    """Load platform data from Excel file with dual annotator columns."""
    df = pd.read_excel(filepath)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    
    records = []
    for _, row in df.iterrows():
        city = row.get("城市", "")
        if city not in TARGET_CITIES:
            continue
        
        text = row.get("text", "")
        
        # First annotator
        obj1 = str(row.get("客观认定1", "")) if pd.notna(row.get("客观认定1")) else ""
        subj1 = str(row.get("主观认定1", "")) if pd.notna(row.get("主观认定1")) else ""
        open1 = str(row.get("开放性1", "")) if pd.notna(row.get("开放性1")) else ""
        
        # Second annotator
        obj2 = str(row.get("客观认定2", "")) if pd.notna(row.get("客观认定2")) else ""
        subj2 = str(row.get("主观认定2", "")) if pd.notna(row.get("主观认定2")) else ""
        open2 = str(row.get("开放性2", "")) if pd.notna(row.get("开放性2")) else ""
        
        records.append({
            "platform": platform_name,
            "city": city,
            "text": text,
            "text_length": len(str(text)),
            "objective_criteria_1": obj1,
            "subjective_criteria_1": subj1,
            "openness_1": open1,
            "objective_criteria_2": obj2,
            "subjective_criteria_2": subj2,
            "openness_2": open2,
        })
    
    return pd.DataFrame(records)


def parse_criteria(obj_str: str, subj_str: str) -> List[str]:
    """Parse criteria strings into list of standardized labels."""
    criteria = []
    
    for s in [obj_str, subj_str]:
        if not s or s == "nan":
            continue
        for part in s.split(","):
            part = part.strip()
            if part in CRITERIA_MAP:
                criteria.append(CRITERIA_MAP[part])
            else:
                for cn, en in CRITERIA_MAP.items():
                    if cn in part:
                        criteria.append(en)
                        break
    
    return list(set(criteria)) if criteria else ["None Mentioned"]


def compute_iaa(df: pd.DataFrame) -> Dict:
    """Compute Inter-Annotator Agreement (IAA) for both annotators."""
    results = {}
    
    if not HAS_SKLEARN:
        return {"error": "sklearn not available"}
    
    # Openness IAA
    open1 = df["openness_1"].fillna("").tolist()
    open2 = df["openness_2"].fillna("").tolist()
    
    # Filter out empty pairs
    valid_pairs = [(o1, o2) for o1, o2 in zip(open1, open2) if o1 and o2]
    
    if len(valid_pairs) > 10:
        labels1, labels2 = zip(*valid_pairs)
        try:
            kappa = cohen_kappa_score(labels1, labels2)
            agreement = sum(1 for a, b in valid_pairs if a == b) / len(valid_pairs)
            results["openness"] = {
                "cohen_kappa": round(kappa, 3),
                "percent_agreement": round(agreement * 100, 1),
                "n": len(valid_pairs),
            }
        except:
            results["openness"] = {"error": "computation failed"}
    
    # Criteria IAA (multi-label - compute per criterion)
    criteria_kappas = {}
    for criterion_en in ALL_CRITERIA:
        labels1 = []
        labels2 = []
        
        for _, row in df.iterrows():
            c1 = parse_criteria(row.get("objective_criteria_1", ""), row.get("subjective_criteria_1", ""))
            c2 = parse_criteria(row.get("objective_criteria_2", ""), row.get("subjective_criteria_2", ""))
            
            labels1.append(1 if criterion_en in c1 else 0)
            labels2.append(1 if criterion_en in c2 else 0)
        
        if sum(labels1) > 5 and sum(labels2) > 5:  # Need enough positive cases
            try:
                kappa = cohen_kappa_score(labels1, labels2)
                criteria_kappas[criterion_en] = round(kappa, 3)
            except:
                pass
    
    if criteria_kappas:
        results["criteria"] = criteria_kappas
        results["criteria_mean_kappa"] = round(np.mean(list(criteria_kappas.values())), 3)
    
    return results


def compute_chi_square_platform(df: pd.DataFrame) -> Dict:
    """Chi-square test for platform × openness."""
    if not HAS_SCIPY:
        return {"error": "scipy not available"}
    
    results = {}
    
    # Build contingency table: platform × openness
    platforms = df["platform"].unique()
    openness_cats = ["Inclusive", "Neutral", "Restrictive"]
    
    contingency = []
    for platform in platforms:
        row = []
        platform_df = df[df["platform"] == platform]
        for openness in openness_cats:
            count = sum(1 for _, r in platform_df.iterrows() 
                       if OPENNESS_MAP.get(r.get("openness_1", ""), "") == openness)
            row.append(count)
        contingency.append(row)
    
    contingency = np.array(contingency)
    
    if contingency.min() >= 5:  # Chi-square requirement
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        results["platform_openness"] = {
            "chi2": round(chi2, 2),
            "p_value": round(p, 4),
            "dof": dof,
            "significant": p < 0.05,
        }
    
    # Chi-square for each criterion: platform × has_criterion
    criteria_tests = {}
    for criterion in ALL_CRITERIA:
        contingency = []
        for platform in platforms:
            platform_df = df[df["platform"] == platform]
            has_criterion = sum(1 for _, r in platform_df.iterrows() 
                               if criterion in parse_criteria(
                                   r.get("objective_criteria_1", ""),
                                   r.get("subjective_criteria_1", "")))
            no_criterion = len(platform_df) - has_criterion
            contingency.append([has_criterion, no_criterion])
        
        contingency = np.array(contingency)
        if contingency.min() >= 5:
            chi2, p, dof, _ = stats.chi2_contingency(contingency)
            if p < 0.1:  # Only report marginally significant
                criteria_tests[criterion] = {
                    "chi2": round(chi2, 2),
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                }
    
    results["criteria_tests"] = criteria_tests
    
    return results


def compute_text_length_analysis(df: pd.DataFrame) -> Dict:
    """Analyze if text length differs by platform/openness."""
    results = {}
    
    # By platform
    platform_lengths = {}
    for platform in df["platform"].unique():
        lengths = df[df["platform"] == platform]["text_length"]
        platform_lengths[platform] = {
            "mean": round(lengths.mean(), 1),
            "std": round(lengths.std(), 1),
            "median": round(lengths.median(), 1),
        }
    results["by_platform"] = platform_lengths
    
    # T-test between platforms
    if HAS_SCIPY and len(df["platform"].unique()) == 2:
        platforms = df["platform"].unique()
        len1 = df[df["platform"] == platforms[0]]["text_length"]
        len2 = df[df["platform"] == platforms[1]]["text_length"]
        t, p = stats.ttest_ind(len1, len2)
        results["platform_ttest"] = {
            "t_statistic": round(t, 2),
            "p_value": round(p, 4),
            "significant": p < 0.05,
        }
    
    return results


def compute_city_profiles(df: pd.DataFrame) -> Dict:
    """Compute criteria distribution per city."""
    profiles = defaultdict(lambda: defaultdict(int))
    city_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        city = row["city"]
        city_counts[city] += 1
        criteria = parse_criteria(
            row.get("objective_criteria_1", ""),
            row.get("subjective_criteria_1", "")
        )
        for c in criteria:
            profiles[city][c] += 1
    
    # Convert to percentages
    for city in profiles:
        for c in profiles[city]:
            profiles[city][c] = profiles[city][c] / city_counts[city] if city_counts[city] > 0 else 0
    
    return dict(profiles), dict(city_counts)


def compute_platform_comparison(df: pd.DataFrame) -> Dict:
    """Compare criteria distribution across platforms."""
    platform_profiles = defaultdict(lambda: defaultdict(int))
    platform_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        platform = row["platform"]
        platform_counts[platform] += 1
        criteria = parse_criteria(
            row.get("objective_criteria_1", ""),
            row.get("subjective_criteria_1", "")
        )
        for c in criteria:
            platform_profiles[platform][c] += 1
    
    # Convert to percentages
    for platform in platform_profiles:
        for c in platform_profiles[platform]:
            platform_profiles[platform][c] = platform_profiles[platform][c] / platform_counts[platform]
    
    return dict(platform_profiles)


def compute_openness_by_platform(df: pd.DataFrame) -> Dict:
    """Compute openness distribution by platform."""
    openness_dist = defaultdict(lambda: defaultdict(int))
    platform_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        platform = row["platform"]
        openness = OPENNESS_MAP.get(row.get("openness_1", ""), "Unknown")
        if openness != "Unknown":
            openness_dist[platform][openness] += 1
            platform_counts[platform] += 1
    
    # Convert to percentages
    for platform in openness_dist:
        for o in openness_dist[platform]:
            openness_dist[platform][o] = openness_dist[platform][o] / platform_counts[platform]
    
    return dict(openness_dist)


def generate_report(
    city_profiles: Dict,
    city_counts: Dict,
    platform_profiles: Dict,
    openness_dist: Dict,
    sample_counts: Dict,
    iaa_results: Dict,
    chi_square_results: Dict,
    text_length_results: Dict,
    output_path: str
):
    """Generate markdown analysis report with statistical tests."""
    lines = ["# Cross-Platform Urban Identity Analysis\n"]
    
    # Data summary
    lines.append("## Data Summary\n")
    lines.append("| Platform | Samples |")
    lines.append("|----------|---------|")
    for platform, count in sample_counts.items():
        lines.append(f"| {platform} | {count} |")
    lines.append(f"| **Total** | **{sum(sample_counts.values())}** |")
    
    # IAA section
    lines.append("\n## Inter-Annotator Agreement (IAA)\n")
    if "openness" in iaa_results:
        o = iaa_results["openness"]
        lines.append(f"**Openness Tendency**: Cohen's κ = {o['cohen_kappa']}, "
                    f"Agreement = {o['percent_agreement']}% (n={o['n']})")
    if "criteria_mean_kappa" in iaa_results:
        lines.append(f"\n**Criteria Labels**: Mean κ = {iaa_results['criteria_mean_kappa']}")
        lines.append("\n| Criterion | Cohen's κ |")
        lines.append("|-----------|-----------|")
        for c, k in sorted(iaa_results.get("criteria", {}).items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {c} | {k} |")
    
    # City profiles
    lines.append("\n## City Identity Profiles\n")
    all_criteria = set()
    for city in city_profiles:
        all_criteria.update(city_profiles[city].keys())
    
    header = "| Criterion | " + " | ".join(f"{CITY_EN.get(c, c)} (n={city_counts.get(c, 0)})" for c in TARGET_CITIES) + " |"
    sep = "|" + "|".join(["---"] * (len(TARGET_CITIES) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    
    for criterion in sorted(all_criteria):
        if criterion == "None Mentioned":
            continue
        row = f"| {criterion} |"
        for city in TARGET_CITIES:
            pct = city_profiles.get(city, {}).get(criterion, 0)
            row += f" {pct:.1%} |"
        lines.append(row)
    
    # Platform comparison
    lines.append("\n## Platform Comparison\n")
    lines.append("| Criterion | Douyin | WeChat | Diff |")
    lines.append("|-----------|--------|--------|------|")
    
    for criterion in sorted(all_criteria):
        if criterion == "None Mentioned":
            continue
        dy = platform_profiles.get("抖音", {}).get(criterion, 0)
        wx = platform_profiles.get("微信视频号", {}).get(criterion, 0)
        diff = dy - wx
        diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
        lines.append(f"| {criterion} | {dy:.1%} | {wx:.1%} | {diff_str} |")
    
    # Chi-square results
    lines.append("\n## Statistical Tests\n")
    lines.append("### Platform × Openness (Chi-Square)\n")
    if "platform_openness" in chi_square_results:
        po = chi_square_results["platform_openness"]
        sig = "✓ Significant" if po["significant"] else "✗ Not significant"
        lines.append(f"χ² = {po['chi2']}, df = {po['dof']}, p = {po['p_value']} ({sig})")
    
    lines.append("\n### Platform × Criteria (Chi-Square, p < 0.1 only)\n")
    if "criteria_tests" in chi_square_results and chi_square_results["criteria_tests"]:
        lines.append("| Criterion | χ² | p-value | Significant |")
        lines.append("|-----------|-----|---------|-------------|")
        for c, test in sorted(chi_square_results["criteria_tests"].items(), key=lambda x: x[1]["p_value"]):
            sig = "✓" if test["significant"] else "△"
            lines.append(f"| {c} | {test['chi2']} | {test['p_value']} | {sig} |")
    else:
        lines.append("No significant differences found.\n")
    
    # Text length
    lines.append("\n### Text Length Analysis\n")
    if "by_platform" in text_length_results:
        lines.append("| Platform | Mean | Std | Median |")
        lines.append("|----------|------|-----|--------|")
        for p, s in text_length_results["by_platform"].items():
            lines.append(f"| {p} | {s['mean']} | {s['std']} | {s['median']} |")
    if "platform_ttest" in text_length_results:
        tt = text_length_results["platform_ttest"]
        sig = "significant" if tt["significant"] else "not significant"
        lines.append(f"\nT-test: t = {tt['t_statistic']}, p = {tt['p_value']} ({sig})")
    
    # Openness comparison
    lines.append("\n## Openness by Platform\n")
    lines.append("| Platform | Inclusive | Neutral | Restrictive |")
    lines.append("|----------|-----------|---------|-------------|")
    
    for platform in ["抖音", "微信视频号"]:
        inc = openness_dist.get(platform, {}).get("Inclusive", 0)
        neu = openness_dist.get(platform, {}).get("Neutral", 0)
        res = openness_dist.get(platform, {}).get("Restrictive", 0)
        lines.append(f"| {platform} | {inc:.1%} | {neu:.1%} | {res:.1%} |")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-platform urban identity analysis")
    parser.add_argument("--douyin", default="dataset/抖音.xlsx", help="Douyin Excel file")
    parser.add_argument("--wechat", default="dataset/微信视频号.xlsx", help="WeChat Channels Excel file")
    parser.add_argument("--xhs_dir", default="dataset/lf_data", help="Xiaohongshu JSONL directory")
    parser.add_argument("--include_xhs", action="store_true", help="Include Xiaohongshu data")
    parser.add_argument("--output_dir", default="outputs/cross_platform", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    def empty_platform_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "platform",
                "city",
                "text",
                "text_length",
                "objective_criteria_1",
                "subjective_criteria_1",
                "openness_1",
                "objective_criteria_2",
                "subjective_criteria_2",
                "openness_2",
            ]
        )
    
    # Load data
    if os.path.exists(args.douyin):
        print("Loading Douyin data...")
        douyin_df = load_platform_xlsx(args.douyin, "抖音")
        print(f"  Loaded {len(douyin_df)} samples")
    else:
        print(f"Warning: Douyin file not found: {args.douyin}")
        douyin_df = empty_platform_df()
    
    if os.path.exists(args.wechat):
        print("Loading WeChat Channels data...")
        wechat_df = load_platform_xlsx(args.wechat, "微信视频号")
        print(f"  Loaded {len(wechat_df)} samples")
    else:
        print(f"Warning: WeChat file not found: {args.wechat}")
        wechat_df = empty_platform_df()
    
    dfs = [douyin_df, wechat_df]
    sample_counts = {
        "抖音": len(douyin_df),
        "微信视频号": len(wechat_df),
    }
    
    # Load Xiaohongshu if requested
    if args.include_xhs:
        print("Loading Xiaohongshu data...")
        xhs_df = load_xiaohongshu(args.xhs_dir)
        print(f"  Loaded {len(xhs_df)} samples (4 cities)")
        dfs.append(xhs_df)
        sample_counts["小红书"] = len(xhs_df)

    if (len(douyin_df) == 0 and len(wechat_df) == 0) and not args.include_xhs:
        print("No input data available. Provide spreadsheet files or pass --include_xhs.")
        return
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal samples: {len(combined_df)}")
    
    # Compute IAA (only for platforms with dual annotators)
    dual_annotator_df = combined_df[combined_df["openness_2"] != ""]
    print(f"\nComputing Inter-Annotator Agreement (n={len(dual_annotator_df)})...")
    iaa_results = compute_iaa(dual_annotator_df) if len(dual_annotator_df) > 0 else {}
    
    # Compute chi-square tests
    print("Computing chi-square tests...")
    chi_square_results = compute_chi_square_platform(combined_df)
    
    # Compute text length analysis
    print("Analyzing text lengths...")
    text_length_results = compute_text_length_analysis(combined_df)
    
    # Compute profiles
    print("Computing city profiles...")
    city_profiles, city_counts = compute_city_profiles(combined_df)
    
    print("Computing platform comparison...")
    platform_profiles = compute_platform_comparison(combined_df)
    
    print("Computing openness distribution...")
    openness_dist = compute_openness_by_platform(combined_df)
    
    # Generate report
    report_path = os.path.join(args.output_dir, "cross_platform_report.md")
    generate_report(
        city_profiles, city_counts, platform_profiles, openness_dist,
        sample_counts, iaa_results, chi_square_results, text_length_results,
        report_path
    )
    
    # Save raw results (convert numpy types to Python types)
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    results = convert_types({
        "iaa": iaa_results,
        "chi_square": chi_square_results,
        "text_length": text_length_results,
        "city_profiles": city_profiles,
        "platform_profiles": platform_profiles,
        "openness_dist": openness_dist,
    })
    with open(os.path.join(args.output_dir, "statistical_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    combined_df.to_csv(os.path.join(args.output_dir, "combined_data.csv"), index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nCity distribution:")
    print(combined_df["city"].value_counts().to_string())
    print(f"\nPlatform distribution:")
    print(combined_df["platform"].value_counts().to_string())
    
    # Print key findings
    print("\n--- Key Statistical Findings ---")
    if "openness" in iaa_results:
        print(f"IAA (Openness): κ = {iaa_results['openness']['cohen_kappa']}")
    if "platform_openness" in chi_square_results:
        po = chi_square_results["platform_openness"]
        print(f"Platform×Openness: χ²={po['chi2']}, p={po['p_value']}")


if __name__ == "__main__":
    main()
