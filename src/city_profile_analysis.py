#!/usr/bin/env python3
"""
City Identity Profile Analysis

Generates detailed profiles of how urban identity is constructed
in each megacity (Beijing, Shanghai, Guangzhou, Shenzhen).

Outputs:
- City profile heatmaps
- Radar charts comparing cities
- Statistical summaries
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# Target cities
TARGET_CITIES = ["北京", "上海", "广州", "深圳"]
CITY_EN = {"北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen"}

# Identity criteria taxonomy (English, neutral terminology)
CRITERIA_TAXONOMY = {
    "Objective": [
        "Residence",
        "Birth/Upbringing", 
        "Legal/Admin Status",
        "Family Heritage",
        "Property/Assets",
        "Intra-city Region",
    ],
    "Socio-cultural": [
        "Self-Identification",
        "Sense of Belonging",
        "Language Ability",
        "Cultural Practice",
        "Community Acceptance",
    ],
}

# Map Chinese labels to English (corrected to match actual data)
CRITERIA_MAP = {
    # Objective criteria
    "当地居住状态": "Residence",
    "当地出生和成长": "Birth/Upbringing", 
    "获得法律认定": "Legal/Admin Status",
    "家族历史传承": "Family Heritage",
    "拥有当地资产": "Property/Assets",
    "城市内部地理片区归属": "Intra-city Region",
    # Subjective criteria
    "个人认定": "Self-Identification",
    "归属感": "Sense of Belonging",
    "语言能力": "Language Ability",
    "文化实践": "Cultural Practice",
    "被社群接纳": "Community Acceptance",
    # Legacy names (backwards compatibility)
    "当前居住状态": "Residence",
    "法律与行政认定": "Legal/Admin Status",
    "文化实践与知识": "Cultural Practice",
    # Short-form English labels used in unified CSV exports
    "Culture": "Cultural Practice",
    "Language": "Language Ability",
    "Community": "Community Acceptance",
    "Belonging": "Sense of Belonging",
    "Self-ID": "Self-Identification",
    "Legal/Admin": "Legal/Admin Status",
}

OPENNESS_MAP = {
    "开放": "Inclusive",
    "中性": "Neutral", 
    "保守": "Restrictive",
    # Already-normalized English labels
    "Inclusive": "Inclusive",
    "Neutral": "Neutral",
    "Restrictive": "Restrictive",
}


def load_unified_csv(csv_path: str) -> pd.DataFrame:
    """
    Load schema-preserving unified CSV shipped with this repository.

    Expected header:
      id,platform,city,text,objective_criteria,subjective_criteria,openness,...
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Unified CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Map CSV columns to internal expectations.
    df = df.rename(
        columns={
            "city": "城市",
            "objective_criteria": "客观认定1",
            "subjective_criteria": "主观认定1",
            "openness": "开放性1",
        }
    )

    # Unify separators so parse_criteria works (comma/semicolon).
    for col in ["客观认定1", "主观认定1"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.replace(";", ",", regex=False)

    # Filter to target cities.
    if "城市" in df.columns:
        df = df[df["城市"].isin(TARGET_CITIES)]

    return df


def load_all_platforms(
    douyin_path: str,
    wechat_path: str,
    unified_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load data from all platforms."""
    all_dfs = []
    
    # Load Douyin
    if os.path.exists(douyin_path):
        df = pd.read_excel(douyin_path)
        df.columns = [c.strip().replace(" ", "") for c in df.columns]
        df["platform"] = "Douyin"
        all_dfs.append(df)
        print(f"Loaded Douyin: {len(df)} samples")
    
    # Load WeChat
    if os.path.exists(wechat_path):
        df = pd.read_excel(wechat_path)
        df.columns = [c.strip().replace(" ", "") for c in df.columns]
        df["platform"] = "WeChat"
        all_dfs.append(df)
        print(f"Loaded WeChat: {len(df)} samples")

    # Review-friendly fallback: load unified CSV sample when raw spreadsheets are absent.
    if not all_dfs:
        if unified_csv_path:
            print(f"No spreadsheets found; falling back to unified CSV: {unified_csv_path}")
            return load_unified_csv(unified_csv_path)
        raise FileNotFoundError(
            "No input data found. Provide --douyin/--wechat spreadsheets or --unified_csv."
        )
    
    # Combine and filter to target cities
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[combined["城市"].isin(TARGET_CITIES)]
    
    return combined


def parse_criteria(row: pd.Series) -> List[str]:
    """Parse criteria from a data row."""
    criteria = []
    
    for col in ["客观认定1", "主观认定1"]:
        val = row.get(col, "")
        if pd.isna(val) or str(val) == "nan":
            continue
        
        parts = (
            str(val)
            .replace(";", ",")
            .split(",")
        )
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Already normalized to taxonomy values (e.g. from unified CSV)
            if part in CRITERIA_TAXONOMY["Objective"] or part in CRITERIA_TAXONOMY["Socio-cultural"]:
                criteria.append(part)
                continue
            if part in CRITERIA_MAP:
                criteria.append(CRITERIA_MAP[part])
            else:
                # Partial match
                for cn, en in CRITERIA_MAP.items():
                    if cn in part:
                        criteria.append(en)
                        break
    
    return list(set(criteria)) if criteria else []


def compute_city_profiles(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute criteria frequency profiles for each city."""
    # Get all criteria
    all_criteria = []
    for cat in CRITERIA_TAXONOMY.values():
        all_criteria.extend(cat)
    
    profiles = {city: {c: 0 for c in all_criteria} for city in TARGET_CITIES}
    city_counts = {city: 0 for city in TARGET_CITIES}
    
    for _, row in df.iterrows():
        city = row["城市"]
        if city not in TARGET_CITIES:
            continue
        
        city_counts[city] += 1
        criteria = parse_criteria(row)
        
        for c in criteria:
            if c in profiles[city]:
                profiles[city][c] += 1
    
    # Convert to percentages
    for city in TARGET_CITIES:
        total = city_counts[city]
        if total > 0:
            for c in profiles[city]:
                profiles[city][c] = profiles[city][c] / total
    
    return profiles


def compute_openness_by_city(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute openness distribution by city."""
    distributions = {city: {"Inclusive": 0, "Neutral": 0, "Restrictive": 0} for city in TARGET_CITIES}
    city_counts = {city: 0 for city in TARGET_CITIES}
    
    for _, row in df.iterrows():
        city = row["城市"]
        if city not in TARGET_CITIES:
            continue
        
        openness_cn = row.get("开放性1", "")
        if pd.isna(openness_cn):
            continue
        
        openness = OPENNESS_MAP.get(str(openness_cn).strip(), None)
        if openness:
            distributions[city][openness] += 1
            city_counts[city] += 1
    
    # Convert to percentages
    for city in TARGET_CITIES:
        total = city_counts[city]
        if total > 0:
            for o in distributions[city]:
                distributions[city][o] = distributions[city][o] / total
    
    return distributions


def generate_heatmap(profiles: Dict, output_path: str):
    """Generate city-criteria heatmap."""
    if not HAS_MATPLOTLIB:
        print("Skipping heatmap (matplotlib not available)")
        return
    
    # Prepare data matrix
    all_criteria = []
    for cat in CRITERIA_TAXONOMY.values():
        all_criteria.extend(cat)
    
    cities = [CITY_EN[c] for c in TARGET_CITIES]
    data = []
    for city_cn in TARGET_CITIES:
        row = [profiles[city_cn].get(c, 0) * 100 for c in all_criteria]
        data.append(row)
    
    data = np.array(data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(all_criteria)))
    ax.set_xticklabels(all_criteria, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities, fontsize=12)
    
    # Add values
    for i in range(len(cities)):
        for j in range(len(all_criteria)):
            val = data[i, j]
            color = 'white' if val > 25 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title('Urban Identity Criteria by City (%)', fontsize=14, pad=20)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.set_ylabel('Percentage', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {output_path}")


def generate_radar_chart(profiles: Dict, output_path: str):
    """Generate radar chart comparing city profiles."""
    if not HAS_MATPLOTLIB:
        print("Skipping radar chart (matplotlib not available)")
        return
    
    # Select top criteria for readability
    top_criteria = [
        "Family Heritage", "Language Ability", "Property/Assets",
        "Birth/Upbringing", "Intra-city Region", "Sense of Belonging"
    ]
    
    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(top_criteria), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    for i, city_cn in enumerate(TARGET_CITIES):
        values = [profiles[city_cn].get(c, 0) * 100 for c in top_criteria]
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=CITY_EN[city_cn], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_criteria, fontsize=11)
    ax.set_ylim(0, 50)
    ax.set_title('City Identity Profiles', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved: {output_path}")


def generate_report(
    profiles: Dict,
    openness: Dict,
    sample_counts: Dict,
    output_path: str
):
    """Generate markdown report."""
    lines = ["# City Identity Profiles Report\n"]
    
    # Summary
    lines.append("## Data Summary\n")
    lines.append("| City | Samples |")
    lines.append("|------|---------|")
    for city in TARGET_CITIES:
        lines.append(f"| {CITY_EN[city]} | {sample_counts.get(city, 0)} |")
    
    # Top criteria per city
    lines.append("\n## Key Identity Markers by City\n")
    
    for city_cn in TARGET_CITIES:
        city = CITY_EN[city_cn]
        profile = profiles[city_cn]
        
        # Sort by frequency
        sorted_criteria = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        top3 = [f"{c} ({v:.0%})" for c, v in sorted_criteria[:3] if v > 0]
        
        lines.append(f"**{city}**: {', '.join(top3)}\n")
    
    # Openness comparison
    lines.append("\n## Openness Tendency by City\n")
    lines.append("| City | Inclusive | Neutral | Restrictive |")
    lines.append("|------|-----------|---------|-------------|")
    
    for city_cn in TARGET_CITIES:
        o = openness[city_cn]
        lines.append(f"| {CITY_EN[city_cn]} | {o['Inclusive']:.0%} | {o['Neutral']:.0%} | {o['Restrictive']:.0%} |")
    
    # Detailed profiles table
    lines.append("\n## Full Criteria Distribution\n")
    
    all_criteria = []
    for cat in CRITERIA_TAXONOMY.values():
        all_criteria.extend(cat)
    
    header = "| Criterion | " + " | ".join(CITY_EN[c] for c in TARGET_CITIES) + " |"
    sep = "|" + "|".join(["---"] * (len(TARGET_CITIES) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    
    for criterion in all_criteria:
        row = f"| {criterion} |"
        for city in TARGET_CITIES:
            pct = profiles[city].get(criterion, 0)
            row += f" {pct:.0%} |"
        lines.append(row)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="City identity profile analysis")
    parser.add_argument("--douyin", default="dataset/抖音.xlsx", help="Douyin data file")
    parser.add_argument("--wechat", default="dataset/微信视频号.xlsx", help="WeChat data file")
    parser.add_argument(
        "--unified_csv",
        default="dataset/unified_3platform.csv",
        help="Unified CSV export (used when spreadsheets are not available)",
    )
    parser.add_argument("--output_dir", default="outputs/city_profiles", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_all_platforms(args.douyin, args.wechat, unified_csv_path=args.unified_csv)
    print(f"Total samples: {len(df)}")
    
    # Compute profiles
    print("\nComputing city profiles...")
    profiles = compute_city_profiles(df)
    
    print("Computing openness distributions...")
    openness = compute_openness_by_city(df)
    
    # Sample counts
    sample_counts = df["城市"].value_counts().to_dict()
    
    # Generate outputs
    print("\nGenerating visualizations...")
    generate_heatmap(profiles, os.path.join(args.output_dir, "city_criteria_heatmap.png"))
    generate_radar_chart(profiles, os.path.join(args.output_dir, "city_radar_chart.png"))
    
    # Generate report
    report_path = os.path.join(args.output_dir, "city_profiles_report.md")
    generate_report(profiles, openness, sample_counts, report_path)
    
    # Save raw data
    with open(os.path.join(args.output_dir, "city_profiles.json"), "w", encoding="utf-8") as f:
        json.dump({
            "profiles": profiles,
            "openness": openness,
            "sample_counts": sample_counts,
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("CITY PROFILE ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
