#!/usr/bin/env python3
"""
City Identity Profile Analysis - Three-Panel Comparison
Compares three normalization approaches:
1. Prevalence (original): % of comments mentioning each criterion
2. Row-normalized: each row sums to 100%
3. Column-normalized (Z-score): relative emphasis across cities
"""
import argparse
import os
from typing import Dict, List

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'DejaVu Sans', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# Target cities
TARGET_CITIES = ["北京", "上海", "广州", "深圳"]
CITY_EN = {"北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen"}

# Identity criteria taxonomy
CRITERIA_TAXONOMY = {
    "Objective": [
        "Residence", "Birth/Upbringing", "Legal/Admin Status",
        "Family Heritage", "Property/Assets", "Intra-city Region",
    ],
    "Socio-cultural": [
        "Self-Identification", "Sense of Belonging", "Language Ability",
        "Cultural Practice", "Community Acceptance",
    ],
}

# Map labels to English
CRITERIA_MAP = {
    "当地居住状态": "Residence", "当地出生和成长": "Birth/Upbringing",
    "获得法律认定": "Legal/Admin Status", "家族历史传承": "Family Heritage",
    "拥有当地资产": "Property/Assets", "城市内部地理片区归属": "Intra-city Region",
    "个人认定": "Self-Identification", "归属感": "Sense of Belonging",
    "语言能力": "Language Ability", "文化实践": "Cultural Practice",
    "被社群接纳": "Community Acceptance",
    "当前居住状态": "Residence", "法律与行政认定": "Legal/Admin Status",
    "文化实践与知识": "Cultural Practice",
    "Culture": "Cultural Practice", "Language": "Language Ability",
    "Community": "Community Acceptance", "Belonging": "Sense of Belonging",
    "Self-ID": "Self-Identification", "Legal/Admin": "Legal/Admin Status",
}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.rename(columns={
        "city": "城市",
        "objective_criteria": "ObjectiveCols",
        "subjective_criteria": "SubjectiveCols",
    }, inplace=True)
    return df[df["城市"].isin(TARGET_CITIES)]


def parse_criteria(row: pd.Series) -> List[str]:
    criteria = []
    for col in ["ObjectiveCols", "SubjectiveCols"]:
        val = row.get(col, "")
        if pd.isna(val) or str(val).strip() == "":
            continue
        for part in str(val).split(";"):
            part = part.strip()
            if not part:
                continue
            # Direct match
            for cat in CRITERIA_TAXONOMY.values():
                if part in cat:
                    criteria.append(part)
                    break
            else:
                if part in CRITERIA_MAP:
                    criteria.append(CRITERIA_MAP[part])
    return list(set(criteria))


def compute_raw_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Get raw counts for each city-criterion pair."""
    all_criteria = []
    for cat in CRITERIA_TAXONOMY.values():
        all_criteria.extend(cat)
    
    counts = {city: {c: 0 for c in all_criteria} for city in TARGET_CITIES}
    city_sample_counts = {city: 0 for city in TARGET_CITIES}
    
    for _, row in df.iterrows():
        city = row["城市"]
        if city not in TARGET_CITIES:
            continue
        city_sample_counts[city] += 1
        for c in parse_criteria(row):
            if c in counts[city]:
                counts[city][c] += 1
    
    return counts, city_sample_counts


def generate_comparison_figure(df: pd.DataFrame, output_path: str):
    """Generate 3-panel comparison figure."""
    if not HAS_MATPLOTLIB:
        print("Skipping (matplotlib not available)")
        return
    
    all_criteria = []
    for cat in CRITERIA_TAXONOMY.values():
        all_criteria.extend(cat)
    
    cities = [CITY_EN[c] for c in TARGET_CITIES]
    counts, city_sample_counts = compute_raw_counts(df)
    
    # --- Panel 1: Prevalence (% of comments) ---
    data1 = []
    for city_cn in TARGET_CITIES:
        n = city_sample_counts[city_cn]
        row = [(counts[city_cn][c] / n * 100) if n > 0 else 0 for c in all_criteria]
        data1.append(row)
    data1 = np.array(data1)
    
    # --- Panel 2: Row-normalized (each row sums to 100%) ---
    data2 = []
    for city_cn in TARGET_CITIES:
        row_sum = sum(counts[city_cn].values())
        row = [(counts[city_cn][c] / row_sum * 100) if row_sum > 0 else 0 for c in all_criteria]
        data2.append(row)
    data2 = np.array(data2)
    
    # --- Panel 3: Column-normalized (Z-score) ---
    # Use prevalence as base, then Z-score by column
    col_means = data1.mean(axis=0)
    col_stds = data1.std(axis=0)
    col_stds[col_stds == 0] = 1
    data3 = (data1 - col_means) / col_stds
    
    # Sort all panels by total prevalence
    col_sums = data1.sum(axis=0)
    sorted_indices = np.argsort(col_sums)[::-1]
    data1 = data1[:, sorted_indices]
    data2 = data2[:, sorted_indices]
    data3 = data3[:, sorted_indices]
    sorted_criteria = [all_criteria[i] for i in sorted_indices]
    
    # Create figure with 3 vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Common colormap for panels 1 and 2 (viridis)
    vmax_12 = max(data1.max(), data2.max())
    
    # Panel 1: Prevalence
    ax = axes[0]
    im1 = ax.imshow(data1, cmap='YlGnBu', aspect='auto', vmin=0, vmax=vmax_12)
    ax.set_xticks(range(len(sorted_criteria)))
    ax.set_xticklabels(sorted_criteria, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities, fontsize=10)
    ax.set_title('(A) Prevalence\n(% of comments)', fontsize=11, fontweight='bold')
    for i in range(len(cities)):
        for j in range(len(sorted_criteria)):
            val = data1[i, j]
            color = 'white' if val > 25 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=7)
    
    # Panel 2: Row-normalized
    ax = axes[1]
    im2 = ax.imshow(data2, cmap='YlGnBu', aspect='auto', vmin=0, vmax=vmax_12)
    ax.set_xticks(range(len(sorted_criteria)))
    ax.set_xticklabels(sorted_criteria, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities, fontsize=10)
    ax.set_title('(B) Row-Normalized\n(each row sums to 100%)', fontsize=11, fontweight='bold')
    for i in range(len(cities)):
        for j in range(len(sorted_criteria)):
            val = data2[i, j]
            color = 'white' if val > 25 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=7)
    
    # Panel 3: Z-score (diverging colormap)
    ax = axes[2]
    vmax3 = np.abs(data3).max()
    im3 = ax.imshow(data3, cmap='coolwarm', aspect='auto', vmin=-vmax3, vmax=vmax3)
    ax.set_xticks(range(len(sorted_criteria)))
    ax.set_xticklabels(sorted_criteria, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities, fontsize=10)
    ax.set_title('(C) Column-Normalized (Z-score)\n(relative emphasis)', fontsize=11, fontweight='bold')
    for i in range(len(cities)):
        for j in range(len(sorted_criteria)):
            val = data3[i, j]
            color = 'white' if abs(val) > 1.2 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=7)
    
    # Colorbars
    fig.colorbar(im1, ax=axes[0], shrink=0.6, label='%')
    fig.colorbar(im2, ax=axes[1], shrink=0.6, label='%')
    fig.colorbar(im3, ax=axes[2], shrink=0.6, label='Z-score')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {output_path}")


def generate_raw_counts_figure(df: pd.DataFrame, output_path: str):
    """Generate heatmap with raw counts (absolute numbers)."""
    if not HAS_MATPLOTLIB:
        print("Skipping (matplotlib not available)")
        return
    
    all_criteria = []
    for cat in CRITERIA_TAXONOMY.values():
        all_criteria.extend(cat)
    
    cities = [CITY_EN[c] for c in TARGET_CITIES]
    counts, city_sample_counts = compute_raw_counts(df)
    
    # Build data matrix with raw counts
    data = []
    for city_cn in TARGET_CITIES:
        row = [counts[city_cn][c] for c in all_criteria]
        data.append(row)
    data = np.array(data)
    
    # Sort columns by total count
    col_sums = data.sum(axis=0)
    sorted_indices = np.argsort(col_sums)[::-1]
    data = data[:, sorted_indices]
    sorted_criteria = [all_criteria[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(data, cmap='YlGnBu', aspect='auto')
    
    ax.set_xticks(range(len(sorted_criteria)))
    ax.set_xticklabels(sorted_criteria, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels([f"{c} (N={city_sample_counts[TARGET_CITIES[i]]})" for i, c in enumerate(cities)], fontsize=10)
    
    # Add count values
    for i in range(len(cities)):
        for j in range(len(sorted_criteria)):
            val = data[i, j]
            color = 'white' if val > 150 else 'black'
            ax.text(j, i, f'{int(val)}', ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title('Raw Counts (Absolute Numbers)', fontsize=12, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Raw counts figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/unified_3platform.csv")
    parser.add_argument("--output_dir", default="outputs/city_profiles")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} samples")
    
    # Generate comparison figure
    output_file = os.path.join(args.output_dir, "normalization_comparison.png")
    generate_comparison_figure(df, output_file)
    
    # Generate raw counts figure
    output_file2 = os.path.join(args.output_dir, "raw_counts_heatmap.png")
    generate_raw_counts_figure(df, output_file2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
