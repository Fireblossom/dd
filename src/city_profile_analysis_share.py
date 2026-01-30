#!/usr/bin/env python3
"""
City Identity Profile Analysis - Share of Voice Version
(Sums to 100% per city)
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
    # Try different font configurations for Chinese support
    # Heiti TC is good for Mac, SimHei for Windows/Linux usually
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'DejaVu Sans', 'Arial Unicode MS']
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
    # Short-form English labels used in CSV
    "Culture": "Cultural Practice",
    "Language": "Language Ability",
    "Community": "Community Acceptance",
    "Belonging": "Sense of Belonging",
    "Self-ID": "Self-Identification",
    "Legal/Admin": "Legal/Admin Status",
}

def load_all_platforms(csv_path: str) -> pd.DataFrame:
    """Load data from the unified CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Map CSV columns to internal script expectations
    # CSV header: id,platform,city,text,objective_criteria,subjective_criteria,openness,...
    # Script expects: 城市, 开放性1, and separate criteria
    
    # Rename for compatibility with existing logic
    df.rename(columns={
        "city": "城市",
        "objective_criteria": "ObjectiveCols", # Temporary
        "subjective_criteria": "SubjectiveCols", # Temporary
        "openness": "开放性1"
    }, inplace=True)
    
    # Filter target cities just in case
    # Map English city names in CSV to Chinese for the script's constants
    # Note: The CSV example shows "北京" (Chinese), so direct filtering works.
    df = df[df["城市"].isin(TARGET_CITIES)]
    
    return df


def parse_criteria(row: pd.Series) -> List[str]:
    """Parse criteria from a data row."""
    criteria = []
    
    # Check both objective and subjective columns
    for col in ["ObjectiveCols", "SubjectiveCols"]:
        val = row.get(col, "")
        if pd.isna(val) or str(val) == "nan" or str(val).strip() == "":
            continue
        
        # Criteria in CSV are already in English/Format: "Birth/Upbringing; Family Heritage"
        # We need to split by semicolon or check if mapping is needed
        
        # The CSV sample shows "Birth/Upbringing; Family Heritage"
        # Since our TAXONOMY is in English, we might be able to use them directly
        # But let's check against our CRITERIA_TAXONOMY values to be safe
        
        parts = str(val).split(";")
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Direct match check (since CSV seems to have English labels)
            is_valid = False
            for cat in CRITERIA_TAXONOMY.values():
                if part in cat:
                    criteria.append(part)
                    is_valid = True
                    break
            
            if not is_valid:
                # If not direct match, try the Chinese map (fallback)
                if part in CRITERIA_MAP:
                    criteria.append(CRITERIA_MAP[part])

    return list(set(criteria)) if criteria else []


def compute_city_profiles_row_normalized(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute row-normalized profiles: each row sums to 100%."""
    # Get all criteria
    all_criteria = []
    for cat in CRITERIA_TAXONOMY.values():
        all_criteria.extend(cat)
    
    profiles = {city: {c: 0 for c in all_criteria} for city in TARGET_CITIES}
    
    for _, row in df.iterrows():
        city = row["城市"]
        if city not in TARGET_CITIES:
            continue
        
        criteria = parse_criteria(row)
        
        for c in criteria:
            if c in profiles[city]:
                profiles[city][c] += 1
    
    # Normalize: divide each value by the row sum (total mentions in that city)
    for city in TARGET_CITIES:
        row_sum = sum(profiles[city].values())
        if row_sum > 0:
            for c in profiles[city]:
                profiles[city][c] = profiles[city][c] / row_sum
        else:
            print(f"Warning: No mentions found for {city}")
    
    return profiles


def generate_heatmap(profiles: Dict, output_path: str):
    """Generate city-criteria heatmap (row-normalized, sums to 100%)."""
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
        row = [profiles[city_cn].get(c, 0) * 100 for c in all_criteria]  # Convert to %
        data.append(row)
    
    data = np.array(data)
    
    # Sort columns by total sum (most mentioned criteria first)
    col_sums = data.sum(axis=0)
    sorted_indices = np.argsort(col_sums)[::-1]
    data = data[:, sorted_indices]
    all_criteria = [all_criteria[i] for i in sorted_indices]
    
    # Create heatmap (viridis is colorblind-friendly)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(all_criteria)))
    ax.set_xticklabels(all_criteria, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities, fontsize=12, rotation=0)
    
    # Add percentage values
    for i in range(len(cities)):
        for j in range(len(all_criteria)):
            val = data[i, j]
            color = 'white' if val < 15 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=9)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.set_ylabel('% of all mentions in city', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="City identity profile analysis - Share of Voice")
    parser.add_argument("--csv", default="dataset/unified_3platform.csv", help="Unified CSV data file")
    
    parser.add_argument("--output_dir", default="outputs/city_profiles", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    try:
        df = load_all_platforms(args.csv)
        print(f"Total samples loaded: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Compute profiles (row-normalized: each row sums to 100%)
    print("\nComputing city profiles (row-normalized)...")
    profiles = compute_city_profiles_row_normalized(df)
    
    # Generate outputs
    print("\nGenerating visualizations...")
    output_file = os.path.join(args.output_dir, "city_criteria_share_heatmap.png")
    generate_heatmap(profiles, output_file)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nShare of Voice Heatmap saved to: {output_file}")


if __name__ == "__main__":
    main()
