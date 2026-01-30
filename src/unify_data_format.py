#!/usr/bin/env python3
"""
Unify data format across three platforms (小红书, 抖音, 微信视频号).

Output format (JSONL):
{
    "id": "platform_city_index",
    "platform": "小红书" | "抖音" | "微信视频号",
    "city": "北京" | "上海" | "广州" | "深圳",
    "text": "原始文本",
    "objective_criteria": ["标准1", "标准2", ...],
    "subjective_criteria": ["标准1", "标准2", ...],
    "openness": "Inclusive" | "Neutral" | "Restrictive",
    "annotator": 1 | 2 | "merged",
    "split": "train" | "dev" | "test"
}
"""

import json
import pandas as pd
import os
import argparse
from typing import List, Dict


TARGET_CITIES = ["北京", "上海", "广州", "深圳"]

# Complete criteria mapping covering ALL label variants from both platforms
CRITERIA_MAP = {
    # ===== Objective Criteria =====
    # XHS long format
    "客观标准|个人在当地的当前居住状态": "Residence",
    "客观标准|个人在当地的出生和成长经历": "Birth/Upbringing",
    "客观标准|个人的法律与行政认定": "Legal/Admin",
    "客观标准|家族在当地的历史传承": "Family Heritage",
    "客观标准|个人或家族在当地的资产与权益": "Property/Assets",
    "客观标准|城市内部地理片区归属": "Intra-city Region",
    # Douyin/WeChat short format
    "当地居住状态": "Residence",
    "当地出生和成长": "Birth/Upbringing",
    "获得法律认定": "Legal/Admin",
    "家族历史传承": "Family Heritage",
    "拥有当地资产": "Property/Assets",
    "城市内部地理片区归属": "Intra-city Region",
    
    # ===== Subjective Criteria =====
    # XHS long format
    "社会文化与心理标准|个人认定": "Self-ID",
    "社会文化与心理标准|个人的归属感": "Belonging",
    "社会文化与心理标准|个人的语言能力": "Language",
    "社会文化与心理标准|个人的文化实践与知识": "Culture",
    "社会文化与心理标准|个人被社群接纳": "Community",
    # Douyin/WeChat short format
    "个人认定": "Self-ID",
    "归属感": "Belonging",
    "语言能力": "Language",
    "文化实践": "Culture",
    "被社群接纳": "Community",
    
    # ===== Special labels (both mean empty/no criteria) =====
    # "以上标准皆不符合|未提及任何标准" - empty, skip
    # "以上标准皆不符合|提及了其它标准" - empty, skip
}

OPENNESS_MAP = {
    # XHS format
    "紧缩排斥": "Restrictive",
    "中性": "Neutral",
    "宽容开放": "Inclusive",
    # Douyin/WeChat format
    "保守": "Restrictive",
    "开放": "Inclusive",
}

OBJECTIVE_CRITERIA_EN = ["Residence", "Birth/Upbringing", "Legal/Admin", 
                         "Family Heritage", "Property/Assets", "Intra-city Region"]
SUBJECTIVE_CRITERIA_EN = ["Self-ID", "Belonging", "Language", "Culture", "Community"]


def parse_and_map_criteria(criteria_str: str) -> tuple:
    """Parse criteria string and return (objective, subjective) lists in English."""
    if pd.isna(criteria_str) or not str(criteria_str).strip():
        return [], []
    
    objective = []
    subjective = []
    
    for part in str(criteria_str).split(","):
        part = part.strip()
        if part in CRITERIA_MAP:
            en = CRITERIA_MAP[part]
            if en in OBJECTIVE_CRITERIA_EN:
                objective.append(en)
            elif en in SUBJECTIVE_CRITERIA_EN:
                subjective.append(en)
    
    return objective, subjective


def load_xiaohongshu(jsonl_dir: str, split: str = "all", filter_empty: bool = True) -> List[Dict]:
    """Load Xiaohongshu data from JSONL files.
    
    Args:
        filter_empty: If True, skip samples with no criteria labels (以上皆非)
    """
    records = []
    skipped_empty = 0
    splits = ["train", "dev", "test"] if split == "all" else [split]
    
    for s in splits:
        jsonl_path = os.path.join(jsonl_dir, f"labels_withphrase_{s}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found")
            continue
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    output = json.loads(d.get("output", "{}"))
                    input_text = d.get("input", "")
                    
                    # Remove "文本：" prefix if present
                    if input_text.startswith("文本："):
                        input_text = input_text[3:]
                    
                    # Extract city
                    city = None
                    for c in TARGET_CITIES:
                        if c in input_text:
                            city = c
                            break
                    if not city:
                        continue
                    
                    # Parse criteria
                    criteria_str = output.get("本地人判定标准", "")
                    objective, subjective = [], []
                    for part in criteria_str.split(";"):
                        part = part.strip()
                        if part in CRITERIA_MAP:
                            en = CRITERIA_MAP[part]
                            if en in OBJECTIVE_CRITERIA_EN:
                                objective.append(en)
                            elif en in SUBJECTIVE_CRITERIA_EN:
                                subjective.append(en)
                    
                    # Filter empty criteria if requested
                    if filter_empty and not objective and not subjective:
                        skipped_empty += 1
                        continue
                    
                    # Parse openness
                    openness_cn = output.get("开放倾向", "中性")
                    openness = OPENNESS_MAP.get(openness_cn, "Neutral")
                    
                    records.append({
                        "id": f"xhs_{s}_{idx}",
                        "platform": "小红书",
                        "city": city,
                        "text": input_text,
                        "objective_criteria": objective,
                        "subjective_criteria": subjective,
                        "openness": openness,
                        "annotator": "merged",
                        "split": s,
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
    
    print(f"小红书: loaded {len(records)} samples (skipped {skipped_empty} empty)")
    return records


def load_douyin_wechat(excel_path: str, platform_name: str) -> List[Dict]:
    """Load Douyin or WeChat data from Excel file.
    
    Merge strategy:
    - Criteria: Union of both annotators (combine all labels)
    - Openness: If agree, use agreed value; if disagree, use annotator 1's value
    """
    if not os.path.exists(excel_path):
        print(f"Warning: {excel_path} not found")
        return []
    
    df = pd.read_excel(excel_path)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    
    records = []
    skipped = 0
    
    for idx, row in df.iterrows():
        city = row.get("城市", "")
        if city not in TARGET_CITIES:
            continue
        
        text = row.get("text", "")
        
        # Get both annotators' labels
        obj1_str = row.get("客观认定1", "")
        subj1_str = row.get("主观认定1", "")
        open1_cn = str(row.get("开放性1", "")).strip()
        
        obj2_str = row.get("客观认定2", "")
        subj2_str = row.get("主观认定2", "")
        open2_cn = str(row.get("开放性2", "")).strip()
        
        # Skip if annotator 1 has no openness (must have at least one valid annotation)
        if pd.isna(open1_cn) or not open1_cn:
            skipped += 1
            continue
        
        # Parse criteria for both annotators
        obj1, _ = parse_and_map_criteria(obj1_str)
        _, subj1 = parse_and_map_criteria(subj1_str)
        obj2, _ = parse_and_map_criteria(obj2_str)
        _, subj2 = parse_and_map_criteria(subj2_str)
        
        # Merge criteria: UNION (combine unique labels from both)
        merged_obj = list(set(obj1 + obj2))
        merged_subj = list(set(subj1 + subj2))
        
        # Merge openness: Agreement or annotator 1
        open1 = OPENNESS_MAP.get(open1_cn, "Neutral")
        open2 = OPENNESS_MAP.get(open2_cn, "Neutral") if (not pd.isna(open2_cn) and open2_cn) else None
        
        if open2 and open1 == open2:
            merged_openness = open1  # Agreement
        else:
            merged_openness = open1  # Use annotator 1 if disagree or annotator 2 missing
        
        records.append({
            "id": f"{platform_name[:2]}_{idx}",
            "platform": platform_name,
            "city": city,
            "text": text,
            "objective_criteria": merged_obj,
            "subjective_criteria": merged_subj,
            "openness": merged_openness,
            "annotator": "merged",
            "split": "annotated",
        })
    
    print(f"{platform_name}: loaded {len(records)} samples (from {len(df)} rows, skipped {skipped})")
    return records


def main():
    parser = argparse.ArgumentParser(description="Unify 3-platform data format")
    parser.add_argument("--xhs_dir", default="dataset/lf_data", help="XHS JSONL directory")
    parser.add_argument("--douyin", default="dataset/抖音_filtered.xlsx", help="Douyin Excel")
    parser.add_argument("--wechat", default="dataset/微信视频号_filtered.xlsx", help="WeChat Excel")
    parser.add_argument("--output", default="dataset/unified_3platform.jsonl", help="Output JSONL")
    parser.add_argument("--xhs_split", default="all", choices=["train", "dev", "test", "all"],
                        help="Which XHS splits to include")
    args = parser.parse_args()
    
    all_records = []
    
    # Load all platforms
    all_records.extend(load_xiaohongshu(args.xhs_dir, args.xhs_split))
    all_records.extend(load_douyin_wechat(args.douyin, "抖音"))
    all_records.extend(load_douyin_wechat(args.wechat, "微信视频号"))
    
    # Save unified dataset
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n=== Summary ===")
    print(f"Total: {len(all_records)} samples")
    
    # Platform distribution
    from collections import Counter
    platform_counts = Counter(r["platform"] for r in all_records)
    for p, c in platform_counts.most_common():
        print(f"  {p}: {c}")
    
    # City distribution
    city_counts = Counter(r["city"] for r in all_records)
    print(f"\nCities:")
    for city, c in city_counts.most_common():
        print(f"  {city}: {c}")
    
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
