#!/usr/bin/env python3
"""
Experiment 1: Prompt Sensitivity Testing - Generate 3 prompt variant JSONL datasets

Variants:
- A (Baseline): Current detailed prompt
- B (Expert Role): Add expert role persona
- C (Minimal): Simplified version, only label lists
"""
import argparse
import csv
import json
import os
from typing import List, Dict

# 4 target cities
TARGET_CITIES = {"beijing", "shanghai", "guangzhou", "shenzhen"}

UNWANTED_LABEL = "以上标准皆不符合|提及了其它标准，请在下面备注"
NONE_LABEL = "以上标准皆不符合|未提及任何标准"

# Label space
LABEL_SPACE = [
    "客观标准|个人在当地的当前居住状态",
    "客观标准|个人在当地的出生和成长经历",
    "客观标准|个人的法律与行政认定",
    "客观标准|家族在当地的历史传承",
    "客观标准|个人或家族在当地的资产与权益",
    "客观标准|城市内部地理片区归属",
    "社会文化与心理标准|个人认定",
    "社会文化与心理标准|个人的归属感",
    "社会文化与心理标准|个人的语言能力",
    "社会文化与心理标准|个人的文化实践与知识",
    "社会文化与心理标准|个人被社群接纳",
    NONE_LABEL,
]

ALLOWED_LABELS_TEXT = "; ".join(LABEL_SPACE)

# ============ Prompt Variant Definitions ============

# Variant A: Baseline (current detailed version)
SYSTEM_A = (
    "你是一名严谨的标注助手。严格遵循以下规则：\n"
    "- 对于'本地人判定标准'，仅从提供的标签清单中选择，逐字匹配；若不涉及任何标准，输出'以上标准皆不符合|未提及任何标准'。\n"
    "- 对于'开放倾向'，仅在'宽容开放/中性/紧缩排斥'中选择；无法判断时使用'中性'。\n"
    "- 输出必须满足要求的严格格式；禁止输出与格式无关的任何解释性文本。\n"
)

GUIDELINES_TEXT = (
    "判定指南：\n"
    "1. 客观标准\n"
    "1.1 客观标准|个人在当地的当前居住状态：仅指个体本人的当前居住地。例如：\"来了就是当地人\"。\n"
    "1.2 客观标准|个人在当地的出生和成长经历：涵盖个体本人的早期生命轨迹。例如：\"在这里出生长大\"。\n"
    "1.3 客观标准|个人的法律与行政认定：政府或官方机构授予的身份。包括：户口、身份证。\n"
    "1.4 客观标准|家族在当地的历史传承：家庭及祖辈的历史根基。包括：父母/祖辈经历、家族定居代数。\n"
    "1.5 客观标准|个人或家族在当地的资产与权益：经济性资产与权利。包括：房产、土地、村集体分红。\n"
    "1.6 客观标准|城市内部地理片区归属：以城市内部边界作为身份门槛。如\"二环里才算北京人\"。\n"
    "2. 社会文化与心理标准\n"
    "2.1 社会文化与心理标准|个人认定：个体认知上的界定和宣告。\n"
    "2.2 社会文化与心理标准|个人的归属感：情感上与城市的联结和忠诚度。\n"
    "2.3 社会文化与心理标准|个人的语言能力：对本地话的掌握和使用。\n"
    "2.4 社会文化与心理标准|个人的文化实践与知识：生活习惯、行为举止、文化知识。\n"
    "2.5 社会文化与心理标准|个人被社群接纳：是否被本地社群感知并接纳。"
)

TONE_GUIDELINES = (
    "宽容开放 / 中性 / 紧缩排斥\n"
    "宽容开放：条件较宽泛，使用\"只要…就…\"、\"来了就是\"等宽松表达。\n"
    "中性：没有明显的包容或排斥倾向。\n"
    "紧缩排斥：条件严格或带有限制性，使用\"只有…才…\"、\"必须…\"等排他表达。"
)

def build_instruction_A(with_phrase: bool = False) -> str:
    if with_phrase:
        return (
            "任务：从给定文本中判断'本地人判定标准'（多选），以及'开放倾向'（单选）。\n\n"
            f"一：'本地人判定标准'可选标签：\n{ALLOWED_LABELS_TEXT}\n\n"
            f"{GUIDELINES_TEXT}\n\n"
            f"二：'开放倾向'可选标签：\n{TONE_GUIDELINES}\n\n"
            "只输出严格JSON：{\"本地人判定标准\": \"标签1; 标签2\", \"判断依据\": \"短语\", \"开放倾向\": \"宽容开放/中性/紧缩排斥\"}。\n"
            "请对下面文本作答："
        )
    else:
        return (
            "任务：从给定文本中判断'本地人判定标准'（多选），以及'开放倾向'（单选）。\n\n"
            f"一：'本地人判定标准'可选标签：\n{ALLOWED_LABELS_TEXT}\n\n"
            f"{GUIDELINES_TEXT}\n\n"
            f"二：'开放倾向'可选标签：\n{TONE_GUIDELINES}\n\n"
            "只输出严格JSON：{\"本地人判定标准\": \"标签1; 标签2\", \"开放倾向\": \"宽容开放/中性/紧缩排斥\"}。\n"
            "请对下面文本作答："
        )


# 变体B: Expert Role (专家角色)
SYSTEM_B = (
    "你是一位资深的社会语言学专家，专注于中国城市身份认同研究。"
    "你在分析社交媒体文本中关于\"谁是本地人\"的讨论时具有丰富经验。\n"
    "请以学术严谨的态度完成以下标注任务。\n"
    "- 对于'本地人判定标准'，仅从提供的标签清单中选择。\n"
    "- 对于'开放倾向'，仅在'宽容开放/中性/紧缩排斥'中选择。\n"
)

def build_instruction_B(with_phrase: bool = False) -> str:
    # 与A相同，但system不同
    return build_instruction_A(with_phrase)


# 变体C: Minimal (精简版)
SYSTEM_C = "你是一个标注助手。按要求输出JSON。"

def build_instruction_C(with_phrase: bool = False) -> str:
    if with_phrase:
        return (
            "从文本中判断：\n"
            f"1. 本地人判定标准（多选）：{ALLOWED_LABELS_TEXT}\n"
            "2. 开放倾向（单选）：宽容开放 / 中性 / 紧缩排斥\n\n"
            "输出JSON：{\"本地人判定标准\": \"...\", \"判断依据\": \"...\", \"开放倾向\": \"...\"}。\n"
            "文本："
        )
    else:
        return (
            "从文本中判断：\n"
            f"1. 本地人判定标准（多选）：{ALLOWED_LABELS_TEXT}\n"
            "2. 开放倾向（单选）：宽容开放 / 中性 / 紧缩排斥\n\n"
            "输出JSON：{\"本地人判定标准\": \"...\", \"开放倾向\": \"...\"}。\n"
            "文本："
        )


VARIANT_CONFIG = {
    "A": {"system": SYSTEM_A, "instruction_fn": build_instruction_A, "name": "Baseline"},
    "B": {"system": SYSTEM_B, "instruction_fn": build_instruction_B, "name": "ExpertRole"},
    "C": {"system": SYSTEM_C, "instruction_fn": build_instruction_C, "name": "Minimal"},
}


def infer_city(sample_id: str) -> str:
    return sample_id.split("_")[0]


def parse_labels(label_field: str) -> List[str]:
    if not label_field or not label_field.strip():
        return []
    try:
        labels = json.loads(label_field)
        if isinstance(labels, list):
            return labels
    except:
        pass
    return [l.strip() for l in label_field.split(";") if l.strip()]


def normalize_labels(labels: List[str]) -> List[str]:
    return [l.strip() for l in labels if isinstance(l, str) and l.strip() and l != UNWANTED_LABEL]


def labels_to_target(labels: List[str]) -> str:
    if not labels:
        return NONE_LABEL
    for l in labels:
        if "未提及任何标准" in l:
            return NONE_LABEL
    return "; ".join(labels)


def load_test_data(splits_dir: str, split: str = "test") -> List[Dict]:
    """Load data from 4 target cities.
    
    Args:
        splits_dir: Directory with city split CSVs
        split: One of 'test', 'train', 'dev', or 'all'
    """
    records = []
    splits_to_load = ["train", "dev", "test"] if split == "all" else [split]
    
    for city in TARGET_CITIES:
        for s in splits_to_load:
            csv_path = os.path.join(splits_dir, f"{city}_{s}.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found")
                continue
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle column name with Chinese quotes
                    label_col = '结果字段-"本地人"身份认定'
                    labels = normalize_labels(parse_labels(row.get(label_col, "")))
                    records.append({
                        "id": row["id"],
                        "city": infer_city(row["id"]),
                        "text": row["text"],
                        "labels": labels,
                        "tone": row.get("结果字段-宽容或严格", "中性"),
                        "phrase": row.get("结果字段-表达宽容或严格的短语", ""),
                        "split": s,
                    })
    return records


def load_jsonl_data(jsonl_dir: str, split: str = "test") -> List[Dict]:
    """Load data from JSONL source files (labels_withphrase_*.jsonl).
    
    Args:
        jsonl_dir: Directory with JSONL files (e.g., dataset/lf_data)
        split: One of 'test', 'train', 'dev', or 'all'
    """
    records = []
    splits_to_load = ["train", "dev", "test"] if split == "all" else [split]
    
    # Mapping for tone values
    TONE_MAP = {
        "紧缩排斥": "紧缩排斥",
        "中性": "中性",
        "宽容开放": "宽容开放",
    }
    
    for s in splits_to_load:
        jsonl_path = os.path.join(jsonl_dir, f"labels_withphrase_{s}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found")
            continue
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    
                    # Parse output JSON
                    output = json.loads(d.get("output", "{}"))
                    
                    # Extract city from input text
                    input_text = d.get("input", "")
                    city = None
                    for c in TARGET_CITIES:
                        if c in input_text:
                            city = c
                            break
                    
                    if not city:
                        continue  # Skip if not in target cities
                    
                    # Parse labels from "本地人判定标准"
                    criteria_str = output.get("本地人判定标准", "")
                    labels = []
                    for part in criteria_str.split(";"):
                        part = part.strip()
                        if part:
                            labels.append(part)
                    
                    # Get tone
                    tone = output.get("开放倾向", "中性")
                    
                    # Get phrase
                    phrase = output.get("判断依据", "")
                    
                    records.append({
                        "id": f"{s}_{line_num}",
                        "city": city,
                        "text": input_text,
                        "labels": labels,
                        "tone": tone,
                        "phrase": phrase,
                        "split": s,
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    continue
    
    print(f"Loaded from JSONL: {len(records)} samples from {splits_to_load}")
    return records


def generate_variant_jsonl(records: List[Dict], variant: str, outdir: str, with_phrase: bool = False):
    """Generate JSONL file for a specific prompt variant."""
    cfg = VARIANT_CONFIG[variant]
    system_text = cfg["system"]
    instruction_fn = cfg["instruction_fn"]
    instruction = instruction_fn(with_phrase)
    
    suffix = "withphrase" if with_phrase else "only"
    out_path = os.path.join(outdir, f"sensitivity_{variant}_{suffix}_test.jsonl")
    
    os.makedirs(outdir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            input_text = f"文本：{r['text']}"
            if with_phrase:
                target_obj = {
                    "本地人判定标准": labels_to_target(r["labels"]),
                    "判断依据": r.get("phrase", "") or "",
                    "开放倾向": r.get("tone", "") or "中性",
                }
            else:
                target_obj = {
                    "本地人判定标准": labels_to_target(r["labels"]),
                    "开放倾向": r.get("tone", "") or "中性",
                }
            
            ex = {
                "system": system_text,
                "instruction": instruction,
                "input": input_text,
                "output": json.dumps(target_obj, ensure_ascii=False),
                "_meta": {"id": r["id"], "city": r["city"], "variant": variant}
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"Generated: {out_path} ({len(records)} samples)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate prompt sensitivity test datasets")
    parser.add_argument("--splits_dir", default="dataset/splits", help="Directory with city split CSVs")
    parser.add_argument("--outdir", default="dataset/lf_data/sensitivity", help="Output directory")
    parser.add_argument("--with_phrase", action="store_true", help="Include phrase extraction")
    parser.add_argument("--split", default="test", choices=["test", "train", "dev", "all"],
                        help="Which split to use: test (default), train, dev, or all")
    args = parser.parse_args()
    
    print(f"Loading {args.split} data from 4 cities...")
    records = load_test_data(args.splits_dir, args.split)
    print(f"Loaded {len(records)} samples")
    
    # Generate JSONL for each variant
    for variant in ["A", "B", "C"]:
        generate_variant_jsonl(records, variant, args.outdir, args.with_phrase)
    
    # Also generate dataset_info.json for LLaMA-Factory
    dataset_info = {}
    for variant in ["A", "B", "C"]:
        suffix = "withphrase" if args.with_phrase else "only"
        split_suffix = args.split if args.split != "test" else "test"
        dataset_info[f"sensitivity_{variant}_{suffix}_{split_suffix}"] = {
            "file_name": f"sensitivity_{variant}_{suffix}_test.jsonl",
            "columns": {"system": "system", "prompt": "instruction", "query": "input", "response": "output"}
        }
    
    info_path = os.path.join(args.outdir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"Generated: {info_path}")
    
    print(f"\nDone! Generated datasets for {len(records)} samples")
    print("Example command:")
    print(f"  python /path/to/vllm_infer.py --dataset sensitivity_A_only_test --dataset_dir {args.outdir}")


if __name__ == "__main__":
    main()

