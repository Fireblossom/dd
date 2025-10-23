#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from collections import defaultdict
UNWANTED_LABEL = "以上标准皆不符合|提及了其它标准，请在下面备注"


RANDOM_SEED = 42


def infer_city(sample_id: str) -> str:
    # sample_id examples: "shenzhen_2_4", "beijing_177_0"
    return sample_id.split("_")[0]


def parse_labels(label_field: str) -> list:
    # Handle both JSON format and direct string format
    if label_field is None:
        return []
    label_field = label_field.strip()
    if not label_field:
        return []
    
    # Try JSON format first (for top20-1.csv)
    try:
        labels = json.loads(label_field)
        if isinstance(labels, list):
            return labels
        return []
    except Exception:
        # Try to normalize common anomalies
        try:
            labels = json.loads(label_field.replace("''", '"'))
            if isinstance(labels, list):
                return labels
        except Exception:
            pass
    
    # Handle direct string format (for merged.csv)
    # Split by semicolon and clean up
    labels = [label.strip() for label in label_field.split(';') if label.strip()]
    return labels


def normalize_labels(labels: list) -> list:
    # Keep original strings but also derive short tags for LLaMA targets
    normalized = []
    for x in labels:
        if not isinstance(x, str):
            continue
        normalized.append(x.strip())
    return normalized


def is_none_label(labels: list) -> bool:
    for x in labels:
        if "未提及任何标准" in x:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/nfs/home/tanz/dd2/annotated/merged.csv")
    parser.add_argument("--outdir", default="/nfs/home/tanz/dd2/annotated/splits")
    parser.add_argument("--lf_outdir", default="/nfs/home/tanz/dd2/annotated/lf_data")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.lf_outdir, exist_ok=True)

    # Read CSV and determine column indices
    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    # Build name->idx
    name_to_idx = {name: i for i, name in enumerate(header)}

    # Required columns present in the sample
    col_id = name_to_idx.get("id")
    col_text = name_to_idx.get("text")
    col_labels = name_to_idx.get("结果字段-“本地人”身份认定")
    col_tone = name_to_idx.get("结果字段-宽容或严格")
    col_phrase = name_to_idx.get("结果字段-表达宽容或严格的短语")

    if None in (col_id, col_text, col_labels):
        raise RuntimeError("Required columns not found in CSV header")

    # Group by city
    city_to_rows = defaultdict(list)
    for row in rows:
        sample_id = row[col_id]
        city = infer_city(sample_id)
        city_to_rows[city].append(row)

    # For each city, shuffle and split 8:1:1
    split_index_files = {"train": [], "dev": [], "test": []}
    city_summaries = {}
    for city, city_rows in city_to_rows.items():
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(city_rows)
        n = len(city_rows)
        n_train = int(n * 0.8)
        n_dev = int(n * 0.1)
        n_test = n - n_train - n_dev
        train_rows = city_rows[:n_train]
        dev_rows = city_rows[n_train : n_train + n_dev]
        test_rows = city_rows[n_train + n_dev :]

        # Write per-city CSVs
        for split_name, split_rows in (("train", train_rows), ("dev", dev_rows), ("test", test_rows)):
            out_path = os.path.join(args.outdir, f"{city}_{split_name}.csv")
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(split_rows)
            split_index_files[split_name].append(out_path)

        city_summaries[city] = {"n": n, "train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)}

    # Also write combined split CSVs across all cities
    for split_name in ("train", "dev", "test"):
        combined = []
        for path in split_index_files[split_name]:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                _ = next(reader)
                for row in reader:
                    combined.append(row)
        out_path = os.path.join(args.outdir, f"all_{split_name}.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(combined)

    # Build JSONL for LLaMA-Factory
    # Task 1: labels-only multi-label classification
    # Input: original text; Output: semicolon-separated labels (or "未提及任何标准")
    def build_records(csv_path):
        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header_local = next(reader)
            name_to_idx_local = {name: i for i, name in enumerate(header_local)}
            id_i = name_to_idx_local["id"]
            text_i = name_to_idx_local["text"]
            labels_i = name_to_idx_local["结果字段-“本地人”身份认定"]
            tone_i = name_to_idx_local.get("结果字段-宽容或严格")
            phrase_i = name_to_idx_local.get("结果字段-表达宽容或严格的短语")
            for row in reader:
                text_val = row[text_i]
                labels = normalize_labels(parse_labels(row[labels_i]))
                labels = [s for s in labels if s != UNWANTED_LABEL]
                tone_val = row[tone_i] if tone_i is not None else ""
                phrase_val = row[phrase_i] if phrase_i is not None else ""
                records.append(
                    {
                        "id": row[id_i],
                        "text": text_val,
                        "labels": labels,
                        "tone": tone_val,
                        "phrase": phrase_val,
                    }
                )
        return records

    def labels_to_target(labels: list) -> str:
        if not labels:
            return "以上标准皆不符合|未提及任何标准"
        if is_none_label(labels):
            return "以上标准皆不符合|未提及任何标准"
        return "; ".join(labels)

    def write_jsonl(split_name: str, records: list, with_phrase: bool):
        # Build improved prompts per improved_prompt.md
        out_path = os.path.join(
            args.lf_outdir,
            f"labels_{'withphrase' if with_phrase else 'only'}_{split_name}.jsonl",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                if with_phrase:
                    instruction = (
                        "任务：从给定文本中判断‘本地人判定标准’（多选），以及‘开放倾向’（单选）。\n\n"
                        "一：‘本地人判定标准’可选标签（只能从下列中选，输出不要修改标签，允许多选用分号（;）分隔）：\n"
                        f"{allowed_labels_text}\n\n"
                        f"{guidelines_text}\n\n"
                        "二：‘开放倾向’可选标签（单选，输出不要修改标签）：\n"
                        f"{tone_guidelines}\n\n"
                        "只输出严格 JSON（三个key）：{\\\"本地人判定标准\\\": \\\"标签1; 标签2\\\" 或 \\\"以上标准皆不符合|未提及任何标准\\\", \\\"判断依据\\\": \\\"依据短语或空字符串\\\", \\\"开放倾向\\\": \\\"宽容开放/中性/紧缩排斥\\\"}。禁止输出其他内容。\n\n"
                        "示例：\n"
                        "- 文本：来了深圳就是深圳人\n"
                        "  输出：{\\\"本地人判定标准\\\": \\\"客观标准|个人在当地的当前居住状态\\\", \\\"判断依据\\\": \\\"来了...就是\\\", \\\"开放倾向\\\": \\\"宽容开放\\\"}\n"
                        "- 文本：怎么也得是爷爷奶奶姥姥姥爷那辈儿\n"
                        "  输出：{\\\"本地人判定标准\\\": \\\"客观标准|家族在当地的历史传承\\\", \\\"判断依据\\\": \\\"怎么也得是\\\", \\\"开放倾向\\\": \\\"紧缩排斥\\\"}\n"
                        "- 文本：这段话没有谈到身份标准\n"
                        "  输出：{\\\"本地人判定标准\\\": \\\"以上标准皆不符合|未提及任何标准\\\", \\\"判断依据\\\": \\\"\\\", \\\"开放倾向\\\": \\\"中性\\\"}\n"
                        "请对下面文本作答："
                    )
                else:
                    instruction = (
                        "任务：从给定文本中判断‘本地人判定标准’（多选），以及‘开放倾向’（单选）。\n\n"
                        "一：‘本地人判定标准’可选标签（只能从下列中选，输出不要修改标签，允许多选用分号（;）分隔）：\n"
                        f"{allowed_labels_text}\n\n"
                        f"{guidelines_text}\n\n"
                        "二：‘开放倾向’可选标签（单选，输出不要修改标签）：\n"
                        f"{tone_guidelines}\n\n"
                        "只输出严格 JSON（两个key）：{\\\"本地人判定标准\\\": \\\"标签1; 标签2\\\" 或 \\\"以上标准皆不符合|未提及任何标准\\\", \\\"开放倾向\\\": \\\"宽容开放/中性/紧缩排斥\\\"}。禁止输出其他内容。\n\n"
                        "示例：\n"
                        "- 文本：来了深圳就是深圳人\n"
                        "  输出：{\\\"本地人判定标准\\\": \\\"客观标准|个人在当地的当前居住状态\\\", \\\"开放倾向\\\": \\\"宽容开放\\\"}\n"
                        "- 文本：怎么也得是爷爷奶奶姥姥姥爷那辈儿\n"
                        "  输出：{\\\"本地人判定标准\\\": \\\"客观标准|家族在当地的历史传承\\\", \\\"开放倾向\\\": \\\"紧缩排斥\\\"}\n"
                        "- 文本：这段话没有谈到身份标准\n"
                        "  输出：{\\\"本地人判定标准\\\": \\\"以上标准皆不符合|未提及任何标准\\\", \\\"开放倾向\\\": \\\"中性\\\"}\n"
                        "请对下面文本作答："
                    )
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
                target = json.dumps(target_obj, ensure_ascii=False)

                ex = {
                    "system": system_text,
                    "instruction": instruction,
                    "input": input_text,
                    "output": target,
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Build records for all splits first
    recs_by_split = {}
    all_recs_for_space = []
    for split_name in ("train", "dev", "test"):
        csv_path = os.path.join(args.outdir, f"all_{split_name}.csv")
        recs = build_records(csv_path)
        recs_by_split[split_name] = recs
        all_recs_for_space.extend(recs)

    # Derive allowed label space from all data and format for prompts
    label_space_set = set()
    for r in all_recs_for_space:
        for l in r.get("labels", []):
            if isinstance(l, str) and l.strip():
                label_space_set.add(l.strip())
    # Normalize the none label to the unified form used in prompts
    if "未提及任何标准" in label_space_set:
        label_space_set.discard("未提及任何标准")
    if UNWANTED_LABEL in label_space_set:
        label_space_set.discard(UNWANTED_LABEL)
    label_space_set.add("以上标准皆不符合|未提及任何标准")
    label_space = sorted(label_space_set)
    allowed_labels_text = "; ".join(label_space)

    # Summarized guidelines from manual.txt to embed in prompts
    guidelines_text = (
        "判定指南：\n"
        "1. 客观标准\n"
        "判断指引：是否可以通过查阅档案、证件或可量化的事实来核实，不涉及个人感受或社会评价。\n"
        "1.1 客观标准|个人在当地的当前居住状态\n"
        "仅指个体本人的当前居住地。例如：\"来了就是当地人\"。\n"
        "1.2 客观标准|个人在当地的出生和成长经历\n"
        "涵盖个体本人的早期生命轨迹与经历。例如：\"在这里出生长大\"。\n"
        "1.3 客观标准|个人的法律与行政认定\n"
        "涵盖由政府或官方机构授予的身份与资格。包括：户口所在地、身份证号等。\n"
        "1.4 客观标准|家族在当地的历史传承\n"
        "涵盖个体家庭及祖辈的历史根基与传承。包括：父母/祖辈的经历和行政认定、家族在本地定居的代数等。\n"
        "1.5 客观标准|个人或家族在当地的资产与权益\n"
        "涵盖与本地直接相关的经济性资产与权利。包括：房产、土地、村集体分红权等。\n"
        "1.6 客观标准|城市内部地理片区归属\n"
        "与\"1.1 个人在当地的当前居住状态\"不同，本类别并非指\"住在本地/定居于本地\"，而是强调\"以城市内部某条边界作为是否算本地人的门槛\"。是否跨过这条\"线\"（空间边界）本身决定身份，可能是在长期居住、落户、家族/资产等条件基础上的进一步限定。\n"
        "判定要点（需同时满足\"边界+门槛\"两要素）：\n"
        "  边界：出现明确的城市内部边界或核心区域术语，如城墙/护城河、（内/外）城、老城/老城区、核心区、（一/二/三）环、内环/外环、城八区、老三区/六城区、特定区县/街道/片区列表等。\n"
        "  门槛：与\"只认/才算/必须/以内/以外/界内/界外/不算\"等排他性措辞搭配，表达\"在界内才算/界外不算\"的含义。\n"
        "正例：\"二环里才算北京人\"\"只认老三区才是本地人\"\"护城河以内才算土著\"。\n"
        "反例（应归入其他类别）：仅说\"住在市区/城区\"但未表达边界门槛（→1.1）；\"在本地长大\"（→1.2）；\"本地户口\"（→1.3）；\"祖辈在城里\"（→1.4）。\n"
        "\n"
        "2. 社会文化与心理标准\n"
        "判断指引：此类别按照个体融入的深度排序，从个人认定到社群接纳。\n"
        "2.1 社会文化与心理标准|个人认定\n"
        "涵盖个体认知上的界定和宣告。例如：\"自己认为是本地人那就是本地人\"。\n"
        "2.2 社会文化与心理标准|个人的归属感\n"
        "涵盖个体在情感上与城市的联结和忠诚度，比单纯的个人认定更深一层。例如：\"打心底里把这个城市当成家\"。\n"
        "2.3 社会文化与心理标准|个人的语言能力\n"
        "涵盖对本地话（方言）的掌握和使用。这是个体主动融入文化圈的关键技能。包括：能否听懂、会不会说、口音是否地道等。\n"
        "2.4 社会文化与心理标准|个人的文化实践与知识\n"
        "涵盖除语言外，个体后天习得并表现出的更深层次的文化特征。包括：生活习惯（穿着、饮食、消费）、行为举止（礼仪）、文化知识（本土记忆）等。\n"
        "2.5 社会文化与心理标准|个人被社群接纳\n"
        "涵盖个体是否被本地社群感知、承认并接纳为\"自己人\"，是融入的最终体现。\n"
        "判断依据包括：直觉与气质判断（\"一眼就能看出来\"）；人际网络构成（是否有较多本地朋友，是否能融入本地人的社交圈）。"
    )
    tone_guidelines = (
        "宽容开放 / 中性 / 紧缩排斥\n"
        "判定指南：\n"
        "宽容开放：条件较宽泛，给人一种包容的感觉。判断依据：使用\"只要…就…\"、\"随便…\"、\"都可以\"、\"来了就是\"等宽松表达。\n"
        "中性：没有明显的包容或排斥倾向。\n"
        "紧缩排斥：条件严格或带有限制性，给人一种排斥的感觉。判断依据：使用\"只有…才…\"、\"必须…\"、\"不行就不是\"、\"怎么也得\"等严格/排他表达。\n"
        "无法判断时默认中性。"
    )
    system_text = (
        "你是一名严谨的标注助手。严格遵循以下规则：\n"
        "- 对于'本地人判定标准'，仅从提供的标签清单中选择，逐字匹配；若不涉及任何标准，输出'以上标准皆不符合|未提及任何标准'。\n"
        "- 对于'判断依据'，提取文本中体现判定标准或开放倾向的关键短语（如关联词、条件句、隐喻等）；若无明显依据，输出空字符串。\n"
        "- 对于'开放倾向'，仅在'宽容开放/中性/紧缩排斥'中选择；无法判断时使用'中性'。\n"
        "- 输出必须满足要求的严格格式；禁止输出与格式无关的任何解释性文本。\n"
    )

    # Write JSONL with improved prompts
    for split_name in ("train", "dev", "test"):
        recs = recs_by_split[split_name]
        write_jsonl(split_name, recs, with_phrase=False)
        write_jsonl(split_name, recs, with_phrase=True)

    # Write a brief summary file
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"cities": city_summaries}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


