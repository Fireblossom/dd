#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, precision_recall_fscore_support, precision_score, recall_score


NONE_LABEL = "未提及任何标准"


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def extract_gold_labels(sample: dict) -> List[str]:
    """Extract gold labels from dataset jsonl entry (our dataset stores target in output)."""
    out = sample.get("output", "").strip()
    if not out:
        return [NONE_LABEL]
    # Expect JSON output; fallback to plain string (semicolon-separated)
    try:
        obj = json.loads(out)
        labels = obj.get("labels", obj.get("本地人判定标准", ""))
        if isinstance(labels, list):
            return [l.strip() for l in labels] if labels else [NONE_LABEL]
        if isinstance(labels, str):
            s = labels.strip()
            if not s or NONE_LABEL in s:
                return [NONE_LABEL]
            return [t.strip() for t in s.split(";") if t.strip()]
    except Exception:
        pass
    if NONE_LABEL in out:
        return [NONE_LABEL]
    parts = [t.strip() for t in out.split(";") if t.strip()]
    return parts if parts else [NONE_LABEL]


def _normalize_tone_value(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = re.sub(r"\s+", "", value)
    if not s:
        return None
    # Canonical set
    if s in {"宽容开放", "中性", "紧缩排斥"}:
        return s
    # Synonyms mapping
    if any(k in s for k in ["宽容", "开放", "包容"]):
        return "宽容开放"
    if any(k in s for k in ["严格", "排斥", "紧缩", "排外"]):
        return "紧缩排斥"
    if any(k in s for k in ["中性", "中立", "一般", "无明显倾向"]):
        return "中性"
    return None


def extract_gold_tone(sample: dict) -> Optional[str]:
    """Extract gold tone (e.g., 宽容/严格) if available in JSON for labels_withphrase mode."""
    out = sample.get("output", "").strip()
    if not out:
        return None
    try:
        obj = json.loads(out)
        tone = obj.get("tone", obj.get("开放倾向", None))
        tone_norm = _normalize_tone_value(tone)
        if tone_norm:
            return tone_norm
    except Exception:
        return None
    return None


def extract_pred_tone(pred_text: str) -> Optional[str]:
    """Extract predicted tone from model output for labels_withphrase mode.
    Prefer JSON field; fallback to keyword spotting for common values (inclusive/restrictive).
    """
    if not pred_text or not pred_text.strip():
        return None
    s = pred_text.strip()
    try:
        obj = json.loads(s)
        tone = obj.get("tone", obj.get("开放倾向", None))
        tone_norm = _normalize_tone_value(tone)
        if tone_norm:
            return tone_norm
    except Exception:
        pass
    # Fallback keyword matching (expanded)
    s_norm = re.sub(r"\s+", "", s)
    tone_norm = _normalize_tone_value(s_norm)
    return tone_norm


def build_label_space(golds: List[List[str]]) -> List[str]:
    s = set()
    for ls in golds:
        for l in ls:
            s.add(l)
    
    # Exclude unwanted categories
    excluded_labels = {
        "以上标准皆不符合|提及了其它标准，请在下面备注"
    }
    s = s - excluded_labels
    
    return sorted(s)


def match_allowed_labels(text: str, allowed: List[str]) -> List[str]:
    text_norm = re.sub(r"\s+", "", text)
    matched = []
    for label in allowed:
        if label == NONE_LABEL:
            continue
        if re.sub(r"\s+", "", label) in text_norm:
            matched.append(label)
    return matched


def extract_pred_labels(pred_text: str, allowed: List[str]) -> List[str]:
    if not pred_text or not pred_text.strip():
        return [NONE_LABEL]
    s = pred_text.strip()
    # Try JSON first
    try:
        obj = json.loads(s)
        labels = obj.get("labels", obj.get("本地人判定标准", ""))
        if isinstance(labels, list):
            labs = [l.strip() for l in labels if isinstance(l, str) and l.strip()]
            return labs if labs else [NONE_LABEL]
        if isinstance(labels, str):
            s2 = labels.strip()
            if not s2 or NONE_LABEL in s2:
                return [NONE_LABEL]
            return [t.strip() for t in s2.split(";") if t.strip()]
    except Exception:
        pass
    # Check explicit NONE
    if NONE_LABEL in s:
        return [NONE_LABEL]
    # Semicolon-separated attempt
    if ";" in s:
        parts = [t.strip() for t in s.split(";") if t.strip()]
        parts = [p for p in parts if p in allowed]
        if parts:
            return parts
    # Fallback: keyword matching from allowed labels
    matched = match_allowed_labels(s, allowed)
    return matched if matched else [NONE_LABEL]


def to_multihot(labels: List[str], label_to_id: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(label_to_id), dtype=int)
    for l in labels:
        if l in label_to_id:
            y[label_to_id[l]] = 1
    return y


def evaluate(pred_file: str, data_file: str) -> dict:
    preds = load_jsonl(pred_file)
    golds = load_jsonl(data_file)
    if len(preds) != len(golds):
        raise ValueError(f"Predictions ({len(preds)}) and data ({len(golds)}) length mismatch")

    gold_label_lists = [extract_gold_labels(g) for g in golds]
    label_space = build_label_space(gold_label_lists + [[NONE_LABEL]])
    label_to_id = {l: i for i, l in enumerate(label_space)}

    Y_true, Y_pred = [], []
    none_true, none_pred = [], []
    for i in range(len(preds)):
        pred_text = preds[i].get("predict", "")
        gold_labels = gold_label_lists[i]
        pred_labels = extract_pred_labels(pred_text, label_space)
        Y_true.append(to_multihot(gold_labels, label_to_id))
        Y_pred.append(to_multihot(pred_labels, label_to_id))
        none_true.append(1 if NONE_LABEL in gold_labels else 0)
        none_pred.append(1 if NONE_LABEL in pred_labels else 0)

    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)

    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    
    # Compute per-class F1 scores
    f1_per_class = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    
    # Build per-class metrics dictionary
    class_f1_dict = {}
    for i, label in enumerate(label_space):
        class_f1_dict[label] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i])
        }
    
    # Compute Jaccard similarity, handling single-label case
    try:
        jacc = jaccard_score(Y_true, Y_pred, average="samples")
    except ValueError:
        # If single-label issue occurs, use macro average
        jacc = jaccard_score(Y_true, Y_pred, average="macro")
    
    hamm = hamming_loss(Y_true, Y_pred)
    exact_match = float(np.mean(np.all(Y_true == Y_pred, axis=1)))
    p_none, r_none, f1_none, _ = precision_recall_fscore_support(
        np.array(none_true), np.array(none_pred), average="binary", zero_division=0
    )

    # Tone metrics
    result = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "jaccard": float(jacc),
        "hamming_loss": float(hamm),
        "exact_match": exact_match,
        "none_precision": float(p_none),
        "none_recall": float(r_none),
        "none_f1": float(f1_none),
        "num_samples": int(len(preds)),
        "num_labels": int(len(label_space)),
        "class_f1_scores": class_f1_dict,
    }

    if True:
        gold_tones = [extract_gold_tone(g) for g in golds]
        valid_idx = [i for i, t in enumerate(gold_tones) if isinstance(t, str) and t.strip()]
        if valid_idx:
            gt = [gold_tones[i] for i in valid_idx]
            pt = [extract_pred_tone(preds[i].get("predict", "")) for i in valid_idx]

            tone_space = sorted(list(set(gt)))  # derive from gold (after normalization)
            # Map unknown predictions to a placeholder to count as errors
            unknown_label = "未知"
            all_classes = tone_space + [unknown_label]
            tone_to_id = {t: i for i, t in enumerate(all_classes)}

            y_true = np.array([tone_to_id[t] for t in gt], dtype=int)
            y_pred = np.array([tone_to_id[p] if isinstance(p, str) and p in tone_to_id else tone_to_id[unknown_label] for p in pt], dtype=int)

            tone_acc = float(np.mean(y_true == y_pred))
            # Compute F1 on known classes only (exclude '未知')
            known_mask = y_pred != tone_to_id[unknown_label]
            if known_mask.any():
                f1_tone_macro = f1_score(y_true[known_mask], y_pred[known_mask], labels=[tone_to_id[t] for t in tone_space], average="macro", zero_division=0)
            else:
                f1_tone_macro = 0.0

            result.update(
                {
                    "tone_classes": tone_space,
                    "tone_accuracy": tone_acc,
                    "tone_macro_f1": float(f1_tone_macro),
                    "tone_num_samples": int(len(valid_idx)),
                }
            )

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--data_file", required=True)
    # mode is deprecated; evaluator always parses JSON and computes tone
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metrics = evaluate(args.pred_file, args.data_file)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


