#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import shutil

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    accuracy_score,
)
import matplotlib


# Use non-interactive backend for headless servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Label spaces (keep consistent with baseline)
ALLOWED_LABELS: List[str] = [
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
    "以上标准皆不符合|未提及任何标准",
]
NONE_LABEL = "以上标准皆不符合|未提及任何标准"
OPENNESS_LABELS: List[str] = ["紧缩排斥", "中性", "宽容开放"]


def _safe_filename(text: str) -> str:
    # Convert arbitrary text (like model ids) into filesystem-safe names
    # Replace path separators first, then collapse other unsafe chars
    t = text.replace("/", "_")
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("._-")
    return t or "unnamed"


@dataclass
class EvalConfig:
    exp_dir: str
    base_model: str
    template: str
    dataset_name: str
    dataset_dir: str
    batch_size: int
    bf16: bool
    output_root: str
    dry_run: bool
    max_checkpoints: Optional[int]
    max_new_tokens: Optional[int]


def list_checkpoints(exp_dir: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    for name in os.listdir(exp_dir):
        full = os.path.join(exp_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m:
            step = int(m.group(1))
            out.append((step, full))
    out.sort(key=lambda t: t[0])
    return out


def run_prediction_for_checkpoint(
    cfg: EvalConfig,
    ckpt_path: str,
    eval_dir: str,
    base_model_override: Optional[str] = None,
    template_override: Optional[str] = None,
    dataset_name_override: Optional[str] = None,
) -> None:
    os.makedirs(eval_dir, exist_ok=True)
    # Assemble command using Hugging Face generation backend (no vLLM)
    cmd = [
        "llamafactory-cli",
        "train",
        "--stage",
        "sft",
        "--model_name_or_path",
        base_model_override or cfg.base_model,
        "--template",
        template_override or cfg.template,
        "--finetuning_type",
        "lora",
        "--adapter_name_or_path",
        ckpt_path,
        "--output_dir",
        eval_dir,
        "--overwrite_output_dir",
        "True",
        "--dataset",
        dataset_name_override or cfg.dataset_name,
        "--eval_dataset",
        dataset_name_override or cfg.dataset_name,
        "--dataset_dir",
        cfg.dataset_dir,
        "--do_train",
        "False",
        "--do_eval",
        "False",
        "--do_predict",
        "True",
        "--predict_with_generate",
        "True",
        "--per_device_eval_batch_size",
        str(cfg.batch_size),
        "--report_to",
        "none",
    ]
    if cfg.bf16:
        cmd += ["--bf16", "True"]
    if cfg.max_new_tokens is not None and cfg.max_new_tokens > 0:
        cmd += ["--max_new_tokens", str(cfg.max_new_tokens)]

    if cfg.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return

    print(f"Running prediction for {ckpt_path} -> {eval_dir}")
    log_path = os.path.join(eval_dir, "run.log")
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n")
            lf.flush()
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    except Exception:
        # Fallback to in-memory capture if log file cannot be opened
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        # Surface error but do not stop the whole loop
        sys.stderr.write(f"Prediction failed for {ckpt_path}. See log: {log_path}\n")
    else:
        print(f"Finished prediction for {ckpt_path}. Log: {log_path}")


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    if text.startswith("```"):
        # take content between first and last triple backticks
        parts = text.split("```")
        if len(parts) >= 3:
            text = "```".join(parts[1:-1]).strip()
    return text


def parse_prediction_obj(predict_field: str) -> Dict[str, Any]:
    s = _strip_code_fences(predict_field)
    # Extract first JSON object substring
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        return json.loads(s[start : end + 1])
    except Exception:
        return {}


def load_predictions(eval_dir: str) -> List[Dict[str, Any]]:
    # Try typical files in order
    candidates = [
        os.path.join(eval_dir, "generated_predictions.jsonl"),
        os.path.join(eval_dir, "inference_results_v2.jsonl"),
        os.path.join(eval_dir, "inference_results.jsonl"),
        os.path.join(eval_dir, "serve_inference_results.jsonl"),
    ]
    pred_path = next((p for p in candidates if os.path.exists(p)), None)
    if pred_path is None:
        return []

    preds: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Different files may store different keys; normalize
            if "predict" in obj and isinstance(obj["predict"], str):
                parsed = parse_prediction_obj(obj["predict"]) or {}
            elif all(k in obj for k in ["本地人判定标准", "判断依据", "开放倾向"]):
                parsed = obj
            else:
                parsed = {}
            preds.append(parsed)
    return preds


def find_predictions_file(eval_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(eval_dir, "generated_predictions.jsonl"),
        os.path.join(eval_dir, "inference_results_v2.jsonl"),
        os.path.join(eval_dir, "inference_results.jsonl"),
        os.path.join(eval_dir, "serve_inference_results.jsonl"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def load_predictions_from_file(jsonl_path: str) -> List[Dict[str, Any]]:
    if not (jsonl_path and os.path.exists(jsonl_path)):
        return []
    preds: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "predict" in obj and isinstance(obj["predict"], str):
                parsed = parse_prediction_obj(obj["predict"]) or {}
            elif all(k in obj for k in ["本地人判定标准", "判断依据", "开放倾向"]):
                parsed = obj
            else:
                parsed = {}
            preds.append(parsed)
    return preds


def summary_has_entry(csv_path: str, predictions_file: str) -> bool:
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                if predictions_file in line:
                    return True
    except Exception:
        return False
    return False


def list_experiment_dirs(root_dir: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root_dir):
        return out
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            has_ckpt = any(
                os.path.isdir(os.path.join(full, d)) and re.fullmatch(r"checkpoint-(\d+)", d)
                for d in os.listdir(full)
            )
        except Exception:
            has_ckpt = False
        if has_ckpt:
            out.append(full)
    out.sort()
    return out


def _try_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def detect_base_model(exp_dir: str, ckpt_dir: Optional[str] = None) -> Optional[str]:
    # Priority: checkpoint-local files, then experiment-level files, then hint files
    candidate_files: List[str] = []
    if ckpt_dir:
        candidate_files.extend([
            os.path.join(ckpt_dir, "adapter_config.json"),
            os.path.join(ckpt_dir, "config.json"),
            os.path.join(ckpt_dir, "run_args.json"),
            os.path.join(ckpt_dir, "training_args.json"),
        ])
    candidate_files.extend([
        os.path.join(exp_dir, "adapter_config.json"),
        os.path.join(exp_dir, "config.json"),
        os.path.join(exp_dir, "run_args.json"),
        os.path.join(exp_dir, "training_args.json"),
        os.path.join(exp_dir, "args.json"),
        os.path.join(exp_dir, "llamafactory_config.json"),
        os.path.join(exp_dir, "base_model.txt"),
    ])

    keys = [
        "base_model_name_or_path",
        "base_model",
        "model_name_or_path",
        "model_name",
    ]

    for p in candidate_files:
        if not os.path.exists(p):
            continue
        if p.endswith(".txt"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                    if val:
                        return val
            except Exception:
                continue
        data = _try_read_json(p)
        for k in keys:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def detect_template(exp_dir: str, ckpt_dir: Optional[str] = None, base_model_hint: Optional[str] = None) -> Optional[str]:
    # Try to read from config files first
    candidate_files: List[str] = []
    if ckpt_dir:
        candidate_files.extend([
            os.path.join(ckpt_dir, "adapter_config.json"),
            os.path.join(ckpt_dir, "config.json"),
            os.path.join(ckpt_dir, "run_args.json"),
            os.path.join(ckpt_dir, "training_args.json"),
        ])
    candidate_files.extend([
        os.path.join(exp_dir, "adapter_config.json"),
        os.path.join(exp_dir, "config.json"),
        os.path.join(exp_dir, "run_args.json"),
        os.path.join(exp_dir, "training_args.json"),
        os.path.join(exp_dir, "args.json"),
        os.path.join(exp_dir, "llamafactory_config.json"),
    ])
    for p in candidate_files:
        if not os.path.exists(p):
            continue
        data = _try_read_json(p)
        v = data.get("template")
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Heuristic mapping from base model name
    if base_model_hint:
        bm = base_model_hint.lower()
        if "qwen" in bm:
            return "qwen"
        if "gemma" in bm:
            return "gemma"
        if "yi" in bm:
            return "yi"
    # Heuristic from experiment folder name
    name = os.path.basename(exp_dir).lower()
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    if "yi" in name:
        return "yi"
    return None


def detect_predict_dataset(exp_dir: str) -> Optional[str]:
    # Prefer explicit eval_dataset if present in args
    candidate_files = [
        os.path.join(exp_dir, "run_args.json"),
        os.path.join(exp_dir, "training_args.json"),
        os.path.join(exp_dir, "config.json"),
        os.path.join(exp_dir, "args.json"),
        os.path.join(exp_dir, "llamafactory_config.json"),
    ]
    for p in candidate_files:
        if not os.path.exists(p):
            continue
        data = _try_read_json(p)
        v = data.get("eval_dataset") or data.get("eval_datasets") or data.get("dataset_eval")
        if isinstance(v, str) and v.strip():
            return v.strip()
        # Some configs only have train dataset; switch to *_test if recognizable
        v = data.get("dataset") or data.get("datasets") or data.get("train_dataset")
        if isinstance(v, str) and v.strip():
            s = v.strip()
            if s.endswith("_train"):
                return s[:-6] + "_test"
            if s.endswith("_dev"):
                return s[:-4] + "_test"
    # Heuristic based on folder name keywords
    name = os.path.basename(exp_dir).lower()
    if ("withphrase" in name) or ("labels_phrase" in name) or ("labels_phrases" in name):
        return "labels_withphrase_test"
    if "labels_only" in name:
        return "labels_only_test"
    return None


def dataset_group_slug(dataset_name: Optional[str], exp_dir: Optional[str] = None) -> str:
    name = (dataset_name or "").lower()
    expn = os.path.basename(exp_dir).lower() if exp_dir else ""
    # With-phrase variants
    if (
        ("withphrase" in name)
        or ("labels_phrase" in name)
        or ("labels_phrases" in name)
        or ("withphrase" in expn)
        or ("labels_phrase" in expn)
        or ("labels_phrases" in expn)
    ):
        return "label_phrase"
    # Labels only
    if ("labels_only" in name) or ("labels_only" in expn):
        return "label_only"
    # Fallback: strip common suffixes and sanitize
    base = re.sub(r"_(test|dev|train)$", "", name)
    base = base or name or "dataset"
    return _safe_filename(base)


def load_test_refs(dataset_file: str) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # dataset lines include an "output" field that is a JSON string
            out_field = obj.get("output")
            if isinstance(out_field, str):
                try:
                    refs.append(json.loads(out_field))
                except Exception:
                    refs.append({})
            else:
                refs.append({})
    return refs


def normalize_labels_field(text: Optional[str]) -> List[str]:
    if not text:
        return [NONE_LABEL]
    s = text.strip()
    if not s:
        return [NONE_LABEL]
    # split by semicolon variants
    parts = [t.strip() for t in re.split(r"[;；]", s) if t.strip()]
    if not parts:
        return [NONE_LABEL]
    # If any NONE present, collapse to NONE only
    if any(p == NONE_LABEL for p in parts):
        return [NONE_LABEL]
    # keep only allowed and preserve order of ALLOWED_LABELS
    allowed = [p for p in parts if p in ALLOWED_LABELS and p != NONE_LABEL]
    if not allowed:
        return [NONE_LABEL]
    seen = set()
    ordered = []
    for label in ALLOWED_LABELS:
        if label == NONE_LABEL:
            continue
        if label in allowed and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def normalize_openness(text: Optional[str]) -> str:
    if not text:
        return "中性"
    s = text.strip()
    return s if s in OPENNESS_LABELS else "中性"


def compute_metrics(
    refs: List[Dict[str, Any]], preds: List[Dict[str, Any]]
) -> Dict[str, float]:
    n = min(len(refs), len(preds))
    refs = refs[:n]
    preds = preds[:n]

    if n == 0:
        return {
            "labels_micro_f1": 0.0,
            "labels_macro_f1": 0.0,
            "labels_jaccard": 0.0,
            "openness_accuracy": 0.0,
            "openness_f1": 0.0,
            "num_samples": 0,
        }

    # Build label mapping excluding NONE
    id_to_label = [l for l in ALLOWED_LABELS if l != NONE_LABEL]
    label_to_id = {l: i for i, l in enumerate(id_to_label)}

    # Multi-label arrays
    y_true = np.zeros((n, len(id_to_label)), dtype=int)
    y_pred = np.zeros((n, len(id_to_label)), dtype=int)

    # NONE stats
    none_true = np.zeros(n, dtype=int)
    none_pred = np.zeros(n, dtype=int)

    # Openness arrays
    openness_to_id = {l: i for i, l in enumerate(OPENNESS_LABELS)}
    o_true = np.zeros(n, dtype=int)
    o_pred = np.zeros(n, dtype=int)

    for i in range(n):
        r = refs[i] or {}
        p = preds[i] or {}

        true_labels = normalize_labels_field(r.get("本地人判定标准"))
        pred_labels = normalize_labels_field(p.get("本地人判定标准"))
        if true_labels == [NONE_LABEL]:
            none_true[i] = 1
        if pred_labels == [NONE_LABEL]:
            none_pred[i] = 1
        for l in true_labels:
            if l != NONE_LABEL:
                y_true[i, label_to_id[l]] = 1
        for l in pred_labels:
            if l != NONE_LABEL:
                y_pred[i, label_to_id[l]] = 1

        o_true[i] = openness_to_id.get(normalize_openness(r.get("开放倾向")), 1)
        o_pred[i] = openness_to_id.get(normalize_openness(p.get("开放倾向")), 1)

    # Scores
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    support_mask = (y_true.sum(axis=0) > 0)
    macro_f1 = (
        f1_score(y_true[:, support_mask], y_pred[:, support_mask], average="macro", zero_division=0)
        if support_mask.any()
        else 0.0
    )
    try:
        jacc = jaccard_score(y_true, y_pred, average="samples")
    except ValueError:
        jacc = jaccard_score(y_true, y_pred, average="macro")

    openness_acc = accuracy_score(o_true, o_pred)
    openness_f1 = f1_score(o_true, o_pred, average="weighted", zero_division=0)
    # Additional openness metrics
    openness_precision_macro = precision_score(o_true, o_pred, average="macro", zero_division=0)
    openness_precision_micro = precision_score(o_true, o_pred, average="micro", zero_division=0)
    openness_recall_macro = recall_score(o_true, o_pred, average="macro", zero_division=0)
    openness_recall_micro = recall_score(o_true, o_pred, average="micro", zero_division=0)

    return {
        "labels_micro_f1": float(micro_f1),
        "labels_macro_f1": float(macro_f1),
        "labels_jaccard": float(jacc),
        "openness_accuracy": float(openness_acc),
        "openness_f1": float(openness_f1),
        "openness_precision_macro": float(openness_precision_macro),
        "openness_precision_micro": float(openness_precision_micro),
        "openness_recall_macro": float(openness_recall_macro),
        "openness_recall_micro": float(openness_recall_micro),
        "num_samples": int(n),
    }


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_csv(path: str, header: List[str], row: List[Any]) -> None:
    is_new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


def plot_curve(csv_path: str, out_png: str, metric: str = "labels_micro_f1") -> None:
    steps: List[int] = []
    values: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx_step = header.index("step")
        idx_metric = header.index(metric)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            steps.append(int(parts[idx_step]))
            values.append(float(parts[idx_metric]))
    if not steps:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, marker="o")
    plt.xlabel("Checkpoint Step")
    plt.ylabel(metric)
    plt.title(f"Performance over checkpoints: {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)


def _get_refs_with_cache(dataset_dir: str, dataset_name: str, cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if dataset_name in cache:
        return cache[dataset_name]
    dataset_file = os.path.join(dataset_dir, f"{dataset_name}.jsonl")
    if not os.path.exists(dataset_file):
        sys.stderr.write(f"Warning: dataset file not found: {dataset_file}\n")
        cache[dataset_name] = []
        return cache[dataset_name]
    refs = load_test_refs(dataset_file)
    cache[dataset_name] = refs
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLaMA-Factory LoRA checkpoints and plot curve (no vLLM)")
    parser.add_argument("--exp_dir", required=False, help="Experiment directory containing checkpoint-*/")
    parser.add_argument("--checkpoints_root", default="/nfs/home/tanz/dd2/checkpoints", help="Root dir with multiple experiments")
    parser.add_argument("--base_model", default="google/gemma-3-27b-it")
    parser.add_argument("--template", default="gemma")
    parser.add_argument("--dataset_name", default="labels_withphrase_test")
    parser.add_argument("--dataset_dir", default="/nfs/home/tanz/dd2/annotated/lf_data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--output_root", default="/nfs/home/tanz/dd2/outputs/checkpoint")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_generation", action="store_true", help="Do not run model prediction; only evaluate existing prediction files")
    parser.add_argument("--max_checkpoints", type=int, default=None)
    parser.add_argument("--metric", default="labels_micro_f1", choices=[
        "labels_micro_f1", "labels_macro_f1", "labels_jaccard", "openness_accuracy", "openness_f1"
    ])
    parser.add_argument(
        "--experiment_filter",
        choices=["all", "labels_only", "labels_withphrase"],
        default="all",
        help="Filter which experiments to run by dataset group",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Limit generation length per sample (passed to llamafactory as --max_new_tokens)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip checkpoints already summarized and reuse existing prediction files",
    )
    args = parser.parse_args()

    cfg = EvalConfig(
        exp_dir=os.path.abspath(args.exp_dir) if args.exp_dir else os.path.abspath(args.checkpoints_root),
        base_model=args.base_model,
        template=args.template,
        dataset_name=args.dataset_name,
        dataset_dir=os.path.abspath(args.dataset_dir),
        batch_size=args.batch_size,
        bf16=bool(args.bf16),
        output_root=os.path.abspath(args.output_root),
        dry_run=bool(args.dry_run),
        max_checkpoints=args.max_checkpoints,
        max_new_tokens=args.max_new_tokens,
    )
    # Determine list of experiment directories
    if args.exp_dir:
        exp_dirs: List[str] = [os.path.abspath(args.exp_dir)]
    else:
        exp_dirs = list_experiment_dirs(os.path.abspath(args.checkpoints_root))
        if not exp_dirs:
            print(f"No experiments with checkpoints found in {args.checkpoints_root}")
            return

    # Apply experiment filter if requested
    if args.experiment_filter != "all":
        filtered: List[str] = []
        for d in exp_dirs:
            name = os.path.basename(d).lower()
            if args.experiment_filter == "labels_only" and "labels_only" in name:
                filtered.append(d)
            elif args.experiment_filter == "labels_withphrase" and (
                "withphrase" in name or "labels_phrase" in name or "labels_phrases" in name
            ):
                filtered.append(d)
            else:
                # Fallback to dataset detection from config if name is inconclusive
                ds = detect_predict_dataset(d) or ""
                ds_l = ds.lower()
                if args.experiment_filter == "labels_only" and "labels_only" in ds_l:
                    filtered.append(d)
                if args.experiment_filter == "labels_withphrase" and (
                    "withphrase" in ds_l or "labels_phrase" in ds_l or "labels_phrases" in ds_l
                ):
                    filtered.append(d)
        exp_dirs = filtered
        if not exp_dirs:
            print("No experiments match the filter; nothing to do.")
            return

    # Central output root where renamed predictions and summary live
    out_root = cfg.output_root
    os.makedirs(out_root, exist_ok=True)

    # References cache per dataset name
    refs_cache: Dict[str, List[Dict[str, Any]]] = {}

    summary_csv = os.path.join(out_root, "summary.csv")
    csv_header = [
        "step",
        "dataset",
        "labels_micro_f1",
        "labels_macro_f1",
        "labels_jaccard",
        "openness_accuracy",
        "openness_f1",
        "openness_precision_macro",
        "openness_precision_micro",
        "openness_recall_macro",
        "openness_recall_micro",
        "num_samples",
        "predictions_file",
        "eval_dir",
        "model",
    ]

    resume = bool(getattr(args, "resume", False))

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir.rstrip("/"))
        temp_eval_root = os.path.join(out_root, exp_name)
        os.makedirs(temp_eval_root, exist_ok=True)

        ckpts = list_checkpoints(exp_dir)
        if cfg.max_checkpoints is not None:
            ckpts = ckpts[: cfg.max_checkpoints]
        if not ckpts:
            print(f"No checkpoints found in {exp_dir}")
            continue

        for step, ckpt_path in ckpts:
            eval_dir = os.path.join(temp_eval_root, f"checkpoint-{step}")
            # Prefer detected base model; fall back to provided default
            detected_base = detect_base_model(exp_dir, ckpt_path)
            base_model_for_run = detected_base or cfg.base_model
            # Detect template and eval dataset per experiment
            template_for_run = detect_template(exp_dir, ckpt_path, base_model_for_run) or cfg.template
            dataset_for_run = detect_predict_dataset(exp_dir) or cfg.dataset_name

            # Determine renamed predictions destination first: model-slug + dataset-group + step
            template_name = _safe_filename(template_for_run)
            group_slug = dataset_group_slug(dataset_for_run, exp_dir)
            exp_lower = exp_name.lower()
            if "qwen" in exp_lower:
                model_slug = "qwen"
            elif "yi" in exp_lower:
                model_slug = "yi"
            elif "gemma" in exp_lower:
                model_slug = "gemma"
            else:
                model_slug = template_name
            pred_basename = f"{model_slug}+{group_slug}+{step}.jsonl"
            pred_dest = os.path.join(out_root, pred_basename)

            # Resume: if already summarized, skip
            if resume and summary_has_entry(summary_csv, pred_dest):
                print(f"[resume] Skipping step {step} at {exp_name}: already summarized -> {pred_dest}")
                continue

            # Run prediction or reuse existing files
            preds: List[Dict[str, Any]] = []
            pred_src = find_predictions_file(eval_dir)
            if pred_src is None:
                if os.path.exists(pred_dest):
                    preds = load_predictions_from_file(pred_dest)
                else:
                    if getattr(args, "skip_generation", False):
                        print(f"[skip_generation] No predictions found for {eval_dir}; skipping step {step}")
                        preds = []
                    else:
                        run_prediction_for_checkpoint(
                            cfg,
                            ckpt_path,
                            eval_dir,
                            base_model_override=base_model_for_run,
                            template_override=template_for_run,
                            dataset_name_override=dataset_for_run,
                        )
                        preds = load_predictions(eval_dir)
            else:
                preds = load_predictions(eval_dir)
                if not os.path.exists(pred_dest):
                    try:
                        shutil.copy2(pred_src, pred_dest)
                    except Exception:
                        pass

            # Load matching refs for the detected dataset
            refs = _get_refs_with_cache(cfg.dataset_dir, dataset_for_run, refs_cache)
            metrics = compute_metrics(refs, preds)
            save_json(os.path.join(eval_dir, "metrics.json"), metrics)
            # Also save per-checkpoint metrics next to the renamed predictions
            metrics_dest = os.path.splitext(pred_dest)[0] + ".metrics.json"
            save_json(metrics_dest, metrics)

            append_csv(
                summary_csv,
                csv_header,
                [
                    step,
                    dataset_for_run,
                    metrics.get("labels_micro_f1", 0.0),
                    metrics.get("labels_macro_f1", 0.0),
                    metrics.get("labels_jaccard", 0.0),
                    metrics.get("openness_accuracy", 0.0),
                    metrics.get("openness_f1", 0.0),
                    metrics.get("openness_precision_macro", 0.0),
                    metrics.get("openness_precision_micro", 0.0),
                    metrics.get("openness_recall_macro", 0.0),
                    metrics.get("openness_recall_micro", 0.0),
                    metrics.get("num_samples", 0),
                    pred_dest,
                    eval_dir,
                    model_slug,
                ],
            )

    # Plot curve
    out_png = os.path.join(out_root, f"perf_curve_{args.metric}.png")
    plot_curve(summary_csv, out_png, metric=args.metric)
    print(f"Done. Summary: {summary_csv}\nPlot: {out_png}")


if __name__ == "__main__":
    main()


