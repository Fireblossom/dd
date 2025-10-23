#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score, hamming_loss, precision_recall_fscore_support
from torch.utils.data import Dataset
import inspect
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    Trainer,
    TrainingArguments,
)

ALLOWED_LABELS = [
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

# Openness tendency labels
OPENNESS_LABELS = ["紧缩排斥", "中性", "宽容开放"]


def infer_city(sample_id: str) -> str:
    return sample_id.split("_")[0]


def parse_labels(label_field: str) -> List[str]:
    if label_field is None:
        return []
    label_field = label_field.strip()
    if not label_field:
        return []
    try:
        labels = json.loads(label_field)
        if isinstance(labels, list):
            return [str(x).strip() for x in labels]
        return []
    except Exception:
        # Try parsing by semicolon (support both ASCII and full-width)
        seps = [';', '；']
        for sep in seps:
            if sep in label_field:
                parts = [t.strip() for t in label_field.split(sep) if t.strip()]
                return parts
        return []


def normalize_labels(labels: List[str]) -> List[str]:
    if not labels:
        return [NONE_LABEL]
    cleaned = [str(x).strip() for x in labels if isinstance(x, str)]
    # NONE label veto: if present, keep only NONE
    if any(l == NONE_LABEL for l in cleaned):
        return [NONE_LABEL]
    # Keep only allowed labels (excluding NONE)
    filtered = [l for l in cleaned if l in ALLOWED_LABELS and l != NONE_LABEL]
    if not filtered:
        return [NONE_LABEL]
    # Deduplicate and sort by ALLOWED_LABELS order for stable output
    order = {l: i for i, l in enumerate(ALLOWED_LABELS)}
    filtered_sorted = sorted(set(filtered), key=lambda l: order[l])
    return filtered_sorted


def parse_openness(openness_field: str) -> str:
    """Parse openness tendency label"""
    if openness_field is None:
        return "中性"
    openness_field = openness_field.strip()
    if not openness_field:
        return "中性"
    # Check if in allowed labels
    if openness_field in OPENNESS_LABELS:
        return openness_field
    # Default to neutral if not in allowed list
    return "中性"


def read_split_csv(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        id_i = idx["id"]
        text_i = idx["text"]
        # Support different CSV header formats (ASCII/full-width quotes, etc.)
        def find_col(d: Dict[str, int], candidates: List[str], substrings: List[str]) -> int:
            for name in candidates:
                if name in d:
                    return d[name]
            # Fallback: fuzzy match column names containing all substrings
            for name in d.keys():
                if all(s in name for s in substrings):
                    return d[name]
            raise KeyError(f"Column not found. Tried {candidates}, looked for substrings {substrings}. Got headers: {list(d.keys())}")

        labels_i = find_col(
            idx,
            [
                "结果字段-\"本地人\"身份认定",  # ASCII quotes
                "结果字段-"本地人"身份认定",   # Full-width quotes
            ],
            ["本地人", "身份认定"],
        )

        openness_i = find_col(
            idx,
            [
                "结果字段-宽容或严格",
            ],
            ["宽容", "严格"],
        )
        for row in reader:
            labels = normalize_labels(parse_labels(row[labels_i]))
            openness = parse_openness(row[openness_i])
            out.append({
                "id": row[id_i], 
                "city": infer_city(row[id_i]), 
                "text": row[text_i], 
                "labels": labels,
                "openness": openness
            })
    return out


def build_label_space(records: List[Dict[str, Any]]) -> List[str]:
    # Fix label space to allowed set; exclude NONE during training (NONE is inference fallback only)
    return [l for l in ALLOWED_LABELS if l != NONE_LABEL]


class TextDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, label_to_id: Dict[str, int], openness_to_id: Dict[str, int], max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.openness_to_id = openness_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(
            r["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Remove batch dimension
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        
        # Multi-label classification labels
        label_vec = np.zeros(len(self.label_to_id), dtype=np.float32)
        for l in r["labels"]:
            if l == NONE_LABEL:
                continue
            idx = self.label_to_id.get(l)
            if idx is not None:
                label_vec[idx] = 1.0
        enc["labels"] = torch.tensor(label_vec)
        
        # Openness tendency classification label
        openness_label = self.openness_to_id[r["openness"]]
        enc["openness_labels"] = torch.tensor(openness_label, dtype=torch.long)
        
        return enc


class MultiTaskModel(nn.Module):
    """Multi-task model: Localness Criteria (multi-label) + Openness Tendency (3-class)"""
    
    def __init__(self, model_name: str, num_labels: int, num_openness_labels: int, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.pos_weight = pos_weight  # shape: [num_labels] or None
        
        # Localness criteria classifier (multi-label)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Openness tendency classifier (3-class)
        self.openness_classifier = nn.Linear(self.bert.config.hidden_size, num_openness_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                labels=None, openness_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Compatible with backbones without pooler, fallback to mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled_output = summed / lengths
        pooled_output = self.dropout(pooled_output)
        
        # Localness criteria prediction
        logits = self.classifier(pooled_output)
        
        # Openness tendency prediction
        openness_logits = self.openness_classifier(pooled_output)
        
        loss = None
        if labels is not None and openness_labels is not None:
            # Multi-label classification loss (use pos_weight for class imbalance)
            if self.pos_weight is not None:
                pw = self.pos_weight.to(logits.device)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
            labels_loss = loss_fct(logits, labels.float())
            
            # 3-class classification loss
            openness_loss_fct = nn.CrossEntropyLoss()
            openness_loss = openness_loss_fct(openness_logits, openness_labels)
            
            # Total loss (weights can be adjusted)
            loss = labels_loss + 0.5 * openness_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'openness_logits': openness_logits
        }


class MultiTaskTrainer(Trainer):
    """自定义Trainer处理多任务学习，兼容Transformers新旧API"""

    # Compatible with transformers >= 4.44: add optional num_items_in_batch parameter
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss

    # Compatible with new transformers prediction_step signature
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
        num_items_in_batch=None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.get("loss")

            logits = outputs.get("logits")
            openness_logits = outputs.get("openness_logits")

            if prediction_loss_only:
                return (loss, None, None)

            labels = inputs.get("labels")
            openness_labels = inputs.get("openness_labels")

            return (loss, (logits, openness_logits), (labels, openness_labels))


def compute_metrics_builder(id_to_label: List[str], id_to_openness: List[str]):
    def compute(eval_pred):
        # Process multi-task prediction results
        (logits, openness_logits), (labels, openness_labels) = eval_pred
        
        # Localness criteria metrics
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        # Compute Macro F1 only on classes present in test set to avoid zero-class bias
        support_mask = (labels.sum(axis=0) > 0)
        if support_mask.any():
            macro_f1 = f1_score(labels[:, support_mask], preds[:, support_mask], average="macro", zero_division=0)
        else:
            macro_f1 = 0.0
        jacc = jaccard_score(labels, preds, average="samples")
        
        # Openness tendency metrics
        openness_preds = np.argmax(openness_logits, axis=-1)
        openness_acc = accuracy_score(openness_labels, openness_preds)
        openness_f1 = f1_score(openness_labels, openness_preds, average="weighted", zero_division=0)
        
        return {
            "labels_micro_f1": micro_f1, 
            "labels_macro_f1": macro_f1, 
            "labels_jaccard": jacc,
            "openness_accuracy": openness_acc,
            "openness_f1": openness_f1
        }
    return compute


def evaluate_per_city(trainer: Trainer, dataset: TextDataset, records: List[Dict[str, Any]], id_to_label: List[str], out_dir: str, threshold: float = 0.5):
    preds = trainer.predict(dataset)
    (logits, openness_logits) = preds.predictions
    probs = 1 / (1 + np.exp(-logits))
    y_true = preds.label_ids[0]  # 多标签分类的真实标签
    y_pred = (probs >= threshold).astype(int)
    # group by city
    per_city = {}
    for i, r in enumerate(records):
        city = r["city"]
        per_city.setdefault(city, {"true": [], "pred": []})
        per_city[city]["true"].append(y_true[i])
        per_city[city]["pred"].append(y_pred[i])
    summary = {}
    for city, d in per_city.items():
        y_t = np.array(d["true"]) ; y_p = np.array(d["pred"]) 
        micro_f1 = f1_score(y_t, y_p, average="micro", zero_division=0)
        macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        jacc = jaccard_score(y_t, y_p, average="samples")
        summary[city] = {"micro_f1": float(micro_f1), "macro_f1": float(macro_f1), "jaccard": float(jacc)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_city_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def comprehensive_evaluation(records: List[Dict[str, Any]], probs: np.ndarray, openness_probs: np.ndarray, 
                           id_to_label: List[str], id_to_openness: List[str], threshold: float = 0.5) -> Dict[str, Any]:
    """进行全面的评估，类似于 evaluate_multilabel.py 的功能"""
    
    # Prepare multi-label classification evaluation
    y_true = []
    y_pred = []
    none_true = []
    none_pred = []
    
    for i, r in enumerate(records):
        # True labels
        true_labels = r["labels"]
        true_vec = np.zeros(len(id_to_label), dtype=int)
        for label in true_labels:
            if label in id_to_label:
                true_vec[id_to_label.index(label)] = 1
        y_true.append(true_vec)
        
        # Predicted labels
        pred_vec = (probs[i] >= threshold).astype(int)
        y_pred.append(pred_vec)
        
        # NONE label statistics: true if only NONE, predicted if no labels selected
        none_true.append(1 if (len(true_labels) == 1 and true_labels[0] == NONE_LABEL) else 0)
        none_pred.append(1 if pred_vec.sum() == 0 else 0)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute multi-label classification metrics
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    # Compute Macro F1 only for classes present in test set
    support_mask = (y_true.sum(axis=0) > 0)
    if support_mask.any():
        macro_f1 = f1_score(y_true[:, support_mask], y_pred[:, support_mask], average="macro", zero_division=0)
    else:
        macro_f1 = 0.0
    
    # Compute per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Build per-class F1 dictionary
    class_f1_dict = {}
    for i, label in enumerate(id_to_label):
        class_f1_dict[label] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i])
        }
    
    # Compute Jaccard similarity (sample-level), fallback to macro on error
    try:
        jacc = jaccard_score(y_true, y_pred, average="samples")
    except ValueError:
        jacc = jaccard_score(y_true, y_pred, average="macro")
    
    hamm = hamming_loss(y_true, y_pred)
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))
    
    # NONE label metrics
    p_none, r_none, f1_none, _ = precision_recall_fscore_support(
        np.array(none_true), np.array(none_pred), average="binary", zero_division=0
    )
    
    # Openness tendency evaluation
    openness_true = []
    openness_pred = []
    for i, r in enumerate(records):
        openness_true.append(id_to_openness.index(r["openness"]))
        openness_pred.append(np.argmax(openness_probs[i]))
    
    openness_acc = accuracy_score(openness_true, openness_pred)
    # Openness tendency metrics with various averaging methods
    openness_f1_macro = f1_score(openness_true, openness_pred, average="macro", zero_division=0)
    openness_f1_micro = f1_score(openness_true, openness_pred, average="micro", zero_division=0)
    openness_f1 = f1_score(openness_true, openness_pred, average="weighted", zero_division=0)
    openness_precision_macro = precision_score(openness_true, openness_pred, average="macro", zero_division=0)
    openness_precision_micro = precision_score(openness_true, openness_pred, average="micro", zero_division=0)
    openness_recall_macro = recall_score(openness_true, openness_pred, average="macro", zero_division=0)
    openness_recall_micro = recall_score(openness_true, openness_pred, average="micro", zero_division=0)
    
    result = {
        "labels_micro_f1": float(micro_f1),
        "labels_macro_f1": float(macro_f1),
        "labels_jaccard": float(jacc),
        "labels_hamming_loss": float(hamm),
        "labels_exact_match": exact_match,
        "labels_none_precision": float(p_none),
        "labels_none_recall": float(r_none),
        "labels_none_f1": float(f1_none),
        "labels_num_samples": int(len(records)),
        "labels_num_classes": int(len(id_to_label)),
        "labels_class_f1_scores": class_f1_dict,
        "openness_accuracy": float(openness_acc),
        "openness_f1_macro": float(openness_f1_macro),
        "openness_f1_micro": float(openness_f1_micro),
        "openness_f1": float(openness_f1),
        "openness_precision_macro": float(openness_precision_macro),
        "openness_precision_micro": float(openness_precision_micro),
        "openness_recall_macro": float(openness_recall_macro),
        "openness_recall_micro": float(openness_recall_micro),
        "openness_classes": id_to_openness,
        "openness_num_samples": int(len(records))
    }
    
    return result


def write_strict_jsonl_predictions(records: List[Dict[str, Any]], probs: np.ndarray, openness_probs: np.ndarray, 
                                 id_to_label: List[str], id_to_openness: List[str], out_path: str, threshold: float = 0.5):
    """Write predictions in strict 3-key JSONL format required by improved prompt.

    The output schema per line:
      {"本地人判定标准": "标签1; 标签2" 或 "以上标准皆不符合|未提及任何标准", "开放倾向": "紧缩排斥/中性/宽容开放"}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            # Localness criteria prediction
            p = probs[i]
            pred_indices = [j for j, v in enumerate(p) if v >= threshold]
            if not pred_indices:
                labels_out = NONE_LABEL
            else:
                labels_out = "; ".join([id_to_label[j] for j in pred_indices])
            
            # Openness tendency prediction
            openness_pred = np.argmax(openness_probs[i])
            openness_out = id_to_openness[openness_pred]
            
            obj = {
                "本地人判定标准": labels_out,
                "开放倾向": openness_out,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", default="/nfs/home/tanz/dd2/annotated/splits")
    parser.add_argument("--output_dir", default="/nfs/home/tanz/dd2/annotated/baseline_outputs")
    parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("--heldout_city", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if args.heldout_city:
        # LOOCV mode
        cities = sorted({fn.split("_")[0] for fn in os.listdir(args.splits_dir) if fn.endswith("_train.csv")})
        others = [c for c in cities if c != args.heldout_city]
        train_records = []
        dev_records = []
        for c in others:
            train_records += read_split_csv(os.path.join(args.splits_dir, f"{c}_train.csv"))
            dev_records += read_split_csv(os.path.join(args.splits_dir, f"{c}_dev.csv"))
        test_records = read_split_csv(os.path.join(args.splits_dir, f"{args.heldout_city}_test.csv"))
        out_dir = os.path.join(args.output_dir, f"loocv_{args.heldout_city}")
    else:
        train_records = read_split_csv(os.path.join(args.splits_dir, "all_train.csv"))
        dev_records = read_split_csv(os.path.join(args.splits_dir, "all_dev.csv"))
        test_records = read_split_csv(os.path.join(args.splits_dir, "all_test.csv"))
        out_dir = os.path.join(args.output_dir, "all")

    label_space = build_label_space(train_records)
    label_to_id = {l: i for i, l in enumerate(label_space)}
    id_to_label = label_space
    
    # Openness tendency label mapping
    openness_to_id = {l: i for i, l in enumerate(OPENNESS_LABELS)}
    id_to_openness = OPENNESS_LABELS

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # 计算正类权重 pos_weight（基于训练集标签频次）
    label_counts = np.zeros(len(label_space), dtype=np.float32)
    for r in train_records:
        for l in r["labels"]:
            if l == NONE_LABEL:
                continue
            j = label_to_id.get(l)
            if j is not None:
                label_counts[j] += 1
    num_samples = max(1, len(train_records))
    # 避免除零；简单设置为 negative/positive 比
    pos = np.clip(label_counts, 1.0, None)
    neg = num_samples - pos
    neg = np.clip(neg, 1.0, None)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32)

    model = MultiTaskModel(
        model_name=args.model,
        num_labels=len(label_space),
        num_openness_labels=len(OPENNESS_LABELS),
        pos_weight=pos_weight,
    )

    train_ds = TextDataset(train_records, tokenizer, label_to_id, openness_to_id)
    dev_ds = TextDataset(dev_records, tokenizer, label_to_id, openness_to_id)
    test_ds = TextDataset(test_records, tokenizer, label_to_id, openness_to_id)

    # Build TrainingArguments with backward/forward compatibility across transformers versions
    sig = inspect.signature(TrainingArguments.__init__)
    param_names = set(sig.parameters.keys())

    args_kwargs = {
        "output_dir": os.path.join(out_dir, "checkpoints"),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "logging_steps": 50,
    }
    if "report_to" in param_names:
        args_kwargs["report_to"] = []
    if "fp16" in param_names:
        args_kwargs["fp16"] = torch.cuda.is_available()
    # 简化配置，避免策略冲突
    if "evaluation_strategy" in param_names:
        args_kwargs["evaluation_strategy"] = "no"
    if "save_strategy" in param_names:
        args_kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(**args_kwargs)

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_builder(id_to_label, id_to_openness),
    )

    trainer.train()
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)

    # Evaluate on test and per-city
    metrics = trainer.evaluate(test_ds)
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)

    # Write strict JSONL predictions for test set
    test_preds = trainer.predict(test_ds)
    (test_logits, test_openness_logits) = test_preds.predictions
    test_probs = 1 / (1 + np.exp(-test_logits))
    test_openness_probs = torch.softmax(torch.tensor(test_openness_logits), dim=-1).numpy()

    # 简单阈值调优：在开发集上网格搜索最佳阈值（提升召回，避免全NONE）
    dev_preds = trainer.predict(dev_ds)
    (dev_logits, _) = dev_preds.predictions
    dev_probs = 1 / (1 + np.exp(-dev_logits))
    dev_labels = dev_preds.label_ids[0]
    best_t, best_micro = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        yhat = (dev_probs >= t).astype(int)
        micro = f1_score(dev_labels, yhat, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro, best_t = micro, float(t)
    with open(os.path.join(out_dir, "best_threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": best_t, "dev_micro_f1": best_micro}, f, ensure_ascii=False, indent=2)
    
    # 进行全面的评估，类似于 evaluate_multilabel.py
    comprehensive_metrics = comprehensive_evaluation(
        records=test_records,
        probs=test_probs,
        openness_probs=test_openness_probs,
        id_to_label=id_to_label,
        id_to_openness=id_to_openness,
        threshold=best_t,
    )
    
    # 保存全面评估结果
    with open(os.path.join(out_dir, "comprehensive_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(comprehensive_metrics, f, ensure_ascii=False, indent=2)
    
    # 打印评估结果
    print("=== 全面评估结果 ===")
    print(f"多标签分类 - Micro F1: {comprehensive_metrics['labels_micro_f1']:.4f}")
    print(f"多标签分类 - Macro F1: {comprehensive_metrics['labels_macro_f1']:.4f}")
    print(f"多标签分类 - Jaccard: {comprehensive_metrics['labels_jaccard']:.4f}")
    print(f"多标签分类 - Exact Match: {comprehensive_metrics['labels_exact_match']:.4f}")
    print(f"开放倾向 - Accuracy: {comprehensive_metrics['openness_accuracy']:.4f}")
    print(f"开放倾向 - F1: {comprehensive_metrics['openness_f1']:.4f}")
    
    write_strict_jsonl_predictions(
        records=test_records,
        probs=test_probs,
        openness_probs=test_openness_probs,
        id_to_label=id_to_label,
        id_to_openness=id_to_openness,
        out_path=os.path.join(out_dir, "predictions.jsonl"),
        threshold=best_t,
    )

    evaluate_per_city(trainer, test_ds, test_records, id_to_label, out_dir, threshold=best_t)


if __name__ == "__main__":
    main()


