#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from typing import List, Tuple


def list_checkpoints(exp_dir: str) -> List[Tuple[int, str]]:
    pairs = []
    for name in os.listdir(exp_dir):
        full = os.path.join(exp_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m:
            pairs.append((int(m.group(1)), full))
    pairs.sort(key=lambda t: t[0])
    return pairs


def run_predict(
    base_model: str,
    template: str,
    adapter_path: str,
    dataset_name: str,
    dataset_dir: str,
    output_dir: str,
    batch_size: int,
    bf16: bool,
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "llamafactory-cli",
        "train",
        "--stage",
        "sft",
        "--model_name_or_path",
        base_model,
        "--template",
        template,
        "--finetuning_type",
        "lora",
        "--adapter_name_or_path",
        adapter_path,
        "--output_dir",
        output_dir,
        "--overwrite_output_dir",
        "True",
        "--dataset",
        dataset_name,
        "--dataset_dir",
        dataset_dir,
        "--do_train",
        "False",
        "--do_eval",
        "False",
        "--do_predict",
        "True",
        "--predict_with_generate",
        "True",
        "--per_device_eval_batch_size",
        str(batch_size),
        "--report_to",
        "none",
    ]
    if bf16:
        cmd += ["--bf16", "True"]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"[ERROR] predict failed for {adapter_path}: {res.stderr}\n")
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Collect predictions for all checkpoints (no metrics)")
    parser.add_argument("--exp_dir", required=True, help="Directory that contains checkpoint-*/")
    parser.add_argument("--base_model", default="google/gemma-3-27b-it")
    parser.add_argument("--template", default="gemma")
    parser.add_argument("--dataset_name", default="labels_withphrase_test")
    parser.add_argument("--dataset_dir", default="/nfs/home/tanz/dd2/annotated/lf_data")
    parser.add_argument("--output_root", default="/nfs/home/tanz/dd2/annotated/ckpt_preds")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    out_root = os.path.abspath(args.output_root)
    os.makedirs(out_root, exist_ok=True)

    ckpts = list_checkpoints(exp_dir)
    if not ckpts:
        print(f"No checkpoints found in {exp_dir}")
        return

    exp_name = os.path.basename(exp_dir.rstrip("/"))
    exp_out = os.path.join(out_root, exp_name)
    os.makedirs(exp_out, exist_ok=True)

    for step, path in ckpts:
        out_dir = os.path.join(exp_out, f"checkpoint-{step}")
        # skip if predictions already present
        already = any(
            os.path.exists(os.path.join(out_dir, fname))
            for fname in [
                "generated_predictions.jsonl",
                "inference_results.jsonl",
                "inference_results_v2.jsonl",
                "serve_inference_results.jsonl",
            ]
        )
        if already:
            print(f"Skip step {step} (predictions exist)")
            continue
        rc = run_predict(
            base_model=args.base_model,
            template=args.template,
            adapter_path=path,
            dataset_name=args.dataset_name,
            dataset_dir=os.path.abspath(args.dataset_dir),
            output_dir=out_dir,
            batch_size=args.batch_size,
            bf16=bool(args.bf16),
        )
        if rc != 0:
            print(f"Step {step} FAILED (see stderr above)")
        else:
            print(f"Step {step} done -> {out_dir}")


if __name__ == "__main__":
    main()


