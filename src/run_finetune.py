import os
import sys
import subprocess
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import argparse
import re

# Configuration parameters
BASE_DIR = "/nfs/home/tanz/dd2/annotated"
LLAMAFACTORY_DIR = "/nfs/home/tanz/dd2/LLaMA-Factory"
LOG_DIR = f"{BASE_DIR}/logs"
RESULTS_DIR = f"{BASE_DIR}/results"
PROGRESS_FILE = f"{RESULTS_DIR}/experiment_progress.jsonl"

# Checkpoint evaluation strategy (can be overridden via environment variables)
CKPT_EVAL_STRATEGY = os.environ.get("CKPT_EVAL_STRATEGY", "all")  # all | interval | last
CKPT_EVAL_INTERVAL = int(os.environ.get("CKPT_EVAL_INTERVAL", "1"))     # Evaluate every N checkpoints
CKPT_EVAL_MAX = int(os.environ.get("CKPT_EVAL_MAX", "0"))               # Max checkpoints to evaluate, 0 = no limit

# Inference batch size (can be overridden via environment variables)
ZEROSHOT_EVAL_BS = int(os.environ.get("ZEROSHOT_EVAL_BS", "128"))      # Zero-shot predict-only per_device_eval_batch_size
PREDICT_EVAL_BS = int(os.environ.get("PREDICT_EVAL_BS", "128"))        # Other predict-only per_device_eval_batch_size (e.g. LOOCV fallback)
PEFT_INFER_BS = int(os.environ.get("PEFT_INFER_BS", "128"))            # peft_infer_swap_adapters.py --batch_size
ZEROSHOT_CUDA = os.environ.get("ZEROSHOT_CUDA", "0")

# Save strategy (can be overridden via environment variables)
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "10"))

# Model configuration
MODELS = {
    "qwen": "Qwen/Qwen3-32B",
    "yi": "01-ai/Yi-1.5-34B-Chat", 
    "gemma": "google/gemma-3-27b-it"
}

TEMPLATES = {
    "qwen": "qwen",
    "yi": "yi",
    "gemma": "gemma"
}

# Setup logging
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(f"{LOG_DIR}/experiment.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def run_command(cmd: List[str], log_file: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run command and log output"""
    logger.info(f"Running command: {' '.join(cmd)}")
    
    if log_file:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"Return code: {result.returncode}")
        if result.stdout:
            logger.error(f"Stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"Stderr: {result.stderr}")
        raise RuntimeError(f"Command execution failed: {' '.join(cmd)}")
    
    return result

def record_progress(event: Dict[str, Any]) -> None:
    """Record experiment progress to JSONL for later plotting."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    event_with_time = {
        **event,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_with_time, ensure_ascii=False) + "\n")

def record_metrics(model: str, phase: str, task: str, dataset: str, pred_path: str, metrics_path: str, extra: Dict[str, Any] = None) -> None:
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {"error": "failed_to_read_metrics"}
    payload = {
        "model": model,
        "phase": phase,           # zeroshot | finetune | loocv
        "task": task,             # labels_only | labels_withphrase | tone
        "dataset": dataset,
        "pred_file": pred_path,
        "metrics_file": metrics_path,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    record_progress(payload)

def list_checkpoints(output_dir: str) -> List[tuple]:
    """List checkpoint-<step> subdirectories, sorted by step."""
    checkpoints = []
    if not os.path.isdir(output_dir):
        return checkpoints
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            m = re.match(r"checkpoint-(\d+)", name)
            if m:
                step = int(m.group(1))
                checkpoints.append((step, path))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def select_checkpoints(checkpoints: List[tuple]) -> List[tuple]:
    """Filter checkpoints for evaluation based on strategy, reducing loading overhead."""
    if not checkpoints:
        return []
    if CKPT_EVAL_STRATEGY == "all":
        selected = checkpoints
    elif CKPT_EVAL_STRATEGY == "last":
        selected = [checkpoints[-1]]
    else:  # interval
        interval = max(1, CKPT_EVAL_INTERVAL)
        selected = checkpoints[::interval]
        if selected[-1][0] != checkpoints[-1][0]:
            selected.append(checkpoints[-1])
    if CKPT_EVAL_MAX > 0 and len(selected) > CKPT_EVAL_MAX:
        # Keep last CKPT_EVAL_MAX checkpoints (covering latest progress)
        selected = selected[-CKPT_EVAL_MAX:]
    return selected

def prepare_data():
    """Prepare data splits"""
    logger.info("Starting data preparation...")
    
    # Create necessary directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/outputs", exist_ok=True)
    
    # 1. Generate city-stratified data
    logger.info("Generating city-stratified data...")
    run_command([
        "python3", f"{BASE_DIR}/make_splits_and_jsonl.py",
        "--csv", f"{BASE_DIR}/merged.csv",
        "--outdir", f"{BASE_DIR}/splits",
        "--lf_outdir", f"{BASE_DIR}/lf_data"
    ], f"{LOG_DIR}/data_prep.log")
    
    # 2. Generate LOOCV data
    logger.info("Generating LOOCV data...")
    run_command([
        "python3", f"{BASE_DIR}/make_loocv_jsonl.py",
        "--splits_dir", f"{BASE_DIR}/splits",
        "--out_dir", f"{BASE_DIR}/lf_data/loocv"
    ], f"{LOG_DIR}/loocv_data.log")
    
    logger.info("Data preparation complete")
    # Clear previous progress file
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
    except Exception:
        pass

def run_zeroshot_experiments(model_name: str, model_path: str, template: str):
    """Zero-shot experiments - using LLaMA-Factory predict-only, no vLLM"""
    logger.info(f"Starting {model_name} zero-shot experiments (LLaMA-Factory)...")
    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ZEROSHOT_CUDA
    
    # labels_only_test prediction
    zs_lo_out = f"{BASE_DIR}/outputs/zs_{model_name}_labels_only"
    os.makedirs(zs_lo_out, exist_ok=True)
    run_command([
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--template", template,
        "--finetuning_type", "full",
        "--output_dir", zs_lo_out,
        "--overwrite_output_dir", "True",
        "--dataset", "labels_only_test",
        "--eval_dataset", "labels_only_test",
        "--dataset_dir", f"{BASE_DIR}/lf_data",
        "--do_train", "False", "--do_eval", "False",
        "--do_predict", "True", "--predict_with_generate", "True",
        "--max_new_tokens", "256",
        "--per_device_eval_batch_size", str(ZEROSHOT_EVAL_BS),
        "--bf16", "True",
        "--report_to", "none"
    ], f"{LOG_DIR}/{model_name}_zeroshot_labels.log")
    pred_lo = f"{zs_lo_out}/generated_predictions.jsonl"
    metrics_lo = f"{BASE_DIR}/outputs/metrics_zeroshot_{model_name}_labels_only.json"
    run_command([
        "python3", f"{BASE_DIR}/evaluate_multilabel.py",
        "--pred_file", pred_lo,
        "--data_file", f"{BASE_DIR}/lf_data/labels_only_test.jsonl",
        "--out", metrics_lo
    ], f"{LOG_DIR}/{model_name}_eval_labels.log")
    record_metrics(model_name, "zeroshot", "labels_only", "labels_only_test", pred_lo, metrics_lo)
    
    # labels_withphrase_test prediction
    zs_lp_out = f"{BASE_DIR}/outputs/zs_{model_name}_labels_withphrase"
    os.makedirs(zs_lp_out, exist_ok=True)
    run_command([
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--template", template,
        "--finetuning_type", "full",
        "--output_dir", zs_lp_out,
        "--overwrite_output_dir", "True",
        "--dataset", "labels_withphrase_test",
        "--eval_dataset", "labels_withphrase_test",
        "--dataset_dir", f"{BASE_DIR}/lf_data",
        "--do_train", "False", "--do_eval", "False",
        "--do_predict", "True", "--predict_with_generate", "True",
        "--max_new_tokens", "512",
        "--per_device_eval_batch_size", str(ZEROSHOT_EVAL_BS),
        "--bf16", "True",
        "--report_to", "none"
    ], f"{LOG_DIR}/{model_name}_zeroshot_phrase.log")
    pred_lp = f"{zs_lp_out}/generated_predictions.jsonl"
    metrics_lp = f"{BASE_DIR}/outputs/metrics_zeroshot_{model_name}_labels_withphrase.json"
    run_command([
        "python3", f"{BASE_DIR}/evaluate_multilabel.py",
        "--pred_file", pred_lp,
        "--data_file", f"{BASE_DIR}/lf_data/labels_withphrase_test.jsonl",
        "--out", metrics_lp
    ], f"{LOG_DIR}/{model_name}_eval_phrase.log")
    record_metrics(model_name, "zeroshot", "labels_withphrase", "labels_withphrase_test", pred_lp, metrics_lp)
    
    if prev_cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    logger.info(f"{model_name} zero-shot experiments complete (LLaMA-Factory)")

def run_finetuning_experiments(model_name: str, model_path: str, template: str):
    """Fine-tuning experiments"""
    logger.info(f"Starting {model_name} fine-tuning experiments...")
    
    # labels-only fine-tuning
    logger.info(f"Running {model_name} labels-only fine-tuning...")
    run_command([
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--template", template,
        "--finetuning_type", "lora",
        "--output_dir", f"{BASE_DIR}/outputs/{model_name}_labels_only",
        "--overwrite_output_dir", "True",
        "--dataset", "labels_only_train",
        "--eval_dataset", "labels_only_test",
        "--dataset_dir", f"{BASE_DIR}/lf_data",
        "--do_train", "True", "--do_eval", "False", "--do_predict", "True", "--predict_with_generate", "True",
        "--learning_rate", "2e-4", "--per_device_train_batch_size", "4", "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "16", "--gradient_checkpointing", "True", "--num_train_epochs", "4", "--bf16", "True",
        "--eval_strategy", "steps", "--eval_steps", str(SAVE_STEPS), "--save_steps", str(SAVE_STEPS), "--save_total_limit", str(SAVE_TOTAL_LIMIT),
        "--save_only_model", "True",
        "--lora_r", "32", "--lora_alpha", "128", "--lora_dropout", "0.05", "--report_to", "none"
    ], f"{LOG_DIR}/{model_name}_finetune_labels.log")
    # Build batched PEFT jobs (load base model once, switch LoRA adapters)
    jobs: List[dict] = []
    ckpts_all = list_checkpoints(f"{BASE_DIR}/outputs/{model_name}_labels_only")
    ckpts = select_checkpoints(ckpts_all)
    logger.info(f"labels_only checkpoints: total={len(ckpts_all)}, selected={len(ckpts)} (strategy={CKPT_EVAL_STRATEGY})")
    for step, ckpt_dir in ckpts:
        # labels_only on labels_only_test
        pred_step_lo = f"{BASE_DIR}/outputs/{model_name}_labels_only_ckpt{step}_labels_only_eval.jsonl"
        jobs.append({
            "adapter_dir": ckpt_dir,
            "adapter_name": f"{model_name}_labels_only_{step}",
            "dataset": "labels_only_test",
            "max_new_tokens": 256,
            "out_path": pred_step_lo,
        })
        # tone on labels_withphrase_test
        pred_step_tone = f"{BASE_DIR}/outputs/{model_name}_labels_only_ckpt{step}_on_withphrase.jsonl"
        jobs.append({
            "adapter_dir": ckpt_dir,
            "adapter_name": f"{model_name}_labels_only_{step}",
            "dataset": "labels_withphrase_test",
            "max_new_tokens": 512,
            "out_path": pred_step_tone,
        })
    
    # Evaluate labels-only fine-tuning results
    pred_ft_lo = f"{BASE_DIR}/outputs/{model_name}_labels_only/generated_predictions.jsonl"
    metrics_ft_lo = f"{BASE_DIR}/outputs/metrics_{model_name}_labels_only.json"
    run_command([
        "python3", f"{BASE_DIR}/evaluate_multilabel.py",
        "--pred_file", pred_ft_lo,
        "--data_file", f"{BASE_DIR}/lf_data/labels_withphrase_test.jsonl",
        "--out", metrics_ft_lo
    ], f"{LOG_DIR}/{model_name}_eval_finetune_labels.log")
    record_metrics(model_name, "finetune", "labels_only", "labels_withphrase_test", pred_ft_lo, metrics_ft_lo)
    
    # Include final adapter inference on phrase test set in batched jobs
    final_adapter_dir = f"{BASE_DIR}/outputs/{model_name}_labels_only"
    pred_tone_final = f"{BASE_DIR}/outputs/{model_name}_labels_only_on_withphrase.jsonl"
    jobs.append({
        "adapter_dir": final_adapter_dir,
        "adapter_name": f"{model_name}_labels_only_final",
        "dataset": "labels_withphrase_test",
        "max_new_tokens": 512,
        "out_path": pred_tone_final,
    })
    
    # labels_withphrase fine-tuning
    logger.info(f"Running {model_name} labels_withphrase fine-tuning...")
    run_command([
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--template", template,
        "--finetuning_type", "lora",
        "--output_dir", f"{BASE_DIR}/outputs/{model_name}_labels_phrase",
        "--overwrite_output_dir", "True",
        "--dataset", "labels_withphrase_train",
        "--eval_dataset", "labels_withphrase_test",
        "--dataset_dir", f"{BASE_DIR}/lf_data",
        "--do_train", "True", "--do_eval", "False", "--do_predict", "True", "--predict_with_generate", "True",
        "--learning_rate", "2e-4", "--per_device_train_batch_size", "4", "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "16", "--gradient_checkpointing", "True", "--num_train_epochs", "4", "--bf16", "True",
        "--eval_strategy", "steps", "--eval_steps", str(SAVE_STEPS), "--save_steps", str(SAVE_STEPS), "--save_total_limit", str(SAVE_TOTAL_LIMIT),
        "--save_only_model", "True",
        "--lora_r", "32", "--lora_alpha", "128", "--lora_dropout", "0.05", "--report_to", "none"
    ], f"{LOG_DIR}/{model_name}_finetune_phrase.log")

    # Add labels_withphrase checkpoints to batched jobs
    ckpts_phrase_all = list_checkpoints(f"{BASE_DIR}/outputs/{model_name}_labels_phrase")
    ckpts_phrase = select_checkpoints(ckpts_phrase_all)
    logger.info(f"labels_withphrase checkpoints: total={len(ckpts_phrase_all)}, selected={len(ckpts_phrase)} (strategy={CKPT_EVAL_STRATEGY})")
    for step, ckpt_dir in ckpts_phrase:
        pred_step_lp = f"{BASE_DIR}/outputs/{model_name}_labels_phrase_ckpt{step}_eval.jsonl"
        jobs.append({
            "adapter_dir": ckpt_dir,
            "adapter_name": f"{model_name}_labels_phrase_{step}",
            "dataset": "labels_withphrase_test",
            "max_new_tokens": 512,
            "out_path": pred_step_lp,
        })

    # Execute all jobs in batch: load base model once, switch LoRA adapters
    jobs_file = f"{LOG_DIR}/{model_name}_peft_jobs.json"
    with open(jobs_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    run_command([
        "python3", f"{BASE_DIR}/tools/peft_infer_swap_adapters.py",
        "--model_name_or_path", model_path,
        "--template", template,
        "--dataset_dir", f"{BASE_DIR}/lf_data",
        "--jobs_file", jobs_file,
        "--batch_size", str(PEFT_INFER_BS),
    ], f"{LOG_DIR}/{model_name}_peft_jobs_run.log")
    
    # Evaluate labels_withphrase fine-tuning results
    pred_ft_lp = f"{BASE_DIR}/outputs/{model_name}_labels_phrase/generated_predictions.jsonl"
    metrics_ft_lp = f"{BASE_DIR}/outputs/metrics_{model_name}_labels_withphrase.json"
    run_command([
        "python3", f"{BASE_DIR}/evaluate_multilabel.py",
        "--pred_file", pred_ft_lp,
        "--data_file", f"{BASE_DIR}/lf_data/labels_withphrase_test.jsonl",
        "--out", metrics_ft_lp
    ], f"{LOG_DIR}/{model_name}_eval_finetune_phrase.log")
    record_metrics(model_name, "finetune", "labels_withphrase", "labels_withphrase_test", pred_ft_lp, metrics_ft_lp)

    # Evaluate batched job outputs and record metrics
    for step, ckpt_dir in ckpts:
        pred_step_lo = f"{BASE_DIR}/outputs/{model_name}_labels_only_ckpt{step}_labels_only_eval.jsonl"
        metrics_step_lo = f"{BASE_DIR}/outputs/metrics_{model_name}_labels_only_step{step}.json"
        run_command([
            "python3", f"{BASE_DIR}/evaluate_multilabel.py",
            "--pred_file", pred_step_lo,
            "--data_file", f"{BASE_DIR}/lf_data/labels_only_test.jsonl",
            "--out", metrics_step_lo
        ], f"{LOG_DIR}/{model_name}_labels_only_ckpt{step}_eval.log")
        record_metrics(model_name, "finetune", "labels_only_step", "labels_only_test", pred_step_lo, metrics_step_lo, {"step": step, "checkpoint": ckpt_dir})

        pred_step_tone = f"{BASE_DIR}/outputs/{model_name}_labels_only_ckpt{step}_on_withphrase.jsonl"
        metrics_step_tone = f"{BASE_DIR}/outputs/metrics_{model_name}_labels_only_tone_step{step}.json"
        run_command([
            "python3", f"{BASE_DIR}/evaluate_multilabel.py",
            "--pred_file", pred_step_tone,
            "--data_file", f"{BASE_DIR}/lf_data/labels_withphrase_test.jsonl",
            "--out", metrics_step_tone
        ], f"{LOG_DIR}/{model_name}_labels_only_ckpt{step}_tone_eval.log")
        record_metrics(model_name, "finetune", "tone_step", "labels_withphrase_test", pred_step_tone, metrics_step_tone, {"step": step, "checkpoint": ckpt_dir})

    for step, ckpt_dir in ckpts_phrase:
        pred_step_lp = f"{BASE_DIR}/outputs/{model_name}_labels_phrase_ckpt{step}_eval.jsonl"
        metrics_step_lp = f"{BASE_DIR}/outputs/metrics_{model_name}_labels_withphrase_step{step}.json"
        run_command([
            "python3", f"{BASE_DIR}/evaluate_multilabel.py",
            "--pred_file", pred_step_lp,
            "--data_file", f"{BASE_DIR}/lf_data/labels_withphrase_test.jsonl",
            "--out", metrics_step_lp
        ], f"{LOG_DIR}/{model_name}_labels_phrase_ckpt{step}_eval.log")
        record_metrics(model_name, "finetune", "labels_withphrase_step", "labels_withphrase_test", pred_step_lp, metrics_step_lp, {"step": step, "checkpoint": ckpt_dir})

    # Final adapter evaluation on phrase
    metrics_tone = f"{BASE_DIR}/outputs/metrics_{model_name}_labels_only_tone.json"
    run_command([
        "python3", f"{BASE_DIR}/evaluate_multilabel.py",
        "--pred_file", pred_tone_final,
        "--data_file", f"{BASE_DIR}/lf_data/labels_withphrase_test.jsonl",
        "--out", metrics_tone
    ], f"{LOG_DIR}/{model_name}_eval_tone.log")
    record_metrics(model_name, "finetune", "tone", "labels_withphrase_test", pred_tone_final, metrics_tone)
    
    logger.info(f"{model_name} fine-tuning experiments complete")

def run_loocv_experiments(model_name: str, model_path: str, template: str):
    """LOOCV experiments - batch training for all cities"""
    logger.info(f"Starting {model_name} LOOCV experiments...")
    
    cities = ["beijing", "shenzhen", "shanghai", "guangzhou"]
    
    for city in cities:
        logger.info(f"Running {model_name} {city} LOOCV...")
        
        # Train labels-only
        logger.info(f"Training {model_name} {city} labels-only...")
        run_command([
            "llamafactory-cli", "train",
            "--stage", "sft",
            "--model_name_or_path", model_path,
            "--template", template,
            "--finetuning_type", "lora",
            "--output_dir", f"{BASE_DIR}/outputs/loocv/{city}/{model_name}_labels_only",
            "--dataset", "labels_only_train",
            "--eval_dataset", "labels_only_dev",
            "--dataset_dir", f"{BASE_DIR}/lf_data/loocv/{city}",
            "--do_train", "True", "--do_eval", "True", "--do_predict", "True", "--predict_with_generate", "True",
            "--learning_rate", "2e-4", "--per_device_train_batch_size", "4", "--per_device_eval_batch_size", "4",
            "--gradient_accumulation_steps", "16", "--gradient_checkpointing", "True", "--num_train_epochs", "4", "--bf16", "True",
            "--eval_strategy", "steps", "--eval_steps", str(SAVE_STEPS), "--save_steps", str(SAVE_STEPS), "--save_total_limit", str(SAVE_TOTAL_LIMIT),
        "--save_only_model", "True",
        "--lora_r", "32", "--lora_alpha", "128", "--lora_dropout", "0.05", "--report_to", "none"
        ], f"{LOG_DIR}/{model_name}_loocv_{city}_labels.log")
        
        # Evaluate
        if os.path.exists(f"{BASE_DIR}/outputs/loocv/{city}/{model_name}_labels_only/generated_predictions.jsonl"):
            logger.info("Using predictions generated during training for evaluation")
            pred_city = f"{BASE_DIR}/outputs/loocv/{city}/{model_name}_labels_only/generated_predictions.jsonl"
            metrics_city = f"{BASE_DIR}/outputs/loocv/{city}/metrics_{model_name}_labels_only.json"
            run_command([
                "python3", f"{BASE_DIR}/evaluate_multilabel.py",
                "--pred_file", pred_city,
                "--data_file", f"{BASE_DIR}/lf_data/loocv/{city}/labels_only_test.jsonl",
                "--out", metrics_city
            ], f"{LOG_DIR}/{model_name}_loocv_{city}_labels_eval.log")
            record_metrics(model_name, "loocv", "labels_only", "labels_only_test", pred_city, metrics_city, {"city": city})
        else:
            logger.info("No predictions from training, using LLaMA-Factory predict-only for inference")
            pred_city_dir = f"{BASE_DIR}/outputs/loocv/{city}/{model_name}_labels_only_eval"
            os.makedirs(pred_city_dir, exist_ok=True)
            metrics_city = f"{BASE_DIR}/outputs/loocv/{city}/metrics_{model_name}_labels_only.json"
            run_command([
                "llamafactory-cli", "train",
                "--stage", "sft",
                "--model_name_or_path", model_path,
                "--template", template,
                "--finetuning_type", "lora",
                "--adapter_name_or_path", f"{BASE_DIR}/outputs/loocv/{city}/{model_name}_labels_only",
                "--output_dir", pred_city_dir,
                "--overwrite_output_dir", "True",
                "--dataset", "labels_only_test",
                "--dataset_dir", f"{BASE_DIR}/lf_data/loocv/{city}",
                "--do_train", "False", "--do_eval", "False",
                "--do_predict", "True", "--predict_with_generate", "True",
            "--per_device_eval_batch_size", str(PREDICT_EVAL_BS),
            "--per_device_eval_batch_size", str(PREDICT_EVAL_BS),
            "--per_device_eval_batch_size", str(PREDICT_EVAL_BS),
            "--per_device_eval_batch_size", str(PREDICT_EVAL_BS),
            "--per_device_eval_batch_size", str(PREDICT_EVAL_BS),
            "--per_device_eval_batch_size", str(PREDICT_EVAL_BS),
                "--report_to", "none"
            ], f"{LOG_DIR}/{model_name}_loocv_{city}_labels_infer.log")
            pred_city = f"{pred_city_dir}/generated_predictions.jsonl"
            
            run_command([
                "python3", f"{BASE_DIR}/evaluate_multilabel.py",
                "--pred_file", pred_city,
                "--data_file", f"{BASE_DIR}/lf_data/loocv/{city}/labels_only_test.jsonl",
                "--out", metrics_city
            ], f"{LOG_DIR}/{model_name}_loocv_{city}_labels_eval.log")
            record_metrics(model_name, "loocv", "labels_only", "labels_only_test", pred_city, metrics_city, {"city": city})
        
        logger.info(f"{model_name} {city} LOOCV complete")
    
    logger.info(f"{model_name} LOOCV experiments complete")

def run_baseline_experiments():
    """Baseline experiments"""
    logger.info("开始Baseline experiments...")
    
    # 全量训练/测试
    logger.info("运行全量Baseline experiments...")
    run_command([
        "python3", f"{BASE_DIR}/baseline_multilabel_roberta.py",
        "--splits_dir", f"{BASE_DIR}/splits",
        "--output_dir", f"{BASE_DIR}/baseline_outputs",
        "--model", "hfl/chinese-roberta-wwm-ext-large",
        "--epochs", "5", "--batch_size", "16", "--lr", "2e-5"
    ], f"{LOG_DIR}/baseline_full.log")
    
    # LOOCV 基线
    logger.info("Running baseline LOOCV experiments...")
    for city in ["beijing", "shenzhen", "shanghai", "guangzhou"]:
        logger.info(f"Running baseline {city} LOOCV...")
        run_command([
            "python3", f"{BASE_DIR}/baseline_multilabel_roberta.py",
            "--splits_dir", f"{BASE_DIR}/splits",
            "--output_dir", f"{BASE_DIR}/baseline_outputs",
            "--model", "hfl/chinese-roberta-wwm-ext-large",
            "--epochs", "5", "--batch_size", "16", "--lr", "2e-5",
            "--heldout_city", city
        ], f"{LOG_DIR}/baseline_loocv_{city}.log")
    
    logger.info("Baseline experiments完成")

def summarize_results():
    """Results summary"""
    logger.info("开始Results summary...")
    
    run_command([
        "python3", f"{BASE_DIR}/summarize_results.py",
        "--results_dir", f"{BASE_DIR}/outputs",
        "--baseline_dir", f"{BASE_DIR}/baseline_outputs",
        "--output_file", f"{RESULTS_DIR}/experiment_summary.json"
    ], f"{LOG_DIR}/summary.log")
    
    logger.info(f"Results summary完成，请查看: {RESULTS_DIR}/experiment_summary.json")

def main():
    """Main function"""
    logger.info("Starting optimized full experiment workflow...")
    logger.info(f"Experiment start time: {datetime.now()}")
    logger.info("Optimization: each model loaded only 2 times (zero-shot once, finetune+LOOCV once)")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Data preparation
    prepare_data()
    
    # Run order: all models zero-shot, then all models fine-tune, then all models LOOCV
    model_order = ["qwen", "yi", "gemma"]

    # 1) All models zero-shot
    for model_key in model_order:
        model_path = MODELS[model_key]
        template = TEMPLATES[model_key]
        logger.info(f"Starting {model_key} zero-shot...")
        os.environ["MODEL"] = model_path
        os.environ["TEMPLATE"] = template
        run_zeroshot_experiments(model_key, model_path, template)
        logger.info(f"{model_key} zero-shot complete")

    # 2) All models fine-tuning
    for model_key in model_order:
        model_path = MODELS[model_key]
        template = TEMPLATES[model_key]
        logger.info(f"Starting {model_key} fine-tuning...")
        os.environ["MODEL"] = model_path
        os.environ["TEMPLATE"] = template
        run_finetuning_experiments(model_key, model_path, template)
        logger.info(f"{model_key} fine-tuning complete")

    # 3) All models LOOCV
    for model_key in model_order:
        model_path = MODELS[model_key]
        template = TEMPLATES[model_key]
        logger.info(f"Starting {model_key} LOOCV...")
        os.environ["MODEL"] = model_path
        os.environ["TEMPLATE"] = template
        run_loocv_experiments(model_key, model_path, template)
        logger.info(f"{model_key} LOOCV complete")
    
    # Baseline experiments
    run_baseline_experiments()
    
    # Results summary
    summarize_results()
    
    logger.info("All experiments complete!")
    logger.info(f"Experiment end time: {datetime.now()}")
    logger.info(f"See results directory: {RESULTS_DIR}")
    logger.info("Optimization: each model loaded only 2 times (zero-shot once, fine-tune+LOOCV once)")

if __name__ == "__main__":
    main()
