## Experimental Steps and Commands

This file documents the reproducible commands for data preparation, prompts, main experiments, and strong baselines.

### 0. Environment Assumptions
- LLaMA-Factory installed with `llamafactory-cli` executable
- Recommended environment variable: `export MODEL=Qwen/Qwen3-32B` (or equivalent local model path)
- Python 3.10+, Transformers / PyTorch installed (for baseline)

Alternative LLMs (directly replace the $MODEL above):
```bash
# Option 1: Yi 34B Chat
export MODEL=01-ai/Yi-1.5-34B-Chat

# Option 2: Gemma 2 27B it
export MODEL=google/gemma-3-27b-it

# Note: Choose appropriate template (TEMPLATE):
# Qwen/Qwen3 uses qwen; Yi uses yi; Gemma uses gemma
export TEMPLATE=qwen   # If using Qwen/Qwen3
# export TEMPLATE=yi    # If using Yi-1.5-34B-Chat
# export TEMPLATE=gemma # If using gemma-3-27b-it
```

Multi-GPU (4×A100-80G) recommended environment setup:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 1. Data Preparation

1) Generate city-stratified 8:1:1 splits and LLaMA-Factory JSONL:
```bash
python3 /nfs/home/tanz/dd2/annotated/make_splits_and_jsonl.py \
  --csv /nfs/home/tanz/dd2/annotated/top20-1.csv \
  --outdir /nfs/home/tanz/dd2/annotated/splits \
  --lf_outdir /nfs/home/tanz/dd2/annotated/lf_data
```

Data outputs:
- 8:1:1 splits: `/nfs/home/tanz/dd2/annotated/splits/all_{train,dev,test}.csv`
- JSONL format: `/nfs/home/tanz/dd2/annotated/lf_data/labels_only_*.jsonl` and `labels_withphrase_*.jsonl`

### 2. Prompts and Formats
- Prompts located at: `/nfs/home/tanz/dd2/annotated/prompts.md`
- Two types:
  - labels-only (strict output of label names or "未提及任何标准")
  - CoT+phrase evidence (JSON output: evidence_phrases/labels/tone)

### 3. Main Experiments

#### 3.1 Zero-shot
Direct inference on test set using labels-only instructions:
```bash
python /nfs/home/tanz/dd2/LLaMA-Factory/scripts/vllm_infer.py \
  --model_name_or_path $MODEL \
  --template $TEMPLATE \
  --dataset labels_only_test \
  --dataset_dir /nfs/home/tanz/dd2/annotated/lf_data \
  --max_new_tokens 256 \
  --batch_size 4096 \
  --pipeline_parallel_size 1 \
  --vllm_config '{"gpu_memory_utilization": 0.92}' \
  --save_name /nfs/home/tanz/dd2/annotated/outputs/vllm_zeroshot_labels_only.jsonl

# Evaluation (including tone metrics)
python3 /nfs/home/tanz/dd2/annotated/evaluate_multilabel.py \
  --pred_file /nfs/home/tanz/dd2/annotated/outputs/vllm_zeroshot_labels_only.jsonl \
  --data_file /nfs/home/tanz/dd2/annotated/lf_data/labels_withphrase_test.jsonl \
  --out /nfs/home/tanz/dd2/annotated/outputs/metrics_zeroshot_labels_only.json
```

Direct inference on test set using labels_withphrase instructions (with tone):
```bash
python /nfs/home/tanz/dd2/LLaMA-Factory/scripts/vllm_infer.py \
  --model_name_or_path $MODEL \
  --template $TEMPLATE \
  --dataset labels_withphrase_test \
  --dataset_dir /nfs/home/tanz/dd2/annotated/lf_data \
  --max_new_tokens 512 \
  --batch_size 4096 \
  --pipeline_parallel_size 1 \
  --vllm_config '{"gpu_memory_utilization": 0.92}' \
  --save_name /nfs/home/tanz/dd2/annotated/outputs/vllm_zeroshot_labels_withphrase.jsonl

# Evaluation (including tone metrics)
python3 /nfs/home/tanz/dd2/annotated/evaluate_multilabel.py \
  --pred_file /nfs/home/tanz/dd2/annotated/outputs/vllm_zeroshot_labels_withphrase.jsonl \
  --data_file /nfs/home/tanz/dd2/annotated/lf_data/labels_withphrase_test.jsonl \
  --out /nfs/home/tanz/dd2/annotated/outputs/metrics_zeroshot_labels_withphrase.json
```

#### 3.2 Fine-tuning (labels-only)
Integrated: generate predictions on test set directly after training (no vLLM needed):
```bash
llamafactory-cli train \
  --stage sft \
  --model_name_or_path $MODEL \
  --template $TEMPLATE \
  --finetuning_type lora \
  --output_dir /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_only \
  --overwrite_output_dir True \
  --dataset labels_only_train \
  --eval_dataset labels_only_test \
  --dataset_dir /nfs/home/tanz/dd2/annotated/lf_data \
  --do_train True --do_eval False --do_predict True --predict_with_generate True \
  --learning_rate 2e-4 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 --gradient_checkpointing True --num_train_epochs 30 --bf16 True \
  --eval_strategy steps --eval_steps 200 --save_steps 200 --save_total_limit 2 \
  --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --report_to none

# Prediction file path: /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_only/generated_predictions.jsonl
python3 /nfs/home/tanz/dd2/annotated/evaluate_multilabel.py \
  --pred_file /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_only/generated_predictions.jsonl \
  --data_file /nfs/home/tanz/dd2/annotated/lf_data/labels_withphrase_test.jsonl \
  --out /nfs/home/tanz/dd2/annotated/outputs/metrics_labels_only.json
```

Optional (use same adapter for tone output): inference and evaluation of tone on labels_withphrase test set:
```bash
python /nfs/home/tanz/dd2/LLaMA-Factory/scripts/vllm_infer.py \
  --model_name_or_path $MODEL \
  --adapter_name_or_path /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_only \
  --template $TEMPLATE \
  --dataset labels_withphrase_test \
  --dataset_dir /nfs/home/tanz/dd2/annotated/lf_data \
  --max_new_tokens 512 \
  --batch_size 4096 \
  --pipeline_parallel_size 1 \
  --vllm_config '{"gpu_memory_utilization": 0.92}' \
  --save_name /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_only_on_withphrase.jsonl

python3 /nfs/home/tanz/dd2/annotated/evaluate_multilabel.py \
  --pred_file /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_only_on_withphrase.jsonl \
  --data_file /nfs/home/tanz/dd2/annotated/lf_data/labels_withphrase_test.jsonl \
  --out /nfs/home/tanz/dd2/annotated/outputs/metrics_labels_only_tone.json
```

#### 3.3 Fine-tuning (labels+phrase, CoT-style)
Integrated: generate predictions on test set directly after training (no vLLM needed):
```bash
llamafactory-cli train \
  --stage sft \
  --model_name_or_path $MODEL \
  --template $TEMPLATE \
  --finetuning_type lora \
  --output_dir /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_phrase \
  --overwrite_output_dir True \
  --dataset labels_withphrase_train \
  --eval_dataset labels_withphrase_test \
  --dataset_dir /nfs/home/tanz/dd2/annotated/lf_data \
  --do_train True --do_eval False --do_predict True --predict_with_generate True \
  --learning_rate 2e-4 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 --gradient_checkpointing True --num_train_epochs 30 --bf16 True \
  --eval_strategy steps --eval_steps 200 --save_steps 200 --save_total_limit 2 \
  --lora_r 32 --lora_alpha 128 --lora_dropout 0.05 --report_to none

# Prediction file path: /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_phrase/generated_predictions.jsonl
python3 /nfs/home/tanz/dd2/annotated/evaluate_multilabel.py \
  --pred_file /nfs/home/tanz/dd2/annotated/outputs/qwen32b_labels_phrase/generated_predictions.jsonl \
  --data_file /nfs/home/tanz/dd2/annotated/lf_data/labels_withphrase_test.jsonl \
  --out /nfs/home/tanz/dd2/annotated/outputs/metrics_labels_withphrase.json
```

### 4. Strong Text Classification Baseline (Multi-label RoBERTa)

```bash
python3 /nfs/home/tanz/dd2/annotated/baseline_multilabel_roberta.py \
  --splits_dir /nfs/home/tanz/dd2/annotated/splits \
  --output_dir /nfs/home/tanz/dd2/annotated/baseline_outputs \
  --model hfl/chinese-roberta-wwm-ext-large \
  --epochs 5 --batch_size 16 --lr 2e-5
```

### 5. Reporting and Paper Writing Recommendations
- Metrics: micro/macro F1, Jaccard, Hamming; separate P/R/F1 for "未提及任何标准"
- Stratification: Report by city with mean±variance; compare objective vs. socio-cultural-psychological criteria groups
- Ablation: 0-shot vs. label-SFT vs. phrase-SFT; RoBERTa baseline; (optional) threshold/temperature and calibration
- Reliability: Two-annotator IAA (Cohen's κ/α), report human-human agreement


