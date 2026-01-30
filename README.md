# Modeling Belonging

This repository contains the data, code, and experimental artifacts for our submission on operationalizing local identity in urban Chinese discourse.

Start with `REPRODUCIBILITY.md` for a quick run + artifact map.

## Repository Structure

This repository is organized to facilitate reproducibility and corresponds directly to sections of the paper:

### `dataset/splits/`
**→ Paper Section 3.1 (Corpus Collection and Scope)**

Contains city-stratified train/dev/test splits following an 8:1:1 ratio:
- **Included samples**: each CSV contains **only the header and 5 sample rows**
- Split organization:
    - Per-city CSV files: `{city}_{train,dev,test}.csv`
    - Aggregated files: `all_{train,dev,test}.csv`
    - Summary statistics: `summary.json`

**Note**: The full dataset will be released upon paper acceptance under a Restricted Academic Use Agreement.

Corresponds to **Table 1** in the paper (city-level corpus composition).

---

### `src/`
**→ Paper Section 4 (Experimental Setup) and Methods**

Core scripts for data processing, training, and evaluation:

| File | Description | Paper Reference |
|------|-------------|----------------|
| `annotation_manual.txt` | Complete annotation guidelines (Chinese) | Appendix §A (Annotation Guidelines) |
| `annotation_manual_en.txt` | Complete annotation guidelines (English) | Appendix §A (Annotation Guidelines) |
| `make_splits_and_jsonl.py` | Data preparation and format conversion | Section 3.1, 4.3 |
| `baseline_multilabel_roberta.py` | Chinese RoBERTa-Large baseline | Section 4.2 (Models and Baselines) |
| `run_finetune.py` | LoRA fine-tuning wrapper script | Section 4.4 (Training) |
| `eval_llamafactory_checkpoints.py` | Checkpoint-level evaluation | Appendix (Learning Curves) |
| `evaluate_multilabel.py` | Metrics computation (F1, Jaccard, etc.) | Section 4.5 (Evaluation Metrics) |
| `collect_llamafactory_preds.py` | Prediction aggregation utility | - |
| **`EXPERIMENTS.md`** | **Detailed reproduction commands** | Full workflow documentation |
| `RESEARCH_NOTES.md` | Development notes (Chinese) | Internal documentation |

---

### `outputs/`
**→ Paper Section 5 (Results)**

This directory contains **lightweight reports and figures** (PNG/PDF/MD/CSV).

Large artifacts (e.g., checkpoint-level prediction dumps) are intentionally excluded.

```
outputs/
├── city_profiles/             # City profile report + figures
├── cross_platform/            # Cross-platform report
├── icwsm/                     # External validity + sensitivity reports
└── results/                   # Aggregated metrics/tables
```

---

### `plots/`
**→ Appendix Figure (Learning Curves)**

Visualization of model performance over training:
- `annotated_curves_grid.png`: Learning curves for all models and tasks (Appendix Figure 1)
- `plot_annotated_curves.py`: Script to generate the visualization
- CSV files with checkpoint-level metrics for Qwen-3, Yi-1.5, and Gemma-3

---

## Tasks

This work introduces two classification tasks grounded in sociolinguistic theory:

### Task 1: Localness Criteria Classification (Multi-label)
**→ Paper Section 3.2 (Annotation Framework)**

Classify which criteria are invoked to define "who counts as a local" from **11 categories**:

**Objective Criteria:**
1. Current Residence (个人在当地的当前居住状态)
2. Birth and Upbringing (个人在当地的出生和成长经历)
3. Administrative or Legal Recognition (个人的法律与行政认定)
4. Family Heritage (家族在当地的历史传承)
5. Assets and Rights (个人或家族在当地的资产与权益)
6. Intra-city Boundary (城市内部地理片区归属)

**Socio-cultural and Psychological Criteria:**
7. Self-Identification (个人认定)
8. Sense of Belonging (个人的归属感)
9. Language Ability (个人的语言能力)
10. Cultural Practice and Knowledge (个人的文化实践与知识)
11. Community Acceptance (个人被社群接纳)

**Special label:** "No Mentioned Criteria" (未提及任何标准)

See **Table 2** (full taxonomy with definitions and examples).

---

### Task 2: Openness Tendency Classification (Single-label)
**→ Paper Section 3.2 (Annotation Framework)**

Classify the stance toward local identity boundaries:
- **Inclusive (宽容开放)**: Broad/permissive criteria (e.g., "as long as...")
- **Neutral (中性)**: Descriptive, no clear stance
- **Restrictive (紧缩排斥)**: Tight/exclusive criteria (e.g., "only if...", "must...")

---

### Experimental Regimes

Two conditions are evaluated:
1. **labels-only**: Models predict labels directly
2. **labels-with-phrase**: Models extract verbatim evidence phrases before predicting labels (chain-of-thought style)

---

## Quick Start

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for LLM fine-tuning
- Transformers, scikit-learn, pandas

### Reproduction Workflow

**1. Data Preparation**
```bash
# Generate stratified splits and JSONL files for training
python src/make_splits_and_jsonl.py \
  --csv <input_annotated_data.csv> \
  --outdir dataset/splits \
  --lf_outdir dataset/lf_data
```
See `src/EXPERIMENTS.md` for detailed commands.

---

**2. Run Baseline (RoBERTa)**
```bash
python src/baseline_multilabel_roberta.py \
  --splits_dir dataset/splits \
  --output_dir outputs/baseline_outputs \
  --model hfl/chinese-roberta-wwm-ext-large \
  --epochs 5 --batch_size 16 --lr 2e-5
```
See `src/EXPERIMENTS.md` for full commands.

---

**3. Fine-tune LLMs (LoRA)**
```bash
# Example: Qwen-3-32B with labels-only regime
llamafactory-cli train \
  --stage sft \
  --model_name_or_path Qwen/Qwen3-32B \
  --template qwen \
  --finetuning_type lora \
  --dataset labels_only_train \
  --eval_dataset labels_only_test \
  --dataset_dir dataset/lf_data \
  --output_dir outputs/qwen_labels_only \
  --do_train True --do_predict True --predict_with_generate True \
  --learning_rate 2e-4 --num_train_epochs 30 \
  --lora_r 32 --lora_alpha 128 --lora_dropout 0.05
```
See `src/EXPERIMENTS.md` for all model configurations.

---

**4. Evaluate Predictions**
```bash
python src/evaluate_multilabel.py \
  --pred_file outputs/qwen_labels_only/generated_predictions.jsonl \
  --data_file dataset/lf_data/labels_withphrase_test.jsonl \
  --out outputs/results/qwen_labels_only_metrics.json
```
See `src/EXPERIMENTS.md` for metrics details.

## Key Results

### Main Findings (Section 5)

**Localness Criteria Task (Multi-label)**
| Model Configuration | Micro F1 | Macro F1 | Jaccard |
|---------------------|----------|----------|---------|
| RoBERTa-Large (baseline) | 0.487 | 0.466 | 0.194 |
| **Gemma-3 (labels-with-phrase, fine-tuned)** | **0.834** | **0.706** | **0.745** |
| Yi-1.5 (labels-with-phrase, fine-tuned) | 0.832 | 0.665 | 0.738 |
| Qwen-3 (labels-only, fine-tuned) | 0.811 | 0.668 | 0.712 |

**Openness Tendency Task (Single-label)**
| Model Configuration | Accuracy | Macro F1 |
|---------------------|----------|----------|
| RoBERTa-Large (baseline) | 0.669 | 0.654 |
| **Gemma-3 (labels-with-phrase, fine-tuned)** | 0.726 | **0.736** |
| Yi-1.5 (labels-with-phrase, fine-tuned) | **0.730** | 0.726 |
| Qwen-3 (labels-only, fine-tuned) | 0.729 | 0.711 |

**Key Takeaways:**
- Fine-tuning yields **substantial performance gains** over zero-shot baselines
- All LLMs reach comparable performance after fine-tuning (Micro F1: 0.810-0.834)
- The `labels-with-phrase` regime shows mixed effects depending on model and task
- Qualitative analysis reveals fine-tuning enables deeper semantic understanding beyond format compliance

Full results in Tables 2-3 and Appendix (per-class metrics, learning curves).

---

## Data Access and Ethics

### Data Statement
This dataset is provided with full documentation:

- **Licensing**: Restricted Academic Use
- **Ethical Safeguards**: PII removed, usernames/timestamps anonymized, no private messages included
- **Intended Use**: Sociolinguistic and NLP research on identity, stance, and inclusion/exclusion

### Ethical Considerations
This research is descriptive in nature and aims to understand how social boundaries are linguistically constructed. The annotation framework and models should **not** be used for:
- Determining actual eligibility or rights
- Automated moderation without human oversight
- Profiling individuals or communities

See paper Section: Ethical Considerations for detailed discussion of potential misuse and mitigation strategies.