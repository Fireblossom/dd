#!/bin/bash
# ICWSM 2025 Supplementary Experiments - Run All Models
#
# Usage:
#   bash src/run_all_models.sh           # Test set only
#   bash src/run_all_models.sh --all     # Full dataset
#   bash src/run_all_models.sh -e 1      # Run only experiment 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
SPLIT="test"
EXPERIMENT=""
for arg in "$@"; do
    case $arg in
        --all)
            SPLIT="all"
            ;;
        -e)
            shift
            EXPERIMENT="$1"
            ;;
        [1-4])
            EXPERIMENT="$arg"
            ;;
    esac
done

# Model configurations
declare -A MODELS
MODELS[qwen]="Qwen/Qwen3-32B:qwen:outputs/checkpoint/qwen_labels_only"
MODELS[yi]="01-ai/Yi-1.5-34B-Chat:yi:outputs/checkpoint/yi_labels_only"
MODELS[gemma]="google/gemma-3-27b-it:gemma:outputs/checkpoint/gemma_labels_only"

echo "=============================================="
echo "ICWSM 2025 Experiments - Batch Run"
echo "=============================================="
echo "Data scope: $SPLIT"
echo "Experiment: ${EXPERIMENT:-all}"
echo ""

for model_name in qwen yi gemma; do
    IFS=':' read -r model_path template adapter <<< "${MODELS[$model_name]}"
    
    echo ""
    echo "=============================================="
    echo "Running model: $model_name"
    echo "  Path: $model_path"
    echo "  Template: $template"
    echo "  Adapter: $adapter"
    echo "=============================================="
    
    OUTPUT_DIR="outputs/icwsm_${model_name}"
    
    CMD="python3 $SCRIPT_DIR/run_icwsm_experiments.py \
        --model $model_path \
        --template $template \
        --adapter $PROJECT_ROOT/$adapter \
        --output_dir $OUTPUT_DIR \
        --split $SPLIT"
    
    if [ -n "$EXPERIMENT" ]; then
        CMD="$CMD -e $EXPERIMENT"
    fi
    
    echo "Executing: $CMD"
    eval $CMD
    
    echo ""
    echo "✓ $model_name complete! Results: $OUTPUT_DIR/"
done

echo ""
echo "=============================================="
echo "All models complete!"
echo "Results directories:"
echo "  - outputs/icwsm_qwen/"
echo "  - outputs/icwsm_yi/"
echo "  - outputs/icwsm_gemma/"
echo "=============================================="
