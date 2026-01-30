#!/bin/bash
# Run all supplementary experiments with all 3 fine-tuned models
# Usage: bash scripts/run_experiments.sh

set -e

cd "$(dirname "$0")/.."

echo "=============================================="
echo "Supplementary Experiments - Full Run"
echo "=============================================="

# Model configurations
declare -A MODELS=(
    ["qwen"]="Qwen/Qwen3-32B:qwen:outputs/checkpoint/qwen_labels_only"
    ["yi"]="01-ai/Yi-1.5-34B-Chat:yi:outputs/checkpoint/yi_labels_only"
    ["gemma"]="google/gemma-3-27b-it:gemma:outputs/checkpoint/gemma_labels_only"
)

for model_name in qwen yi gemma; do
    IFS=':' read -r model_path template adapter <<< "${MODELS[$model_name]}"
    
    echo ""
    echo "=============================================="
    echo "Model: $model_name"
    echo "=============================================="
    
    python3 src/run_icwsm_experiments.py \
        --model "$model_path" \
        --template "$template" \
        --adapter "$adapter" \
        --output_dir "outputs/experiments_${model_name}" \
        --split test
    
    echo "✓ $model_name complete!"
done

echo ""
echo "=============================================="
echo "All experiments complete!"
echo ""
echo "Results:"
echo "  outputs/experiments_qwen/"
echo "  outputs/experiments_yi/"
echo "  outputs/experiments_gemma/"
echo "=============================================="
