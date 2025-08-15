#!/bin/bash
# Script to run EuroBERT-610m evaluation with memory optimizations

# Parse command line arguments
MAX_STEPS=""
DATA_PERCENTAGE=""
PER_DEVICE_TRAIN_BATCH_SIZE="1"
PER_DEVICE_EVAL_BATCH_SIZE="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --data-percentage)
            DATA_PERCENTAGE="$2"
            shift 2
            ;;
        --per-device-train-batch-size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --per-device-eval-batch-size)
            PER_DEVICE_EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set up environment
export LOCAL_DATASET_DIR=/lustre/fswork/projects/rech/vrl/uok92vw/data
export MODEL_PATH=/lustre/fswork/projects/rech/vrl/uok92vw/huggingface_models/EuroBERT-610m
export ENCODEVAL_DIR=/lustre/fswork/projects/rech/vrl/uok92vw/EncodEval

# Add EncodEval to PYTHONPATH
export PYTHONPATH="${ENCODEVAL_DIR}:${PYTHONPATH}"

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "========================================="
echo "Starting EuroBERT-610m Evaluation (Memory Optimized)"
echo "Model: $MODEL_PATH"
echo "Train Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Eval Batch Size: $PER_DEVICE_EVAL_BATCH_SIZE"
if [ -n "$MAX_STEPS" ]; then
    echo "Max Steps Override: $MAX_STEPS"
fi
echo "========================================="

# Function to run evaluation with memory optimizations
run_evaluation() {
    local task_name=$1
    local config_file=$2
    
    echo ""
    echo "========================================="
    echo "Running: $task_name"
    echo "Config: $config_file"
    echo "========================================="
    
    # Clear GPU cache before each task
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # Create temporary config with batch size overrides
    CONFIG_PATH="${ENCODEVAL_DIR}/configs/${config_file}"
    TEMP_CONFIG="/tmp/eurobert_config_$$.yaml"
    
    # Copy and modify config
    cp "$CONFIG_PATH" "$TEMP_CONFIG"
    
    # Override batch sizes
    sed -i "s/per_device_train_batch_size: [0-9]*/per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE/g" "$TEMP_CONFIG"
    sed -i "s/per_device_eval_batch_size: [0-9]*/per_device_eval_batch_size: $PER_DEVICE_EVAL_BATCH_SIZE/g" "$TEMP_CONFIG"
    
    # Add gradient accumulation if needed
    if ! grep -q "gradient_accumulation_steps:" "$TEMP_CONFIG"; then
        sed -i "/per_device_train_batch_size:/a\\    gradient_accumulation_steps: 32" "$TEMP_CONFIG"
    fi
    
    # Override max_steps if provided
    if [ -n "$MAX_STEPS" ]; then
        sed -i "s/max_steps: [0-9]*/max_steps: $MAX_STEPS/g" "$TEMP_CONFIG"
        if ! grep -q "max_steps:" "$TEMP_CONFIG"; then
            sed -i "/tr_args_kwargs:/a\\    max_steps: $MAX_STEPS" "$TEMP_CONFIG"
        fi
    fi
    
    # Build command
    CMD="python ${ENCODEVAL_DIR}/main.py --config_file $TEMP_CONFIG --model_path ${MODEL_PATH}"
    
    if [ -n "$DATA_PERCENTAGE" ]; then
        CMD="$CMD --data_percentage $DATA_PERCENTAGE"
    fi
    
    echo "Executing: $CMD"
    
    # Run with error handling
    eval $CMD
    EXIT_CODE=$?
    
    # Clean up
    [ -f "$TEMP_CONFIG" ] && rm "$TEMP_CONFIG"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $task_name completed successfully"
    else
        echo "✗ $task_name failed with error code $EXIT_CODE"
        # Clear GPU cache on failure
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
}

# Run evaluations with memory management between tasks

# Retrieval Tasks
echo ""
echo "=== RETRIEVAL TASKS ==="
run_evaluation "MS MARCO Training" \
    "retrieval_tasks/msmarco_train.yaml"

run_evaluation "MIRACL Evaluation" \
    "retrieval_tasks/miracl_eval.yaml"

# Clear cache between major task types
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Sequence Classification Tasks
echo ""
echo "=== SEQUENCE CLASSIFICATION TASKS ==="
run_evaluation "XNLI" \
    "sequence_classification_tasks/xnli.yaml"

run_evaluation "Amazon MASSIVE Intent" \
    "sequence_classification_tasks/amazon_massive_intent.yaml"

# Clear cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Sequence Regression Tasks
echo ""
echo "=== SEQUENCE REGRESSION TASKS ==="
run_evaluation "Seahorse" \
    "sequence_regression_tasks/seahorse.yaml"

# Token Classification Tasks
echo ""
echo "=== TOKEN CLASSIFICATION TASKS ==="
run_evaluation "NER" \
    "token_classification_tasks/ner.yaml"

echo ""
echo "========================================="
echo "EuroBERT-610m evaluation completed!"
echo "Results saved in: ${ENCODEVAL_DIR}/results/"
echo "========================================="
