#!/bin/bash
# Script to run EuroBERT evaluation on selected tasks
# Following the paper's protocol with English-only datasets

# Parse command line arguments
MAX_STEPS=""
DATA_PERCENTAGE=""
PER_DEVICE_TRAIN_BATCH_SIZE=""
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
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--max-steps NUMBER] [--data-percentage NUMBER] [--per-device-train-batch-size NUMBER]"
            echo "Example: $0 --max-steps 100 --data-percentage 10 --per-device-train-batch-size 16"
            exit 1
            ;;
    esac
done

# Set up environment
export LOCAL_DATASET_DIR=/lustre/fswork/projects/rech/vrl/uok92vw/data
export MODEL_PATH=/lustre/fswork/projects/rech/vrl/uok92vw/huggingface_models/EuroBERT-610m
export ENCODEVAL_DIR=/lustre/fswork/projects/rech/vrl/uok92vw/EncodEval

# Load modules and activate environment
module purge
module load arch/h100
module load miniforge/24.9.0
conda activate encoder

echo "========================================="
echo "Starting EuroBERT Evaluation"
echo "Model: $MODEL_PATH"
echo "Dataset Directory: $LOCAL_DATASET_DIR"
if [ -n "$MAX_STEPS" ]; then
    echo "Max Steps Override: $MAX_STEPS"
fi
if [ -n "$DATA_PERCENTAGE" ]; then
    echo "Data Percentage: $DATA_PERCENTAGE%"
fi
if [ -n "$PER_DEVICE_TRAIN_BATCH_SIZE" ]; then
    echo "Per Device Train Batch Size Override: $PER_DEVICE_TRAIN_BATCH_SIZE"
fi
echo "========================================="

# Function to run evaluation
run_evaluation() {
    local task_name=$1
    local config_file=$2
    
    echo ""
    echo "========================================="
    echo "Running: $task_name"
    echo "Config: $config_file"
    echo "========================================="
    
    # Check if we need to create a temporary config file
    if [ -n "$MAX_STEPS" ] || [ -n "$PER_DEVICE_TRAIN_BATCH_SIZE" ]; then
        # Create temporary config file with overrides
        TEMP_CONFIG="/tmp/temp_config_$.yaml"
        cp "${ENCODEVAL_DIR}/configs/${config_file}" "$TEMP_CONFIG"
        
        # Replace max_steps value if provided
        if [ -n "$MAX_STEPS" ]; then
            sed -i "s/max_steps: [0-9]*/max_steps: $MAX_STEPS/" "$TEMP_CONFIG"
        fi
        
        # Replace per_device_train_batch_size value if provided
        if [ -n "$PER_DEVICE_TRAIN_BATCH_SIZE" ]; then
            sed -i "s/per_device_train_batch_size: [0-9]*/per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE/" "$TEMP_CONFIG"
        fi
        
        # Use the temporary config
        CMD="python ${ENCODEVAL_DIR}/main.py --config_file $TEMP_CONFIG --model_path ${MODEL_PATH}"
    else
        # Use original config
        CMD="python $ENCODEVAL_DIR/main.py --model_path $MODEL_PATH --config_file $ENCODEVAL_DIR/configs/$config_file"
    fi
    
    # Add data_percentage if provided
    if [ -n "$DATA_PERCENTAGE" ]; then
        CMD="$CMD --data_percentage $DATA_PERCENTAGE"
    fi
    
    # Run evaluation
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $task_name"
    else
        echo "✗ Failed: $task_name"
    fi
    
    # Clean up temporary config if created
    if [ -f "$TEMP_CONFIG" ]; then
        rm -f "$TEMP_CONFIG"
    fi
}

# Retrieval/IR Tasks
echo ""
echo "=== RETRIEVAL TASKS ==="

echo "--- Step 1: Training on MS MARCO ---"
run_evaluation "MS MARCO Training (Required for retrieval evaluation)" \
	    "retrieval_tasks/msmarco_train.yaml"

echo ""
echo "--- Step 2: Evaluating on MIRACL using fine-tuned model ---"
run_evaluation "MIRACL (Multilingual Information Retrieval)" \
	    "retrieval_tasks/miracl_eval.yaml"

echo ""

# Sequence Classification Tasks
echo ""
echo "=== SEQUENCE CLASSIFICATION TASKS ==="

run_evaluation "XNLI (Cross-lingual Natural Language Inference)" \
    "sequence_classification_tasks/xnli.yaml"

run_evaluation "Amazon MASSIVE Intent Classification" \
    "sequence_classification_tasks/amazon_massive_intent.yaml"

# Sequence Regression Tasks
echo ""
echo "=== SEQUENCE REGRESSION TASKS ==="

run_evaluation "Seahorse (Sequence Quality Evaluation)" \
    "sequence_regression_tasks/seahorse.yaml"

# Token Classification Tasks
echo ""
echo "=== TOKEN CLASSIFICATION TASKS ==="

run_evaluation "NER (Named Entity Recognition)" \
    "token_classification_tasks/ner.yaml"


echo ""
echo "========================================="
echo "EuroBERT Evaluation Complete"
echo "========================================="
