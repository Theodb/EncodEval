#!/bin/bash
# Script to run Qwen3-0.6B-Base evaluation on selected tasks
# Using custom modeling_qwen.py for sequence and token classification

# Parse command line arguments
MAX_STEPS=""
DATA_PERCENTAGE=""
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
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--max-steps NUMBER] [--data-percentage NUMBER]"
            echo "Example: $0 --max-steps 100 --data-percentage 10"
            exit 1
            ;;
    esac
done

# Set up environment
export LOCAL_DATASET_DIR=/lustre/fswork/projects/rech/vrl/uok92vw/data
export MODEL_PATH=/lustre/fswork/projects/rech/vrl/uok92vw/huggingface_models/Qwen3-0.6B
export ENCODEVAL_DIR=/lustre/fswork/projects/rech/vrl/uok92vw/EncodEval

# Load modules and activate environment
module purge
module load arch/h100
module load miniforge/24.9.0
conda activate encoder

# Add EncodEval to PYTHONPATH
export PYTHONPATH="${ENCODEVAL_DIR}:${PYTHONPATH}"

echo "========================================="
echo "Starting Qwen3-0.6B-Base Evaluation"
echo "Model: $MODEL_PATH"
echo "Dataset Directory: $LOCAL_DATASET_DIR"
if [ -n "$MAX_STEPS" ]; then
    echo "Max Steps Override: $MAX_STEPS"
fi
if [ -n "$DATA_PERCENTAGE" ]; then
    echo "Data Percentage: $DATA_PERCENTAGE%"
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
    
    # Build command
    CMD="python ${ENCODEVAL_DIR}/main.py --config_file ${ENCODEVAL_DIR}/configs/${config_file} --model_path ${MODEL_PATH}"
    
    # Add data percentage if provided
    if [ -n "$DATA_PERCENTAGE" ]; then
        CMD="$CMD --data_percentage $DATA_PERCENTAGE"
    fi
    
    # Add max_steps override if provided
    if [ -n "$MAX_STEPS" ]; then
        # Create temporary config file with max_steps override
        TEMP_CONFIG="/tmp/temp_config_$$.yaml"
        cp "${ENCODEVAL_DIR}/configs/${config_file}" "$TEMP_CONFIG"
        
        # Replace max_steps value in the temporary config
        sed -i "s/max_steps: [0-9]*/max_steps: $MAX_STEPS/" "$TEMP_CONFIG"
        
        # Use the temporary config
        CMD="python ${ENCODEVAL_DIR}/main.py --config_file $TEMP_CONFIG --model_path ${MODEL_PATH}"
    fi
    
    # Execute the command
    eval $CMD
    
    # Capture exit code
    EXIT_CODE=$?
    
    # Clean up temporary config if it exists
    [ -f "$TEMP_CONFIG" ] && rm "$TEMP_CONFIG"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $task_name completed successfully"
    else
        echo "✗ $task_name failed with error code $EXIT_CODE"
    fi
}

# Retrieval Tasks
echo ""
echo "=== RETRIEVAL TASKS ==="
echo ""
echo "--- Step 1: Training on MS MARCO ---"
#run_evaluation "MS MARCO Training (Required for retrieval evaluation)" \
#	    "retrieval_tasks/msmarco_train_qwen3.yaml"

echo ""
echo "--- Step 2: Evaluating on MIRACL using fine-tuned model ---"
#run_evaluation "MIRACL (Multilingual Information Retrieval)" \
#	    "retrieval_tasks/miracl_eval_qwen3.yaml"

# Sequence Classification Tasks
echo ""
echo "=== SEQUENCE CLASSIFICATION TASKS ==="
#run_evaluation "XNLI (Cross-lingual Natural Language Inference)" \
#    "sequence_classification_tasks/xnli_qwen3.yaml"

#run_evaluation "Amazon MASSIVE Intent Classification" \
#    "sequence_classification_tasks/amazon_massive_intent_qwen3.yaml"

# Sequence Regression Tasks
echo ""
echo "=== SEQUENCE REGRESSION TASKS ==="
run_evaluation "Seahorse (Sequence Quality Evaluation)" \
    "sequence_regression_tasks/seahorse_qwen3.yaml"

# Token Classification Tasks
echo ""
echo "=== TOKEN CLASSIFICATION TASKS ==="
#run_evaluation "NER (Named Entity Recognition)" \
#    "token_classification_tasks/ner_qwen3.yaml"


echo ""
echo "========================================="
echo "All Qwen3 evaluations completed!"
echo "Results are saved in: ${ENCODEVAL_DIR}/results/"
echo "========================================="
