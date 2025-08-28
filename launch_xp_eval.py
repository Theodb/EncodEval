#!/usr/bin/env python3
import os
import re
import string
import random
import glob
import numpy as np

# Configuration
MODEL_PATH = "/lustre/fswork/projects/rech/vrl/uok92vw/models"
TMP_YAML = "/lustre/fswork/projects/rech/vrl/uok92vw/tmp_yaml/"
CONFIGS_BASE = "/lustre/fswork/projects/rech/vrl/uok92vw/EncodEval/configs"
FORCE_OVERWRITE = True

# Experiment settings
MODELS = ["BiQwen3-0.6B-0.05_lr1.0e-04"]
#MODELS = os.listdir(MODEL_PATH)
TASKS = ["retrieval_tasks", "sequence_classification_tasks", "sequence_regression_tasks", "token_classification_tasks"]
#TASKS = ["sequence_regression_tasks"]
LR_VALUES = [f"{lr:.2e}" for lr in np.logspace(np.log10(1e-5), np.log10(1e-4), 2)]

TASK_ABBREV = {
    "retrieval_tasks": "IR",
    "sequence_classification_tasks": "SC",
    "sequence_regression_tasks": "SR",
    "token_classification_tasks": "TC"
}

def extract_dataset_name(config_path):
    """Extract dataset name from config file"""
    try:
        with open(config_path, "r") as f:
            content = f.read()
        
        # Look for: load_dataset_from_custom_fn: !ext encodeval.datasets.DATASET_NAME
        match = re.search(r"load_dataset_from_custom_fn:\s*!ext\s+encodeval\.datasets\.([^\s\n]+)", content)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Warning: Could not read {config_path}: {e}")
    
    # Fallback to filename
    return os.path.basename(config_path).replace("xp_eval_", "").replace(".yaml", "")

def modify_config(content, model, lr_value, task):
    """Apply training modifications to config content"""
    # Clean format for LR to use in paths
    lr_clean = lr_value.replace(".", "_").replace("-", "_").replace("+", "p")
    task_abbrev = TASK_ABBREV.get(task, task[:2].upper())
    
    # Replace placeholders with actual values
    content = content.replace("${model}", model)
    content = content.replace("${lr}", lr_clean)
    content = content.replace("${task}", task_abbrev)
    
    # Core training settings
    content = re.sub(r"learning_rate:\s*[0-9.eE+-]+", f"learning_rate: {lr_value}", content)
    content = re.sub(r"warmup_ratio:\s*[0-9.]+", "warmup_ratio: 0.1", content)
    
    # LR scheduler
    if "lr_scheduler_type:" in content:
        content = re.sub(r"lr_scheduler_type:\s*\w+", "lr_scheduler_type: linear", content)
    else:
        content = re.sub(r"(learning_rate:\s*[0-9.eE+-]+)", r"\1\nlr_scheduler_type: linear", content)
    
    # Training steps
    max_steps = 1000 if task == "retrieval_tasks" else 10000
    content = re.sub(r"max_steps:\s*\d+", f"max_steps: {max_steps}", content)
    if "max_steps:" not in content:
        content = re.sub(r"(num_train_epochs:\s*\d+)", f"max_steps: {max_steps}\n# \1", content)
    
    # Batch size (32 for non-regression tasks)
    if task != "sequence_regression_tasks":
        content = re.sub(r"per_device_train_batch_size:\s*\d+", "per_device_train_batch_size: 32", content)
        content = re.sub(r"per_device_eval_batch_size:\s*\d+", "per_device_eval_batch_size: 32", content)
    
    # Best model loading
    if "load_best_model_at_end:" not in content:
        content = re.sub(r"(metric_for_best_model:\s*\w+)", r"\1\nload_best_model_at_end: true", content)
    else:
        content = re.sub(r"load_best_model_at_end:\s*\w+", "load_best_model_at_end: true", content)
    
    return content

def create_selective_cleanup_command(model, task_abbrev, dataset_name, lr_value):
    """Create selective cleanup commands for FORCE_OVERWRITE"""
    if not FORCE_OVERWRITE:
        return "# No cleanup - FORCE_OVERWRITE is disabled"
    model_path = os.path.join(MODEL_PATH, model)
    results_path = f"./results/main/{model}/{task_abbrev}/{dataset_name}"
    weights_path = f"{model_path}/evaluation/weights/{task_abbrev}/{dataset_name}"
    logs_path = f"{model_path}/evaluation/logs/{task_abbrev}/{dataset_name}"

    return f"""
# FORCE_OVERWRITE: Cleanup of existing results, logs, and weights
echo "Performing cleanup for {dataset_name} with LR={lr_value}..."

rm -rf "{results_path}" 2>/dev/null || true
rm -rf "{weights_path}" 2>/dev/null || true
rm -rf "{logs_path}" 2>/dev/null || true

echo "Cleanup completed for {dataset_name} (task: {task_abbrev}, lr: {lr_value})"
"""

def create_slurm_job(model, task, config_file, lr_value, tmp_config_path, dataset_name):
    """Create SLURM job script"""
    task_abbrev = TASK_ABBREV.get(task, task[:2].upper())
    lr_clean = lr_value.replace('.', '_').replace('-', '_').replace('+', 'p')
    
    # Job configuration
    job_name = f"{dataset_name}_{task[:4]}_{model[:10]}_{lr_clean}"
    time_limit = "10:00:00" if task == "retrieval_tasks" else "20:00:00"
    
    cleanup_cmd = create_selective_cleanup_command(model, task_abbrev, dataset_name, lr_value)
    
    return f"""#!/bin/bash
#SBATCH -A qjm@h100
#SBATCH --job-name={job_name}
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --partition=gpu_p6s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time={time_limit}
#SBATCH --output=%j_{dataset_name}.out
#SBATCH --error=%j_{dataset_name}.err

export CUDA_VISIBLE_DEVICES=0

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate encoder

export LOCAL_DATASET_DIR="/lustre/fswork/projects/rech/vrl/uok92vw/data"

echo "=========================================="
echo "Task: {task}"
echo "Model: {model}"
echo "Config: {config_file}"
echo "Dataset: {dataset_name}"
echo "LR: {lr_value}"
echo "Force Overwrite: {FORCE_OVERWRITE}"
echo "=========================================="

{cleanup_cmd}

echo ""
echo "Starting training..."
python main.py --config_file {tmp_config_path} --model_path {os.path.join(MODEL_PATH, model)}

echo ""
echo "Job completed successfully"
"""

def handle_retrieval_task(model, task, lr_value):
    """Special handling for retrieval tasks with MS-MARCO training + MIRACL evaluation"""
    config_dir = os.path.join(CONFIGS_BASE, task)
    jobs_submitted = 0
    
    # MS-MARCO training config
    msmarco_config = "xp_eval_msmarco_subset.yaml"
    msmarco_path = os.path.join(config_dir, msmarco_config)
    
    if not os.path.exists(msmarco_path):
        print(f"    MS-MARCO config not found: {msmarco_config}")
        return 0
    
    # Get dataset names
    msmarco_dataset = extract_dataset_name(msmarco_path)
    miracl_config = "xp_eval_miracl.yaml"
    miracl_path = os.path.join(config_dir, miracl_config)
    miracl_dataset = extract_dataset_name(miracl_path) if os.path.exists(miracl_path) else "miracl_en_val_as_test"
    
    print(f"    MS-MARCO: {msmarco_dataset}")
    print(f"    MIRACL: {miracl_dataset}")
    
    for lr_value in LR_VALUES:
        lr_clean = lr_value.replace('.', '_').replace('-', '_').replace('+', 'p')
        model_ft = f"{model}_ftlr_{lr_clean}"
        
        # Prepare MS-MARCO config
        with open(msmarco_path, 'r') as f:
            content = f.read()
        
        modified_content = modify_config(content, model, lr_value, task)
        modified_content = modified_content.replace("${dataset}", msmarco_dataset)
        
        # Create temporary config for MS-MARCO
        random_name = ''.join(random.choices(string.ascii_letters, k=8))
        tmp_msmarco_path = os.path.join(TMP_YAML, f"{random_name}_msmarco.yaml")
        
        with open(tmp_msmarco_path, 'w') as f:
            f.write(modified_content)
        
        # Prepare MIRACL config
        with open(miracl_path, 'r') as f:
            content = f.read()
        
        modified_content = modify_config(content, model, lr_value, task)
        modified_content = modified_content.replace("${dataset}", miracl_dataset)
        
        tmp_miracl_path = os.path.join(TMP_YAML, f"{random_name}_miracl.yaml")
        
        with open(tmp_miracl_path, 'w') as f:
            f.write(modified_content)
        
        # Enhanced cleanup for retrieval (both datasets)
        cleanup_cmd = f"""
# FORCE_OVERWRITE: Cleanup for retrieval tasks
echo "Performing cleanup for retrieval tasks..."

msmarco_path="./results/main/{model_ft}/IR/{msmarco_dataset}"
miracl_path="./results/main/{model_ft}/IR/{miracl_dataset}"

rm -rf "$msmarco_path" 2>/dev/null || true
rm -rf "$miracl_path" 2>/dev/null || true

echo "Retrieval cleanup completed"
"""
        
        fine_tuned_path = f"./results/main/{model_ft}/IR/{msmarco_dataset}/checkpoint-best"
        
        slurm_script = f"""#!/bin/bash
#SBATCH -A qjm@h100
#SBATCH --job-name=retr_{model[:10]}_{lr_clean}
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --partition=gpu_p6s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=%j_retrieval.out
#SBATCH --error=%j_retrieval.err

export CUDA_VISIBLE_DEVICES=0

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate encoder

export LOCAL_DATASET_DIR="/lustre/fswork/projects/rech/vrl/uok92vw/data"

echo "=========================================="
echo "Retrieval Task: MS-MARCO Training + MIRACL Evaluation"
echo "Model: {model}"
echo "LR: {lr_value}"
echo "MS-MARCO Dataset: {msmarco_dataset}"
echo "MIRACL Dataset: {miracl_dataset}"
echo "Force Overwrite: {FORCE_OVERWRITE}"
echo "=========================================="

{cleanup_cmd}

echo ""
echo "Step 1: Training on MS-MARCO..."
python main.py --config_file {tmp_msmarco_path} --model_path {os.path.join(MODEL_PATH, model)}

echo ""
echo "Step 2: Evaluating on MIRACL..."
python main.py --config_file {tmp_miracl_path} --model_path {fine_tuned_path}

echo ""
echo "Retrieval job completed successfully"
"""
        
        # Submit job
        script_name = f"tmp_{random_name}_retr.sh"
        with open(script_name, "w") as f:
            f.write(slurm_script)
        
        result = os.system(f"sbatch {script_name}")
        print(f"      LR {lr_value}: {'✓' if result == 0 else '✗'} (MS-MARCO + MIRACL)")
        
        os.remove(script_name)
        jobs_submitted += 1
    
    return jobs_submitted

def process_task(model, task):
    """Process all configs for a task"""
    if task == "retrieval_tasks":
        return handle_retrieval_task(model, task, LR_VALUES[0])
    
    config_dir = os.path.join(CONFIGS_BASE, task)
    
    # Find all xp_eval config files
    configs = [os.path.basename(f) for f in glob.glob(os.path.join(config_dir, "xp_eval*.yaml"))]
    
    if not configs:
        print(f"  No configs found for {task}")
        return 0
    
    print(f"  Found {len(configs)} config(s): {configs}")
    jobs_submitted = 0
    
    for config_file in configs:
        config_path = os.path.join(config_dir, config_file)
        if not os.path.exists(config_path):
            continue
            
        # Extract dataset name
        dataset_name = extract_dataset_name(config_path)
        print(f"    {config_file} -> {dataset_name}")
        
        for lr_value in LR_VALUES:
            # Read and modify config
            with open(config_path, 'r') as f:
                content = f.read()
            
            modified_content = modify_config(content, model, lr_value, task)
            modified_content = modified_content.replace("${dataset}", dataset_name)
            
            # Create temporary config file
            random_name = ''.join(random.choices(string.ascii_letters, k=8))
            tmp_config_path = os.path.join(TMP_YAML, f"{random_name}.yaml")
            
            with open(tmp_config_path, 'w') as f:
                f.write(modified_content)
            
            # Create and submit SLURM job
            slurm_script = create_slurm_job(model, task, config_file, lr_value, tmp_config_path, dataset_name)
            script_name = f"tmp_{random_name}.sh"
            
            with open(script_name, "w") as f:
                f.write(slurm_script)
            
            result = os.system(f"sbatch {script_name}")
            print(f"      LR {lr_value}: {'✓' if result == 0 else '✗'}")
            
            os.remove(script_name)
            jobs_submitted += 1
    
    return jobs_submitted

def main():
    """Main execution"""
    os.makedirs(TMP_YAML, exist_ok=True)
    
    print("=" * 60)
    print("LAUNCH SCRIPT - SELECTIVE CLEANUP")
    print("=" * 60)
    print(f"Models: {MODELS}")
    print(f"Tasks: {TASKS}")
    print(f"Learning Rates: {LR_VALUES}")
    print(f"Force Overwrite: {FORCE_OVERWRITE}")
    print("=" * 60)
    
    if FORCE_OVERWRITE:
        print("⚠️  Cleanup enabled:")
        print("   - Deletes entire /results/main/model_ftlr_lr/task/dataset")
        print("   - Includes logs, weights, and results")
        import time
        time.sleep(3)
    
    total_jobs = 0
    
    for model in MODELS:
        print(f"\n[{model}]")
        
        for task in TASKS:
            print(f"\nTask: {task}")
            jobs = process_task(model, task)
            total_jobs += jobs
    
    print("\n" + "=" * 60)
    print(f"SUBMITTED {total_jobs} JOBS")
    print("=" * 60)

if __name__ == "__main__":
    main()
