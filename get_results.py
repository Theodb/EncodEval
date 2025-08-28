"""
Script to find the best fine-tuning learning rate for each pretrained model on each dataset
Analyzes evaluation results with hyperparameters and logs to W&B
"""

import json
import argparse
import re
from pathlib import Path
from prettytable import PrettyTable
from sklearn.metrics import f1_score
import numpy as np
from collections import defaultdict
from datetime import datetime
import pandas as pd

# Add wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable logging.")

def extract_hyperparameters(model_name):
    """
    Extract hyperparameters from model name
    Expected format: BiQwen3-0.6B-{masking_ratio}_lr{learning_rate}_ftlr_{ft_learning_rate}
    Example: BiQwen3-0.6B-0.30_lr1e-4_ftlr_1_00e_04 or BiQwen3-0.6B-0.30_lr0.0001_ftlr_1_00e_05
    """
    hyperparams = {
        'base_model': 'BiQwen3-0.6B',
        'masking_ratio': None,
        'learning_rate': None,
        'finetuning_lr': None,
        'full_name': model_name
    }
    
    # Extract masking ratio
    masking_match = re.search(r'BiQwen3-0\.6B-(\d+\.\d+)', model_name)
    if masking_match:
        hyperparams['masking_ratio'] = float(masking_match.group(1))
    
    # Extract learning rate - handle scientific notation and decimal
    lr_match = re.search(r'_lr([\d.e-]+)(?:_|$)', model_name)
    if lr_match:
        lr_str = lr_match.group(1)
        try:
            hyperparams['learning_rate'] = float(lr_str)
        except ValueError:
            # Handle special cases
            pass
    
    # Extract fine-tuning learning rate
    # Format could be: ftlr_1_00e_04 (representing 1.00e-04)
    ft_match = re.search(r'ftlr_(.+?)(?:_|$)', model_name)
    if ft_match:
        ft_lr_str = ft_match.group(1)
        # Convert underscore format to proper scientific notation
        # e.g., 1_00e_04 -> 1.00e-04
        ft_lr_str = ft_lr_str.replace('_', '.')
        # Fix the 'e.' pattern that might occur
        ft_lr_str = re.sub(r'e\.', 'e-', ft_lr_str)
        try:
            hyperparams['finetuning_lr'] = float(ft_lr_str)
        except ValueError:
            # Try another pattern if the above doesn't work
            try:
                # Handle format like "1_00e_04" -> "1.00e-04"
                parts = ft_match.group(1).split('e')
                if len(parts) == 2:
                    mantissa = parts[0].replace('_', '.')
                    exponent = parts[1].replace('_', '')
                    ft_lr_str = f"{mantissa}e-{exponent}"
                    hyperparams['finetuning_lr'] = float(ft_lr_str)
            except:
                pass
    
    # Create pretrained model identifier (without ftlr)
    if hyperparams['masking_ratio'] is not None and hyperparams['learning_rate'] is not None:
        hyperparams['pretrained_model_id'] = f"BiQwen3-0.6B-{hyperparams['masking_ratio']:.2f}_lr{hyperparams['learning_rate']:.1e}"
    else:
        hyperparams['pretrained_model_id'] = model_name.split('_ftlr')[0] if '_ftlr' in model_name else model_name
    
    return hyperparams

# Parse command line arguments
parser = argparse.ArgumentParser(description='Find best fine-tuning LR for each pretrained model on each dataset')
parser.add_argument('--wandb', action='store_true', help='Push results to wandb')
parser.add_argument('--wandb-entity', type=str, default='Dec2Enc', help='Wandb entity/team name')
parser.add_argument('--wandb-project', type=str, default='FT_LR_Optimization', help='Wandb project name')
parser.add_argument('--wandb-group', type=str, default=None, help='Group name for related runs')
parser.add_argument('--wandb-tags', nargs='+', default=[], help='Tags for the run')
parser.add_argument('--base-path', type=str, default='./results/main', help='Base path for results')
parser.add_argument('--output-csv', type=str, default='best_ft_lr_results.csv', help='Output CSV file for best configurations')
args = parser.parse_args()

base_path = Path(args.base_path)
models = sorted([d.name for d in base_path.iterdir() if d.is_dir()])

print("="*100)
print(" FINE-TUNING LEARNING RATE OPTIMIZATION ANALYSIS")
print("="*100)
print(f"Models found: {len(models)}")
print()

def process_results(data, split_name):
    """Process results for a given split (validation or test)"""
    results = {}
    
    if split_name not in data:
        return None
    
    split_data = data[split_name]
    
    # Sequence Classification - accuracy
    if 'accuracy' in split_data:
        acc_data = split_data['accuracy']
        if isinstance(acc_data, list):
            results['accuracy'] = sum(acc_data) / len(acc_data)
        else:
            results['accuracy'] = acc_data
    
    # Sequence Regression - spearman correlation
    if 'prediction' in split_data and 'label' in split_data:
        from scipy.stats import spearmanr
        preds = split_data['prediction']
        labels = split_data['label']
        if len(preds) == len(labels) and len(preds) > 0:
            corr, _ = spearmanr(preds, labels)
            results['spearman'] = corr
    
    # Token Classification - F1 score
    if 'average' in split_data and split_data['average'] is not None:
        avg = split_data['average']
        if isinstance(avg, dict) and 'micro-f1' in avg:
            results['f1'] = avg['micro-f1']
        elif isinstance(avg, (int, float)):
            results['f1'] = avg
    elif 'per_instance' in split_data and split_data['per_instance']:
        per_instance = split_data['per_instance']
        if 'prediction_token' in per_instance and 'labels_token' in per_instance:
            all_preds = []
            all_labels = []
            for pred_list, label_list in zip(per_instance['prediction_token'], 
                                            per_instance['labels_token']):
                if pred_list and label_list:
                    all_preds.extend(pred_list)
                    all_labels.extend(label_list)
            if all_preds and all_labels:
                f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
                results['f1'] = f1_micro
        elif isinstance(per_instance, dict):
            if 'f1' in per_instance and isinstance(per_instance['f1'], list):
                f1_scores = [s for s in per_instance['f1'] if s is not None]
                if f1_scores:
                    results['f1'] = sum(f1_scores) / len(f1_scores)
    
    # Information Retrieval - NDCG
    if 'ndcg' in split_data:
        ndcg_scores = split_data['ndcg']
        if isinstance(ndcg_scores, list) and ndcg_scores:
            results['ndcg'] = sum(ndcg_scores) / len(ndcg_scores)
    
    return results

# Task types and their metrics
task_types = {
    'SC': 'Sequence Classification',
    'SR': 'Sequence Regression',
    'TC': 'Token Classification',
    'IR': 'Information Retrieval'
}

task_metrics = {
    'SC': 'accuracy',
    'SR': 'spearman',
    'TC': 'f1',
    'IR': 'ndcg'
}

# Collect all results organized by pretrained model and dataset
results_by_pretrained = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# Structure: results_by_pretrained[pretrained_model_id][dataset_key][ft_lr] = score

print("Collecting results...")
print("-" * 50)

for model in models:
    hp = extract_hyperparameters(model)
    
    if hp['finetuning_lr'] is None:
        print(f"Warning: No fine-tuning LR found for {model}, skipping...")
        continue
    
    pretrained_id = hp['pretrained_model_id']
    
    for task_type in task_types.keys():
        task_dir = base_path / model / task_type
        if task_dir.exists():
            for dataset_dir in task_dir.iterdir():
                if dataset_dir.is_dir():
                    results_file = dataset_dir / "results.json"
                    
                    if results_file.exists():
                        dataset_name = dataset_dir.name
                        dataset_key = f"{task_type}/{dataset_name}"
                        metric_name = task_metrics[task_type]
                        
                        with open(results_file) as f:
                            data = json.load(f)
                            
                            # Use test results for finding best configuration
                            test_res = process_results(data, 'test')
                            if test_res and metric_name in test_res:
                                score = test_res[metric_name]
                                results_by_pretrained[pretrained_id][dataset_key][hp['finetuning_lr']].append(score)

print(f"Found {len(results_by_pretrained)} unique pretrained models")
print()

# Find best fine-tuning LR for each pretrained model on each dataset
best_configs = []
print("="*100)
print(" BEST FINE-TUNING LEARNING RATES PER PRETRAINED MODEL AND DATASET")
print("="*100)

for pretrained_id in sorted(results_by_pretrained.keys()):
    print(f"\n📊 Pretrained Model: {pretrained_id}")
    print("-" * 80)
    
    # Create a table for this pretrained model
    table = PrettyTable()
    table.field_names = ["Dataset", "Best FT LR", "Best Score", "All FT LRs Tested", "All Scores"]
    
    for dataset_key in sorted(results_by_pretrained[pretrained_id].keys()):
        ft_lr_results = results_by_pretrained[pretrained_id][dataset_key]
        
        # Calculate average score for each FT LR (in case of multiple runs)
        ft_lr_avg_scores = {}
        for ft_lr, scores in ft_lr_results.items():
            ft_lr_avg_scores[ft_lr] = sum(scores) / len(scores) if scores else 0
        
        if ft_lr_avg_scores:
            # Find best FT LR
            best_ft_lr = max(ft_lr_avg_scores.keys(), key=lambda k: ft_lr_avg_scores[k])
            best_score = ft_lr_avg_scores[best_ft_lr]
            
            # Sort FT LRs for display
            sorted_ft_lrs = sorted(ft_lr_avg_scores.keys())
            all_ft_lrs_str = ", ".join([f"{lr:.1e}" for lr in sorted_ft_lrs])
            all_scores_str = ", ".join([f"{ft_lr_avg_scores[lr]:.4f}" for lr in sorted_ft_lrs])
            
            table.add_row([
                dataset_key,
                f"{best_ft_lr:.1e}",
                f"{best_score:.4f}",
                all_ft_lrs_str,
                all_scores_str
            ])
            
            # Extract pretrained hyperparameters from ID
            masking_ratio = float(re.search(r'-(\d+\.\d+)_lr', pretrained_id).group(1)) if re.search(r'-(\d+\.\d+)_lr', pretrained_id) else None
            learning_rate = float(re.search(r'_lr([\d.e-]+)', pretrained_id).group(1)) if re.search(r'_lr([\d.e-]+)', pretrained_id) else None
            
            best_configs.append({
                'pretrained_model': pretrained_id,
                'masking_ratio': masking_ratio,
                'pretraining_lr': learning_rate,
                'dataset': dataset_key,
                'best_finetuning_lr': best_ft_lr,
                'best_score': best_score,
                'num_ft_lrs_tested': len(ft_lr_avg_scores),
                'all_ft_lrs': sorted_ft_lrs,
                'all_scores': [ft_lr_avg_scores[lr] for lr in sorted_ft_lrs]
            })
    
    print(table)

# Save results to CSV
if best_configs:
    df = pd.DataFrame(best_configs)
    df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Results saved to {args.output_csv}")

# Statistical summary
print("\n" + "="*100)
print(" STATISTICAL SUMMARY")
print("="*100)

if best_configs:
    df = pd.DataFrame(best_configs)
    
    # Group by pretrained model
    print("\n📈 Average Performance by Pretrained Model:")
    pretrained_summary = df.groupby('pretrained_model')['best_score'].agg(['mean', 'std', 'count'])
    print(pretrained_summary.sort_values('mean', ascending=False))
    
    # Find globally best FT LR across all configurations
    print("\n🎯 Most Frequently Optimal Fine-tuning LRs:")
    ft_lr_counts = df['best_finetuning_lr'].value_counts()
    for ft_lr, count in ft_lr_counts.head(5).items():
        percentage = (count / len(df)) * 100
        print(f"  {ft_lr:.1e}: {count} times ({percentage:.1f}%)")
    
    # Best FT LR per task type
    print("\n📊 Best Fine-tuning LRs by Task Type:")
    for task_type, task_name in task_types.items():
        task_df = df[df['dataset'].str.startswith(f"{task_type}/")]
        if not task_df.empty:
            most_common_ft_lr = task_df['best_finetuning_lr'].mode()[0]
            avg_score = task_df['best_score'].mean()
            print(f"  {task_type} ({task_name}): {most_common_ft_lr:.1e} (avg score: {avg_score:.4f})")

# W&B logging
if args.wandb and WANDB_AVAILABLE and best_configs:
    print("\n" + "="*100)
    print(" LOGGING TO WEIGHTS & BIASES")
    print("="*100)
    
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"ft_lr_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        group=args.wandb_group or "ft_lr_analysis",
        tags=args.wandb_tags + ["optimization", "BiQwen3-0.6B"],
        config={
            "analysis_type": "fine_tuning_lr_optimization",
            "num_pretrained_models": len(results_by_pretrained),
            "total_configurations": len(best_configs)
        }
    )
    
    # Log the main results table
    wandb.log({"best_ft_lr_table": wandb.Table(dataframe=df)})
    
    # Create visualization of best FT LR distribution
    if len(df) > 1:
        # Histogram of best FT LRs
        ft_lr_hist_data = [[ft_lr, count] for ft_lr, count in df['best_finetuning_lr'].value_counts().items()]
        ft_lr_hist_table = wandb.Table(data=ft_lr_hist_data, columns=["ft_lr", "count"])
        wandb.log({"ft_lr_distribution": wandb.plot.bar(ft_lr_hist_table, "ft_lr", "count", 
                                                        title="Distribution of Best Fine-tuning LRs")})
        
        # Scatter plot: Pretraining LR vs Best FT LR
        if 'pretraining_lr' in df.columns and df['pretraining_lr'].notna().any():
            scatter_data = [[row['pretraining_lr'], row['best_finetuning_lr'], row['best_score']] 
                           for _, row in df.iterrows() 
                           if pd.notna(row['pretraining_lr'])]
            scatter_table = wandb.Table(data=scatter_data, columns=["pretraining_lr", "best_ft_lr", "score"])
            wandb.log({"pretraining_vs_finetuning_lr": wandb.plot.scatter(scatter_table, 
                                                                          "pretraining_lr", 
                                                                          "best_ft_lr",
                                                                          title="Pretraining LR vs Best Fine-tuning LR")})
        
        # Heatmap data: Pretrained model vs Dataset with best FT LR as values
        pivot_table = df.pivot_table(values='best_finetuning_lr', 
                                     index='pretrained_model', 
                                     columns='dataset', 
                                     aggfunc='first')
        
        # Log pivot table
        wandb.log({"ft_lr_heatmap": wandb.Table(dataframe=pivot_table.reset_index())})
    
    # Log summary statistics
    run.summary.update({
        'total_experiments': sum(len(results_by_pretrained[pm][dk]) for pm in results_by_pretrained for dk in results_by_pretrained[pm]),
        'unique_pretrained_models': len(results_by_pretrained),
        'unique_datasets': len(set(config['dataset'] for config in best_configs)),
        'most_common_best_ft_lr': df['best_finetuning_lr'].mode()[0] if not df.empty else None,
        'avg_best_score': df['best_score'].mean() if not df.empty else None
    })
    
    run.finish()
    print(f"\n✅ Results successfully logged to W&B!")
    print(f"   Project: {args.wandb_entity}/{args.wandb_project}")

print("\n" + "="*100)
print(" ANALYSIS COMPLETE")
print("="*100)
