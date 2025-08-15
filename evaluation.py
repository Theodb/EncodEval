#!/usr/bin/env python3
"""
Comprehensive evaluation script for EncodEval results.
Analyzes and compares model performance across all tasks.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd

from encodeval.system_ranking import get_results


class EncodEvalAnalyzer:
    """Analyzes evaluation results across models and tasks."""
    
    def __init__(self, base_path: str = "./results"):
        self.base_path = Path(base_path)
        self.task_configs = {
            "SC": [
                ("xnli", ["en"]),
                ("amazon_massive_intent", ["en"]),
                ("amazon_reviews_classification", ["en"]),
                ("paws_x", ["en"])
            ],
            "SR": [
                ("seahorse", ["en"])
            ],
            "TC": [
                ("ner", ["en"])
            ],
            "IR": [
                ("msmarco", ["en"]),
                ("miracl", ["en"])
            ]
        }
        
    def get_available_models(self) -> List[str]:
        """Get list of models with results."""
        models = []
        if self.base_path.exists():
            models = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        return sorted(models)
    
    def get_available_results(self) -> Dict[str, Dict[str, List[str]]]:
        """Get structure of available results."""
        results_structure = {}
        
        for model_dir in self.base_path.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            results_structure[model_name] = {}
            
            for task_type_dir in model_dir.iterdir():
                if not task_type_dir.is_dir():
                    continue
                    
                task_type = task_type_dir.name
                datasets = [d.name for d in task_type_dir.iterdir() if d.is_dir()]
                results_structure[model_name][task_type] = datasets
                
        return results_structure
    
    def evaluate_task(self, models: List[str], task_type: str, dataset: str, 
                     langs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate a specific task across models."""
        try:
            average_scores, system_ranking = get_results(
                base_path=str(self.base_path),
                models=models,
                task_type=task_type,
                dataset=dataset,
                valid_langs=langs,
            )
            return average_scores, system_ranking
        except Exception as e:
            print(f"Error evaluating {task_type}/{dataset}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def format_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Simple table formatter without external dependencies."""
        if not rows:
            return "No data available"
            
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Create separator
        separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
        
        # Format header
        header_row = "|" + "|".join([f" {h:<{w}} " for h, w in zip(headers, col_widths)]) + "|"
        
        # Format rows
        formatted_rows = []
        for row in rows:
            formatted_row = "|" + "|".join([f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)]) + "|"
            formatted_rows.append(formatted_row)
        
        # Combine all parts
        table = [separator, header_row, separator] + formatted_rows + [separator]
        return "\n".join(table)
    
    def create_comparison_table(self, df: pd.DataFrame) -> str:
        """Create a formatted comparison table from DataFrame."""
        if df.empty:
            return "No results available"
            
        # Convert DataFrame to list format for table
        headers = ["Model"] + list(df.columns)
        rows = []
        
        for model_name, row in df.iterrows():
            row_data = [str(model_name)]
            for value in row.values:
                if isinstance(value, float):
                    row_data.append(f"{value:.4f}")
                else:
                    row_data.append(str(value))
            rows.append(row_data)
        
        return self.format_table(headers, rows)
    
    def analyze_all_tasks(self, models: Optional[List[str]] = None):
        """Analyze all tasks for specified models."""
        if models is None:
            models = self.get_available_models()
            
        if not models:
            print("No models found in results directory!")
            return
            
        print(f"\n{'='*80}")
        print(f"EncodEval Comprehensive Results Analysis")
        print(f"Models: {', '.join(models)}")
        print(f"{'='*80}\n")
        
        # Show available results structure
        structure = self.get_available_results()
        print("Available Results:")
        for model, tasks in structure.items():
            print(f"\n{model}:")
            for task_type, datasets in tasks.items():
                print(f"  {task_type}: {', '.join(datasets)}")
        
        # Analyze each task type
        all_results = {}
        
        for task_type, task_list in self.task_configs.items():
            print(f"\n{'='*60}")
            print(f"{task_type} Tasks")
            print(f"{'='*60}")
            
            task_results = {}
            
            for dataset, langs in task_list:
                print(f"\n--- {dataset} ---")
                
                # Check which models have results for this task
                available_models = []
                for model in models:
                    result_path = self.base_path / model / task_type / dataset / "results.json"
                    if result_path.exists():
                        available_models.append(model)
                    else:
                        # Check subdirectories (for IR tasks)
                        dataset_path = self.base_path / model / task_type / dataset
                        if dataset_path.exists():
                            for subdir in dataset_path.iterdir():
                                if subdir.is_dir() and (subdir / "results.json").exists():
                                    available_models.append(model)
                                    break
                
                if not available_models:
                    print(f"No results found for any model")
                    continue
                    
                average_scores, system_ranking = self.evaluate_task(
                    available_models, task_type, dataset, langs
                )
                
                # Check if we got results
                if not average_scores.empty:
                    # Store results
                    task_results[dataset] = {
                        "scores": average_scores,
                        "ranking": system_ranking
                    }
                    
                    # Display results
                    print("\nScores:")
                    print(self.create_comparison_table(average_scores))
                    
                    # Display ranking based on borda score if available
                    if not system_ranking.empty:
                        print("\nRanking:")
                        if 'borda' in system_ranking.columns:
                            # Sort by borda score
                            ranking_sorted = system_ranking.sort_values('borda', ascending=False)
                            for i, (model_name, row) in enumerate(ranking_sorted.iterrows(), 1):
                                borda_score = row['borda']
                                print(f"{i}. {model_name}: {borda_score:.4f}")
                        else:
                            # If no borda score, just show the dataframe
                            print(self.create_comparison_table(system_ranking))
                
            all_results[task_type] = task_results
        
        # Summary across all tasks
        print(f"\n{'='*80}")
        print("Summary Across All Tasks")
        print(f"{'='*80}")
        
        model_summary = {model: {"total_score": 0, "task_count": 0} for model in models}
        
        for task_type, task_results in all_results.items():
            for dataset, results in task_results.items():
                scores_data = results["scores"]
                
                # Handle DataFrame format
                for model in scores_data.index:
                    if model in model_summary:
                        # Use 'average' column if available, otherwise first column
                        if 'average' in scores_data.columns:
                            primary_score = scores_data.loc[model, 'average']
                        elif len(scores_data.columns) > 0:
                            primary_score = scores_data.loc[model].iloc[0]
                        else:
                            continue
                            
                        if isinstance(primary_score, (int, float)):
                            model_summary[model]["total_score"] += primary_score
                            model_summary[model]["task_count"] += 1
        
        # Calculate average scores
        summary_data = []
        for model, summary in model_summary.items():
            if summary["task_count"] > 0:
                avg_score = summary["total_score"] / summary["task_count"]
                summary_data.append([
                    model,
                    str(summary["task_count"]),
                    f"{avg_score:.4f}"
                ])
        
        if summary_data:
            summary_data.sort(key=lambda x: float(x[2]), reverse=True)
            print(self.format_table(
                ["Model", "Tasks Evaluated", "Average Score"],
                summary_data
            ))
        
        return all_results
    
    def export_results(self, results: Dict, output_file: str = "evaluation_report.json"):
        """Export results to JSON file."""
        # Convert DataFrames to dictionaries for JSON serialization
        json_results = {}
        for task_type, task_results in results.items():
            json_results[task_type] = {}
            for dataset, data in task_results.items():
                json_results[task_type][dataset] = {
                    "scores": data["scores"].to_dict() if isinstance(data["scores"], pd.DataFrame) else data["scores"],
                    "ranking": data["ranking"].to_dict() if isinstance(data["ranking"], pd.DataFrame) else data["ranking"]
                }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze EncodEval results")
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific models to analyze (default: all available)"
    )
    parser.add_argument(
        "--base-path", 
        default="./results",
        help="Base path for results (default: ./results)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--task-type",
        choices=["SC", "SR", "TC", "IR"],
        help="Analyze only specific task type"
    )
    parser.add_argument(
        "--dataset",
        help="Analyze only specific dataset"
    )
    
    args = parser.parse_args()
    
    analyzer = EncodEvalAnalyzer(base_path=args.base_path)
    
    if args.task_type and args.dataset:
        # Analyze specific task
        models = args.models or analyzer.get_available_models()
        langs = ["en"]  # Default to English
        
        average_scores, system_ranking = analyzer.evaluate_task(
            models, args.task_type, args.dataset, langs
        )
        
        print(f"\nResults for {args.task_type}/{args.dataset}:")
        print(analyzer.create_comparison_table(average_scores))
        
        if not system_ranking.empty:
            print("\nRanking:")
            if 'borda' in system_ranking.columns:
                ranking_sorted = system_ranking.sort_values('borda', ascending=False)
                for i, (model_name, row) in enumerate(ranking_sorted.iterrows(), 1):
                    print(f"{i}. {model_name}: {row['borda']:.4f}")
            else:
                print(analyzer.create_comparison_table(system_ranking))
    else:
        # Analyze all tasks
        results = analyzer.analyze_all_tasks(models=args.models)
        
        if args.export and results:
            analyzer.export_results(results)


if __name__ == "__main__":
    main()
