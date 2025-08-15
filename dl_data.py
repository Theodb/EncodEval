#!/usr/bin/env python3
"""
Script to download and save EncodEval datasets locally for offline usage.
"""
import os
from datasets import load_dataset
from tqdm import tqdm

# Define the local storage directory
LOCAL_DATASET_DIR = "/lustre/fswork/projects/rech/vrl/uok92vw/data"  # Change this to your preferred path

# List of datasets used in EncodEval
DATASETS = [
    # Sequence Classification
    ("mteb/xnli", None),  # Will need to handle configs separately
    ("Samoed/AmazonReviewsClassification", None),  # Will need to handle configs separately
    ("mteb/amazon_massive_intent", None),  # Will need to handle configs separately
    ("hgissbkh/paws-x", None),
    ("ObscuraCoder/code-classification", "defect-detection"),
    ("ObscuraCoder/code-classification", "complexity-prediction"),
    ("trl-lib/math_shepherd", None),
    
    # Sequence Regression
    ("RicardoRei/wmt-da-human-evaluation", None),
    ("hgissbkh/seahorse", None),
    
    # Token Classification
    ("hgissbkh/ner", None),
    
    # Retrieval
    ("bclavie/msmarco-10m-triplets", None),
    ("sentence-transformers/miracl", None),  # Will need to handle configs separately
    ("sentence-transformers/mldr", None),  # Will need to handle configs separately
    ("Samoed/WikipediaRetrievalMultilingual", None),  # Will need to handle configs separately
    ("hgissbkh/multilingual_cc_news", None),  # Will need to handle configs separately
    ("sentence-transformers/codesearchnet", None),
    ("mteb/cqadupstack-mathematica", None),  # Multiple splits
    ("hgissbkh/math_formula_retrieval_sampled", None),
]

# Datasets that need config handling
DATASETS_WITH_CONFIGS = {
    "mteb/xnli": ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"],
    "Samoed/AmazonReviewsClassification": ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"],
    "mteb/amazon_massive_intent": ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh-CN"],
    "sentence-transformers/miracl": [f"{lang}-triplet" for lang in ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"]],
    "sentence-transformers/mldr": [f"{lang}-triplet" for lang in ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"]],
    "Samoed/WikipediaRetrievalMultilingual": [],  # Need to get configs dynamically
    "hgissbkh/multilingual_cc_news": ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"],
}

# Special handling for WikipediaRetrievalMultilingual
WIKIPEDIA_RETRIEVAL_CONFIGS = []
for lang in ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"]:
    WIKIPEDIA_RETRIEVAL_CONFIGS.extend([f"{lang}-queries", f"{lang}-corpus", f"{lang}-qrels"])
DATASETS_WITH_CONFIGS["Samoed/WikipediaRetrievalMultilingual"] = WIKIPEDIA_RETRIEVAL_CONFIGS

# Special handling for cqadupstack-mathematica
CQADUPSTACK_SPLITS = {
    "mteb/cqadupstack-mathematica": {
        "queries": {"split": "queries"},
        "corpus": {"split": "corpus"},
        "default": {"split": "test"}
    }
}

def download_and_save_dataset(dataset_name, config=None, special_splits=None):
    """Download and save a dataset to local storage."""
    # Create save directory
    save_dir = os.path.join(LOCAL_DATASET_DIR, dataset_name.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        if special_splits:
            # Handle datasets with special split configurations
            for split_name, split_config in special_splits.items():
                print(f"  Downloading split: {split_name}")
                if split_name == "queries":
                    dataset = load_dataset(dataset_name, "queries", split=split_config["split"])
                elif split_name == "corpus":
                    dataset = load_dataset(dataset_name, "corpus", split=split_config["split"])
                else:
                    dataset = load_dataset(dataset_name, split=split_config["split"])
                
                # Save based on the structure
                if hasattr(dataset, 'save_to_disk'):
                    if split_name in ["queries", "corpus"]:
                        # Save with config name
                        save_path = os.path.join(save_dir, split_name)
                        dataset.save_to_disk(save_path)
                    else:
                        save_path = save_dir
                        dataset.save_to_disk(save_path)
                    print(f"  Saved to: {save_path}")
        else:
            # Standard dataset loading
            if config:
                dataset = load_dataset(dataset_name, config)
                save_path = os.path.join(save_dir, config)
            else:
                dataset = load_dataset(dataset_name)
                save_path = save_dir
            
            dataset.save_to_disk(save_path)
            print(f"  Saved to: {save_path}")
            
    except Exception as e:
        print(f"  ERROR: Failed to download {dataset_name} {config}: {str(e)}")
        return False
    
    return True

def main():
    """Main function to download all datasets."""
    print(f"Downloading datasets to: {LOCAL_DATASET_DIR}")
    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
    
    # Download datasets without configs
    print("\n=== Downloading standard datasets ===")
    for dataset_name, config in DATASETS:
        if dataset_name not in DATASETS_WITH_CONFIGS and dataset_name not in CQADUPSTACK_SPLITS:
            print(f"\nDownloading: {dataset_name}")
            download_and_save_dataset(dataset_name, config)
    
    # Download datasets with configs
    print("\n=== Downloading datasets with configurations ===")
    for dataset_name, configs in DATASETS_WITH_CONFIGS.items():
        print(f"\nDownloading: {dataset_name}")
        for config in tqdm(configs, desc=f"Configs for {dataset_name}"):
            print(f"  Config: {config}")
            download_and_save_dataset(dataset_name, config)
    
    # Download datasets with special splits
    print("\n=== Downloading datasets with special splits ===")
    for dataset_name, splits in CQADUPSTACK_SPLITS.items():
        print(f"\nDownloading: {dataset_name}")
        download_and_save_dataset(dataset_name, special_splits=splits)
    
    print("\n=== Download complete! ===")
    print(f"\nTo use these datasets offline, set the environment variable:")
    print(f"export LOCAL_DATASET_DIR={os.path.abspath(LOCAL_DATASET_DIR)}")

if __name__ == "__main__":
    main()
