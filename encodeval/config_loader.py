"""
Configuration loader for EncodEval that handles language specification.
"""
import yaml
import os
from pathlib import Path
from language_config import set_language_configuration, apply_language_filter_to_datasets


def load_config_with_language_filter(config_path):
    """
    Load a YAML configuration file and apply language filtering.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Loaded configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract language configuration if present
    if 'eval_config' in config and 'languages' in config['eval_config']:
        languages = config['eval_config']['languages']
        print(f"Found language configuration: {languages}")
        
        # Set the language configuration
        set_language_configuration(languages)
        
        # Apply the language filter to datasets
        apply_language_filter_to_datasets()
    else:
        print("No language configuration found, using all available languages")
    
    return config


def setup_language_from_config(config_path):
    """
    Setup language configuration from a config file without loading the full config.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'eval_config' in config and 'languages' in config['eval_config']:
            languages = config['eval_config']['languages']
            set_language_configuration(languages)
            apply_language_filter_to_datasets()
            return languages
        else:
            print("No language configuration found in config file")
            return None
    except Exception as e:
        print(f"Error reading config file {config_path}: {e}")
        return None


def setup_language_from_env():
    """
    Setup language configuration from environment variable.
    This can be used as an alternative to config files.
    """
    languages = os.environ.get('EVAL_LANGUAGES')
    if languages:
        set_language_configuration(languages.split(','))
        apply_language_filter_to_datasets()
        return languages.split(',')
    return None
