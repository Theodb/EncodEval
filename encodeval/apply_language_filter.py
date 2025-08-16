"""
Simple language filter that can be imported before loading datasets.
This modifies the VALID_LANGS and related constants in datasets.py at runtime.
"""
import os
import sys

def apply_language_filter(languages):
    """
    Apply language filtering by modifying the global variables in datasets module.
    
    Args:
        languages: List of language codes to keep (e.g., ['en'])
    """
    # Import the datasets module
    import encodeval.datasets as datasets_module
    
    # Update the language constants
    datasets_module.VALID_LANGS = languages
    datasets_module.VALID_LPS = [f"{lang1}-{lang2}" for lang1 in languages for lang2 in languages]
    
    # Update EURO languages if needed
    euro_langs = ["de", "en", "es", "fr", "it", "nl", "pl", "pt"]
    configured_euro_langs = [lang for lang in languages if lang in euro_langs]
    datasets_module.VALID_EURO_LANGS = configured_euro_langs
    datasets_module.VALID_EURO_LPS = [f"{lang1}-{lang2}" for lang1 in configured_euro_langs for lang2 in languages]
    
    print(f"Applied language filter: {languages}")


def setup_english_only():
    """Convenience function to set up English-only filtering."""
    apply_language_filter(['en'])


def setup_from_env():
    """Setup language filtering from EVAL_LANGUAGES environment variable."""
    languages = os.environ.get('EVAL_LANGUAGES')
    if languages:
        apply_language_filter(languages.split(','))
        return languages.split(',')
    return None
