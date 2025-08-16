"""
Language configuration module for EncodEval.
This module allows specifying languages in configuration files
without modifying the existing dataset functions.
"""
import os


def set_language_configuration(languages):
    """
    Set the languages to be used in dataset loading.
    
    Args:
        languages: List of language codes (e.g., ['en']) or string of comma-separated codes
    """
    if isinstance(languages, list):
        lang_str = ','.join(languages)
    else:
        lang_str = str(languages)
    
    os.environ['EVAL_LANGUAGES'] = lang_str
    print(f"Language configuration set to: {lang_str}")


def get_configured_languages(default_langs=None):
    """
    Get the configured languages from environment variable.
    
    Args:
        default_langs: Default languages to use if none configured
        
    Returns:
        List of language codes
    """
    if default_langs is None:
        default_langs = ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"]
    
    configured_langs = os.environ.get('EVAL_LANGUAGES', None)
    if configured_langs:
        return configured_langs.split(',')
    return default_langs


def apply_language_filter_to_datasets():
    """
    Apply language filtering by modifying the global VALID_LANGS variables
    in the datasets module.
    """
    import encodeval.datasets as datasets_module
    
    # Get configured languages
    configured_langs = get_configured_languages()
    
    # Update the global variables in the datasets module
    datasets_module.VALID_LANGS = configured_langs
    datasets_module.VALID_LPS = [f"{lang1}-{lang2}" for lang1 in configured_langs for lang2 in configured_langs]
    
    # Also update EURO languages if needed
    euro_langs = ["de", "en", "es", "fr", "it", "nl", "pl", "pt"]
    configured_euro_langs = [lang for lang in configured_langs if lang in euro_langs]
    datasets_module.VALID_EURO_LANGS = configured_euro_langs
    datasets_module.VALID_EURO_LPS = [f"{lang1}-{lang2}" for lang1 in configured_euro_langs for lang2 in configured_langs]
    
    print(f"Applied language filter: {configured_langs}")
