import os
import random

from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import get_dataset_config_names as _get_dataset_config_names
from datasets import get_dataset_split_names as _get_dataset_split_names
from datasets import load_dataset as _load_dataset
import pandas as pd
import random
from tqdm import tqdm


# Wrapper for loading datasets
def load_dataset(*args, **kwargs) -> DatasetDict:
    print(
        f"Loading dataset {args[0]}" + 
        (f", {kwargs['name']}" if "name" in kwargs else "") + 
        (f", {kwargs['split']}" if "split" in kwargs else "") 
    )
    dataset_name = args[0].split("/")[-1]
    
    if "LOCAL_DATASET_DIR" in os.environ:
        print(f"Loading dataset from local storage at {os.environ['LOCAL_DATASET_DIR']}")
        return _load_dataset(f"{os.environ['LOCAL_DATASET_DIR']}/{dataset_name}", *args[1:], **kwargs)
    
    else:
        print("Loading dataset from Hugging Face")
        return _load_dataset(*args, **kwargs)

# Wrapper for getting dataset config names
def get_dataset_config_names(*args, **kwargs) -> DatasetDict:
    dataset_name = args[0].split("/")[-1]
    if "LOCAL_DATASET_DIR" in os.environ:
        return _get_dataset_config_names(f"{os.environ['LOCAL_DATASET_DIR']}/{dataset_name}", *args[1:], **kwargs)
    else:
        return _get_dataset_config_names(*args, **kwargs)
    
# Wrapper for getting dataset split names
def get_dataset_split_names(*args, **kwargs) -> DatasetDict:
    dataset_name = args[0].split("/")[-1]
    if "LOCAL_DATASET_DIR" in os.environ:
        return _get_dataset_split_names(f"{os.environ['LOCAL_DATASET_DIR']}/{dataset_name}", *args[1:], **kwargs)
    else:
        return _get_dataset_split_names(*args, **kwargs)

# Train-test split function for retrieval datasets
def split_retrieval_dataset(dataset, train_size=0.95, seed=42, shard_size=10_000):
    random.seed(seed)
    anchors = sorted(list(set(dataset["anchor"])))
    anchors_train = random.sample(anchors, round(train_size * len(anchors)))
    dataset_train, dataset_test = [], []
    for i in tqdm(range(0, len(dataset), shard_size)):
        shard = pd.DataFrame(dataset[i:i+shard_size])
        is_train = shard["anchor"].isin(anchors_train)
        dataset_train.append(Dataset.from_pandas(shard.loc[is_train].reset_index(drop=True)))
        dataset_test.append(Dataset.from_pandas(shard.loc[~is_train].reset_index(drop=True)))
    dataset_train = concatenate_datasets(dataset_train)
    dataset_test = concatenate_datasets(dataset_test)
    return dataset_train, dataset_test

# Valid languages and pairs
VALID_LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "tr", "vi", "zh"]
VALID_LPS = [f"{lang1}-{lang2}" for lang1 in VALID_LANGS for lang2 in VALID_LANGS]
VALID_EURO_LANGS = ["de", "en", "es", "fr", "it", "nl", "pl", "pt"]
VALID_EURO_LPS = [f"{lang1}-{lang2}" for lang1 in VALID_EURO_LANGS for lang2 in VALID_LANGS]
LANG_IDS_DICT_3_TO_2 = {
    "ara": "ar",
    "deu": "de",
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "hin": "hi",
    "ita": "it",
    "jpn": "ja",
    "nld": "nl",
    "pol": "pl",
    "por": "pt",
    "rus": "ru",
    "tur": "tr",
    "vie": "vi",
    "zho": "zh",
}
LANG_IDS_DICT_2_TO_3 = {v: k for k, v in LANG_IDS_DICT_3_TO_2.items()}
LANG_IDS_DICT_3_ALPHABET_TO_2 = {
    "arb_Arab": "ar",
    "deu_Latn": "de",
    "eng_Latn": "en",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "hin_Deva": "hi",
    "ita_Latn": "it",
    "jpn_Jpan": "ja",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "por_Latn": "pt",
    "rus_Cyrl": "ru",
    "tur_Latn": "tr",
    "vie_Latn": "vi",
    "zho_Hans": "zh",
}
LANG_IDS_DICT_2_TO_3_ALPHABET = {v: k for k, v in LANG_IDS_DICT_3_ALPHABET_TO_2.items()}
LANG_IDS_DICT_FULL_TO_2 = {
    "arabic": "ar",
    "german": "de",
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "turkish": "tr",
    "vietnamese": "vi",
    "chinese": "zho",
}
LANG_IDS_DICT_2_TO_FULL = {v: k for k, v in LANG_IDS_DICT_FULL_TO_2.items()}


#========================
# Sequence classification
#========================

# Multilingual
def xnli() -> DatasetDict:
    config_names = get_dataset_config_names("mteb/xnli")
    valid_config_names = sorted(list(set(config_names) & set(VALID_LANGS)))
    dataset_train, dataset_val, dataset_test = [], [], []
    for config_name in tqdm(valid_config_names):
        subset = load_dataset("mteb/xnli", name=config_name) 
        dataset_train.append(subset["train"])
        dataset_val.append(subset["validation"])
        dataset_test.append(subset["test"])
    dataset = DatasetDict({
        "train": concatenate_datasets(dataset_train),
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    dataset = dataset.rename_column("lang", "subset")
    def concat_premise_hypothesis(example):
        example["text"] = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
        return example
    dataset = dataset.map(concat_premise_hypothesis)
    dataset = dataset.remove_columns(["premise", "hypothesis"])
    return dataset

def amazon_reviews_classification() -> DatasetDict:
    config_names = get_dataset_config_names("Samoed/AmazonReviewsClassification")
    valid_config_names = sorted(list(set(config_names) & set(VALID_LANGS)))
    dataset_train, dataset_val, dataset_test = [], [], []
    for config_name in tqdm(valid_config_names):
        subset = load_dataset("Samoed/AmazonReviewsClassification", name=config_name) 
        subset["train"], subset["validation"], subset["test"] = (
            subset["train"].add_column("subset", [config_name] * len(subset["train"])),
            subset["validation"].add_column("subset", [config_name] * len(subset["validation"])),
            subset["test"].add_column("subset", [config_name] * len(subset["test"])),
        )
        dataset_train.append(subset["train"])
        dataset_val.append(subset["validation"])
        dataset_test.append(subset["test"])
    dataset = DatasetDict({
        "train": concatenate_datasets(dataset_train),
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    return dataset

def amazon_massive_intent() -> DatasetDict:
    config_names = get_dataset_config_names("mteb/amazon_massive_intent")
    valid_config_names = sorted(list(
        set([config_name[:2] for config_name in config_names]) & set(VALID_LANGS)
    ))
    dataset_train, dataset_val, dataset_test = [], [], []
    for config_name in tqdm(valid_config_names):
        config_name = "zh-CN" if config_name == "zh" else config_name
        subset = load_dataset("mteb/amazon_massive_intent", name=config_name) 
        subset = subset.map(lambda _: {"lang": "zh"}) if config_name == "zh-CN" else subset
        dataset_train.append(subset["train"])
        dataset_val.append(subset["validation"])
        dataset_test.append(subset["test"])
    dataset = DatasetDict({
        "train": concatenate_datasets(dataset_train),
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    dataset = dataset.rename_column("lang", "subset")
    labels = sorted(list(
        set(dataset["train"]["label"]) | 
        set(dataset["validation"]["label"]) | 
        set(dataset["test"]["label"])
    ))
    def get_label(example):
        example["label"] = labels.index(example["label"])
        return example
    dataset = dataset.map(get_label)
    dataset = dataset.remove_columns(["id", "label_text"])
    return dataset

def paws_x() -> DatasetDict:
    dataset = load_dataset("hgissbkh/paws-x")
    dataset = dataset.filter(lambda x: x["lang"] in VALID_LANGS)
    def concat_s1_s2(example):
        example["text"] = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}"
        return example
    dataset = dataset.map(concat_s1_s2, remove_columns=["sentence1", "sentence2"])
    dataset = dataset.rename_column("lang", "subset")
    return dataset

# Code
def code_defect_detection() -> DatasetDict:
    dataset = load_dataset("ObscuraCoder/code-classification", "defect-detection")
    dataset = dataset.rename_column("source_code", "text")
    return dataset

def code_complexity_prediction() -> DatasetDict:
    dataset = load_dataset("ObscuraCoder/code-classification", "complexity-prediction")
    dataset = dataset.rename_column("source_code", "text")
    return dataset

# Math
def math_shepherd() -> DatasetDict:
    dataset = load_dataset("trl-lib/math_shepherd")
    dataset = dataset.filter(lambda x: len(x["labels"]) == 3)
    def get_label(example):
        example["label"] = (sum(example["labels"]) == len(example["labels"])) * 1
        return example
    dataset = dataset.map(get_label, remove_columns=["labels"])
    dataset_train_true = dataset["train"].filter(lambda x: x["label"] == True)
    dataset_train_false = dataset["train"].filter(lambda x: x["label"] == False)
    dataset_train_false = dataset_train_false.shuffle(seed=42).select(range(len(dataset_train_true)))
    dataset["train"] = concatenate_datasets([dataset_train_true, dataset_train_false]).shuffle(seed=42)
    dataset_val_test = dataset["test"].train_test_split(train_size=0.5, seed=42)
    dataset["validation"], dataset["test"] = dataset_val_test["train"],dataset_val_test["test"]
    def get_prompt(example):
        example["text"] = f"Question: {example['prompt']}\nAnswer: {' '.join(example['completions'])}"
        return example
    dataset = dataset.map(get_prompt, remove_columns=["prompt", "completions"])
    return dataset


#====================
# Sequence regression
#====================

# Multilingual
def wmt_da_human_evaluation_src_mt() -> DatasetDict:
    dataset = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train")
    valid_lps = sorted(list(set(dataset["lp"]) & set(VALID_LPS)))
    dataset_train, dataset_val, dataset_test = [], [], []
    for lp in tqdm(valid_lps):
        subset = dataset.filter(lambda x: x["lp"] == lp)
        subset_train_val_test = subset.train_test_split(train_size=0.9, seed=42)
        subset_train, subset_val_test = subset_train_val_test["train"], subset_train_val_test["test"]
        subset_val_test = subset_val_test.train_test_split(train_size=0.5, seed=42)
        subset_val, subset_test = subset_val_test["train"], subset_val_test["test"]
        dataset_train.append(subset_train)
        dataset_val.append(subset_val)
        dataset_test.append(subset_test)
    dataset = DatasetDict({
        "train": concatenate_datasets(dataset_train),
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    def create_prompt(example):
        example["text"] = f"Source: {example['src']}\nTarget: {example['mt']}"
        return example
    dataset = dataset.map(create_prompt, load_from_cache_file=False)
    dataset = dataset.rename_column("score", "label").rename_column("lp", "subset")   
    dataset = dataset.remove_columns(["src", "mt", "ref", "raw", "annotators", "domain", "year"])
    return dataset

def wmt_da_human_evaluation_src_ref_mt() -> DatasetDict:
    dataset = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train")
    valid_lps = sorted(list(set(dataset["lp"]) & set(VALID_LPS)))
    dataset_train, dataset_val, dataset_test = [], [], []
    for lp in tqdm(valid_lps):
        subset = dataset.filter(lambda x: x["lp"] == lp)
        subset_train_val_test = subset.train_test_split(train_size=0.9, seed=42)
        subset_train, subset_val_test = subset_train_val_test["train"], subset_train_val_test["test"]
        subset_val_test = subset_val_test.train_test_split(train_size=0.5, seed=42)
        subset_val, subset_test = subset_val_test["train"], subset_val_test["test"]
        dataset_train.append(subset_train)
        dataset_val.append(subset_val)
        dataset_test.append(subset_test)
    dataset = DatasetDict({
        "train": concatenate_datasets(dataset_train),
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    def create_prompt(example):
        example["text"] = f"Source: {example['src']}\nReference: {example['ref']}\nTarget: {example['mt']}"
        return example
    dataset = dataset.map(create_prompt, load_from_cache_file=False)
    dataset = dataset.rename_column("score", "label").rename_column("lp", "subset")   
    dataset = dataset.remove_columns(["src", "mt", "ref", "raw", "annotators", "domain", "year"])  
    return dataset

def seahorse() -> DatasetDict:
    dataset = load_dataset("hgissbkh/seahorse")
    valid_langs = sorted(list(set(dataset["train"]["lang"]) & set(VALID_LANGS)))
    dataset = dataset.filter(lambda x: x["lang"] in valid_langs)
    def prepare_dataset(example):
        return {
            "text": f"Summary: {example['summary']}\nText: {example['text']}",
            "label": (
                example["comprehensible"] + 
                example["repetition"] +
                example["grammar"] +
                example["attribution"] +
                example["main_ideas"] +
                example["conciseness"]
            ) / 6,
            "subset": example["lang"],
        }
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=[
            "gem_id",
            "lang",
            "model",
            "summary", 
            "comprehensible", 
            "repetition", 
            "grammar", 
            "attribution", 
            "main_ideas", 
            "conciseness"
        ],
    )
    return dataset


#=====================
# Token classification
#=====================

# Multilingual
def ner() -> DatasetDict:
    dataset = load_dataset("hgissbkh/ner")
    dataset = dataset.rename_column("words", "tokens")
    dataset = dataset.rename_column("ner", "tags")
    dataset = dataset.rename_column("lang", "subset")
    return dataset


#==========
# Retrieval
#==========

# English
def msmarco_train() -> DatasetDict:
    dataset_dict = {}
    for dataset_fn in [msmarco]:
        dataset = dataset_fn()["train"]
        for k in dataset:
            dataset_dict[f"{dataset_fn.__class__.__name__}-{k}"] = dataset[k].select(range(1_000_000))
    dataset_dict = DatasetDict(dataset_dict)
    dataset = DatasetDict({"train": dataset_dict, "validation": None, "test": None})
    return dataset

def msmarco() -> DatasetDict:
    dataset = load_dataset("bclavie/msmarco-10m-triplets")
    dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.map(
        lambda x: {
            "anchor": f"Query: {x['anchor']}", 
            "positive": f"Document: {x['positive']}", 
            "negative": f"Document: {x['negative']}",
        }
    )
    dataset["train"], dataset_val_test = split_retrieval_dataset(dataset["train"], train_size=0.9, seed=42)
    dataset["validation"], dataset["test"] = split_retrieval_dataset(dataset_val_test, train_size=0.5, seed=42)
    dataset["train"] = DatasetDict({"en": dataset["train"]})
    return dataset

# Multilingual
def miracl() -> DatasetDict:
    config_names = get_dataset_config_names("sentence-transformers/miracl")
    valid_config_names = sorted(list(set(config_names) & set(f"{lang}-triplet" for lang in VALID_LANGS)))
    dataset_val, dataset_test = [], []
    for config_name in tqdm(valid_config_names):
        subset = load_dataset("sentence-transformers/miracl", name=config_name, split="train")
        lang = config_name.split("-")[0]
        subset = subset.add_column("subset", [lang] * len(subset))
        subset = subset.map(
            lambda x: {
                "anchor": f"Query: {x['anchor']}", 
                "positive": f"Document: {x['positive']}", 
            },
            remove_columns=["negative"],
        )
        subset_val, subset_test = split_retrieval_dataset(subset, train_size=0.5, seed=42)
        dataset_val.append(subset_val)
        dataset_test.append(subset_test)
    dataset = DatasetDict({
        "train": None,
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    return dataset

def mldr() -> DatasetDict:
    config_names = get_dataset_config_names("sentence-transformers/mldr")
    valid_config_names = sorted(list(set(config_names) & set(f"{lang}-triplet" for lang in VALID_LANGS)))
    dataset_val, dataset_test = [], []
    for config_name in tqdm(valid_config_names):
        subset = load_dataset("sentence-transformers/mldr", name=config_name, split="train")
        subset = subset.add_column("subset", [config_name.split("-")[0]] * len(subset))
        subset = subset.map(
            lambda x: {
                "anchor": f"Query: {x['anchor']}", 
                "positive": f"Document: {x['positive']}", 
            },
            remove_columns=["negative"],
        )
        subset_val, subset_test = split_retrieval_dataset(subset, train_size=0.5, seed=42)
        dataset_val.append(subset_val)
        dataset_test.append(subset_test)
    dataset = DatasetDict({
        "train": None, 
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    }) 
    return dataset

def wikipedia_retrieval_multilingual() -> DatasetDict:
    config_names = get_dataset_config_names("Samoed/WikipediaRetrievalMultilingual")
    dataset_langs = set(config_name.split("-")[0] for config_name in config_names)
    valid_langs = sorted(list(set(dataset_langs) & set(VALID_LANGS)))
    dataset_val, dataset_test = [], []
    for lang in valid_langs:
        queries = load_dataset("Samoed/WikipediaRetrievalMultilingual", name=f"{lang}-queries", split="test")
        corpus = load_dataset("Samoed/WikipediaRetrievalMultilingual", name=f"{lang}-corpus", split="test")
        qrels = load_dataset("Samoed/WikipediaRetrievalMultilingual", name=f"{lang}-qrels", split="test")
        queries = {x["_id"]: x["text"] for x in queries}
        corpus = {x["_id"]: x["text"]["text"] for x in corpus}
        subset = Dataset.from_list([
            {"anchor": queries[x["query-id"]], "positive": corpus[x["corpus-id"]]} 
            for x in qrels if x["score"] == 1
        ])
        subset = subset.add_column("subset", [lang] * len(subset))
        subset = subset.map(
            lambda x: {
                "anchor": f"Query: {x['anchor']}", 
                "positive": f"Document: {x['positive']}", 
            }
        )
        subset_val, subset_test = split_retrieval_dataset(subset, train_size=0.5, seed=42)
        dataset_val.append(subset_val)
        dataset_test.append(subset_test)
    dataset = DatasetDict({
        "train": None,
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    return dataset

def multilingual_cc_news() -> DatasetDict:
    config_names = get_dataset_config_names("hgissbkh/multilingual_cc_news")
    valid_config_names = sorted(list(set(config_names) & set(VALID_LANGS)))
    dataset_val, dataset_test = [], []
    for config_name in tqdm(valid_config_names):
        subset = load_dataset("hgissbkh/multilingual_cc_news", name=config_name, split="train")
        if "title" in subset.column_names:
            subset = subset.rename_column("title", "anchor")
        if "maintext" in subset.column_names:
            subset = subset.rename_column("maintext", "positive")
        subset = subset.add_column("subset", [config_name] * len(subset))   
        subset = subset.map(
            lambda x: {
                "anchor": f"Query: {x['anchor']}", 
                "positive": f"Document: {x['positive']}",
            }
        )
        subset, _ = split_retrieval_dataset(subset, train_size=0.1, seed=42)
        subset_val, subset_test = split_retrieval_dataset(subset, train_size=0.5, seed=42)
        dataset_val.append(subset_val)
        dataset_test.append(subset_test)
    dataset = DatasetDict({
        "train": None,
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    return dataset 

# Code
def codesearchnet() -> DatasetDict:
    dataset = load_dataset("sentence-transformers/codesearchnet")
    dataset = dataset.rename_column("comment", "anchor").rename_column("code", "positive")
    dataset = dataset.map(
        lambda x: {
            "anchor": f"Query: {x['anchor']}", 
            "positive": f"Document: {x['positive']}",
        }
    )
    dataset["train"], _ = split_retrieval_dataset(dataset["train"], train_size=0.1, seed=42)
    dataset["validation"], dataset["test"] = split_retrieval_dataset(dataset["train"], train_size=0.5, seed=42)
    dataset["train"] = None
    return dataset

def cqadupstack_mathematica() -> DatasetDict:
    queries = load_dataset("mteb/cqadupstack-mathematica", "queries", split="queries")
    corpus = load_dataset("mteb/cqadupstack-mathematica", "corpus", split="corpus")
    qrels = load_dataset("mteb/cqadupstack-mathematica", split="test")
    queries = {x["_id"]: x["text"] for x in queries}
    corpus = {x["_id"]: x["text"] for x in corpus}
    dataset = Dataset.from_list([
        {"anchor": queries[x["query-id"]], "positive": corpus[x["corpus-id"]]} 
        for x in qrels if x["score"] == 1
    ])
    dataset = dataset.map(
        lambda x: {
            "anchor": f"Query: {x['anchor']}", 
            "positive": f"Document: {x['positive']}",
        }
    )
    dataset_val, dataset_test = split_retrieval_dataset(dataset, train_size=0.5, seed=42)
    dataset = DatasetDict({
        "train": None,
        "validation": dataset_val, 
        "test": dataset_test
    })
    return dataset

# Math
def math_formula_retrieval() -> DatasetDict:
    dataset = load_dataset("hgissbkh/math_formula_retrieval_sampled")
    dataset = dataset.rename_column("formula", "anchor")
    dataset = dataset.rename_column("positives", "positive")
    dataset = dataset.remove_columns(["formula_name", "negatives"])
    return dataset

#====================
# Training EN only eval
#====================

def xnli_en_train() -> DatasetDict:
    subset = load_dataset("mteb/xnli", name="en", split="train")
    train_test = subset.train_test_split(test_size=8000, seed=42)
    test_val = train_test["test"].train_test_split(test_size=5000, seed=42)

    dataset = DatasetDict({
        "train": train_test["train"],
        "validation": test_val["train"],
        "test": test_val["test"],
    })
    dataset = dataset.rename_column("lang", "subset")
    def concat_premise_hypothesis(example):
        example["text"] = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
        return example
    dataset = dataset.map(concat_premise_hypothesis)
    dataset = dataset.remove_columns(["premise", "hypothesis"])
    return dataset

def ner_en_train() -> DatasetDict:
    dataset = load_dataset("hgissbkh/ner")
    dataset = dataset.rename_column("words", "tokens")
    dataset = dataset.rename_column("ner", "tags")
    dataset = dataset.rename_column("lang", "subset")

    subset = dataset["train"]
    train_test = subset.train_test_split(test_size=3000, seed=42)
    test_val = train_test["test"].train_test_split(test_size=2000, seed=42)

    dataset = DatasetDict({
        "train": train_test["train"],
        "validation": test_val["train"],
        "test": test_val["test"],
    })
    return dataset

def seahorse_en_train() -> DatasetDict:
    dataset = load_dataset("hgissbkh/seahorse")
    dataset = dataset.filter(lambda x: x["lang"] == 'en')

    def prepare_dataset(example):
        return {
            "text": f"Summary: {example['summary']}\nText: {example['text']}",
            "label": (
                example["comprehensible"] + 
                example["repetition"] +
                example["grammar"] +
                example["attribution"] +
                example["main_ideas"] +
                example["conciseness"]
            ) / 6,
            "subset": example["lang"],
        }
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=[
            "gem_id",
            "lang",
            "model",
            "summary", 
            "comprehensible", 
            "repetition", 
            "grammar", 
            "attribution", 
            "main_ideas", 
            "conciseness"
        ],
    )

    subset = dataset["train"]
    train_test = subset.train_test_split(test_size=8000, seed=42)
    test_val = train_test["test"].train_test_split(test_size=5000, seed=42)
    dataset = DatasetDict({
        "train": train_test["train"],
        "validation": test_val["train"],
        "test": test_val["test"],
    })
    return dataset

def msmarco_subset_en_train() -> DatasetDict:
    dataset = load_dataset("bclavie/msmarco-10m-triplets")
    dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.map(
        lambda x: {
            "anchor": f"Query: {x['anchor']}",
            "positive": f"Document: {x['positive']}",
            "negative": f"Document: {x['negative']}",
        }
    )
    dataset["train"] = dataset["train"].select(range(100000))

    dataset["train"], dataset_val_test = split_retrieval_dataset(dataset["train"], train_size=0.9, seed=42)
    dataset["validation"], dataset["test"] = split_retrieval_dataset(dataset_val_test, train_size=0.5, seed=42)
    dataset["train"] = DatasetDict({"en": dataset["train"]})
    return dataset

def miracl_en_val_as_test() -> DatasetDict:

    config_name = "en-triplet"

    dataset_val, dataset_test = [], []
    subset = load_dataset("sentence-transformers/miracl", name=config_name, split="train")
    lang = config_name.split("-")[0]

    subset = subset.add_column("subset", [lang] * len(subset))
    subset = subset.map(
        lambda x: {
            "anchor": f"Query: {x['anchor']}", 
            "positive": f"Document: {x['positive']}", 
        },
        remove_columns=["negative"],
    )
    subset_val, subset_test = split_retrieval_dataset(subset, train_size=0.5, seed=42)
    
    dataset_val.append(subset_val)
    dataset_test.append(subset_test)
    dataset = DatasetDict({
        "train": None,
        "validation": None,
        "test": concatenate_datasets(dataset_val),
    })
    return dataset

def amazon_massive_intent_en_train() -> DatasetDict:
    config_name = "en"
    dataset_train, dataset_val, dataset_test = [], [], []
    
    subset = load_dataset("mteb/amazon_massive_intent", name=config_name, split="train")
    subset = subset.map(lambda _: {"lang": "zh"}) if config_name == "zh-CN" else subset
    
    split_result = subset.train_test_split(test_size=0.1, seed=42)
    subset_train = split_result["train"]
    subset_val_test = split_result["test"]
    
    val_test_split = subset_val_test.train_test_split(test_size=0.5, seed=42)
    subset_validation = val_test_split["train"]
    subset_test = val_test_split["test"]

    dataset_train.append(subset_train)
    dataset_val.append(subset_validation)
    dataset_test.append(subset_test)

    
    dataset = DatasetDict({
        "train": concatenate_datasets(dataset_train),
        "validation": concatenate_datasets(dataset_val),
        "test": concatenate_datasets(dataset_test),
    })
    dataset = dataset.rename_column("lang", "subset")
    labels = sorted(list(
        set(dataset["train"]["label"]) |
        set(dataset["validation"]["label"]) |
        set(dataset["test"]["label"])
    ))
    def get_label(example):
        example["label"] = labels.index(example["label"])
        return example
    dataset = dataset.map(get_label)
    dataset = dataset.remove_columns(["id", "label_text"])
    return dataset
