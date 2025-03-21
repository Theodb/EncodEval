# EncodEval: Benchmarking Pre-trained Encoders Across Tasks

## Overview

**EncodEval** is a lightweight evaluation framework designed to benchmark general-purpose pre-trained encoders on a diverse set of downstream NLP tasks:

- Sequence Classification (SC)  
- Sequence Regression (SR)  
- Token Classification (TC)  
- Information Retrieval (IR)

This repository was used for evaluation in the paper  
[**EuroBERT: Scaling Multilingual Encoders for European Languages**](https://arxiv.org/abs/2503.05500).


## Installation

To install EncodEval directly via pip:

```bash
pip install git+https://github.com/hgissbkh/EncodEval.git
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/hgissbkh/EncodEval.git
cd EncodEval
pip install -e .
```


## Running evaluations

To run a task evaluation from the command line:

```bash
python main.py \ 
    --config_file <config_file_path> \ 
    --model_path <model_path>
```


## Task evaluation files

Task-specific evaluation logic is implemented in [`encodeval/eval_tasks/`](encodeval/eval_tasks/). These modules handle both fine-tuning and evaluation.

Example usage in Python:
```python
from encodeval.eval_tasks import EvalConfig, SequenceClassificationEval

config_file = "./configs/sequence_classification_tasks/example.yaml"
eval_config: EvalConfig = configue.load(
    config_file,  
    sub_path="eval_config",
)
evaluator = SequenceClassificationEval(eval_config)
evaluator.train() # Fine-tune on the target task
evaluator.validate() # Evaluate on the validation set
evaluator.test() # Evaluate on the test set
```


## Datasets

Dataset loading and preprocessing are managed in [`encodeval/datasets.py`](encodeval/datasets.py). To add a new dataset, implement the loading logic in this file.

Example (loading the XNLI dataset):

```python
from encodeval.datasets import xnli
dataset = xnli()
```


## Configuration files

Examples of configuration files are available in the [configs/](configs/) folder.


## Citation

If you use this framework in your research, please consider citing:

```bibtex
@misc{boizard2025eurobertscalingmultilingualencoders,
  title={EuroBERT: Scaling Multilingual Encoders for European Languages}, 
  author={Nicolas Boizard and Hippolyte Gisserot-Boukhlef and Duarte M. Alves and André Martins and Ayoub Hammal and Caio Corro and Céline Hudelot and Emmanuel Malherbe and Etienne Malaboeuf and Fanny Jourdan and Gabriel Hautreux and João Alves and Kevin El-Haddad and Manuel Faysse and Maxime Peyrard and Nuno M. Guerreiro and Patrick Fernandes and Ricardo Rei and Pierre Colombo},
  year={2025},
  eprint={2503.05500},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2503.05500}
}
```