# EncodEval: Evaluating Pretrained Encoders Across NLP Tasks with Confidence-Aware Rankings

## Overview

**EncodEval** is a lightweight evaluation framework designed to benchmark general-purpose pre-trained encoders on a diverse set of downstream NLP tasks:

- Sequence Classification (SC)  
- Sequence Regression (SR)  
- Token Classification (TC)  
- Information Retrieval (IR)

This repository was used for evaluation in the paper  
[EuroBERT: Scaling Multilingual Encoders for European Languages](https://arxiv.org/abs/2503.05500). If you're interested in training the EuroBERT model, please refer to the [EuroBERT repository](https://github.com/Nicolas-BZRD/EuroBERT).


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


## Running Evaluations

To run a task evaluation from the command line:

```bash
python main.py \ 
    --config_file <config_file_path> \ 
    --model_path <model_path>
```

This will generate a `results.json` file with instance-level scores. For analyzing and comparing system-level results, see the [System Evaluation](#system-evaluation) section below.


## Task Evaluation Modules

Task-specific evaluation logic is implemented in [encodeval/eval_tasks/](encodeval/eval_tasks/). These modules handle both fine-tuning and evaluation.

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

Dataset loading and preprocessing are managed in [encodeval/datasets.py](encodeval/datasets.py). To add a new dataset, implement the loading logic in this file.

Example (loading the XNLI dataset):

```python
from encodeval.datasets import xnli
dataset = xnli()
```


## Configuration Files

Examples of configuration files are available in the [configs/](configs/) folder.


## System evaluation

To compare and rank models on a given task, use the `get_results` function. This will:

* Run hyperparameter search on the validation set (if available and if multiple configurations are provided — see [results/toy/](results/toy/) for an example). Otherwise, it simply loads the existing results.
* Compute average scores across languages (`average_scores`)
* Perform statistical testing and calculate Borda counts for rankings (`system_ranking`)

Example usage:

```python
from encodeval.system_ranking import get_results

average_scores, system_ranking = get_results(
    base_path="./results/toy",
    models=["model1", "model2", "model3"], 
    task_type="SC",
    dataset="dataset_sc", 
    valid_langs=["en", "fr"],
)

print(average_scores)
print(system_ranking)
```

*Note: Rankings are based on statistical significance at the 95% confidence level.*


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