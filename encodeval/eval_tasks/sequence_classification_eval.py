from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer

from .abstract_eval import AbstractEval


class SequenceClassificationEval(AbstractEval):
    """
    A class for training and evaluating sequence classification models.
    It supports tokenizing datasets, training with a Trainer instance, 
    and evaluating performance using accuracy and other metrics.
    """

    def train(self) -> None:
        """
        Train the sequence classification model. 
        This method tokenizes datasets, initializes the Trainer, and executes training.
        If `do_predict` is False, the trained model is saved to `output_dir`.
        """
        print("Tokenizing training dataset")
        tokenization_fn = self.get_tokenization_fn()
        
        # Tokenize and retain only necessary columns
        train_dataset = self.dataset["train"].map(tokenization_fn, batched=True, load_from_cache_file=False)        
        train_dataset = train_dataset.remove_columns(
            [f for f in train_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
        )
        
        # Load and tokenize validation dataset if evaluation is enabled
        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"].map(tokenization_fn, batched=True, load_from_cache_file=False)
            val_dataset = val_dataset.remove_columns(
                [f for f in val_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
            )
        else:
            val_dataset = None

        print("==== Training Arguments ====")
        print(self.tr_args)
        print("=============================")

        # Define data collator for padding
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,            
            args=self.tr_args,
        )

        print("Training model")
        trainer.train()

        # Save model after training if evaluation is not performed
        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Dictionary containing average and per-instance metrics.
        """
        print("Evaluating on validation dataset")
        return self.evaluate("validation")
    
    def test(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluate the model on the test dataset.

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Dictionary containing average and per-instance metrics.
        """
        print("Evaluating on test dataset")
        return self.evaluate("test")
    
    def evaluate(self, split: str) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        General evaluation function for computing model performance.

        Args:
            split (str): Dataset split to evaluate on ('validation' or 'test').

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Dictionary containing 
            average and per-instance classification metrics.
        """
        print(f"Tokenizing {split} dataset")
        tokenization_fn = self.get_tokenization_fn()

        # Tokenize and extract relevant columns
        eval_dataset = self.dataset[split].map(tokenization_fn, batched=True, load_from_cache_file=False)
        subsets = eval_dataset["subset"] if "subset" in eval_dataset.column_names else None
        eval_dataset = eval_dataset.remove_columns(
            [f for f in eval_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
        )

        # Prepare data loader with padding
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

        # Perform evaluation
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                predictions.append(output.logits.cpu())

        # Compute per-instance metrics
        metrics_per_instance = self.compute_metrics_instances(
            (torch.cat(predictions), torch.tensor(eval_dataset["label"]))
        )
        
        # If dataset contains subsets, add them to per-instance metrics
        if subsets is not None:
            metrics_per_instance["subset"] = subsets

        return {
            "average": {k: np.mean(v) for k, v in metrics_per_instance.items() if k != "subset"},
            "per_instance": metrics_per_instance,
        }
    
    def get_tokenization_fn(self):
        """
        Selects the appropriate tokenization function based on the model type.

        Returns:
            Callable: The tokenization function.
        """
        if self.model.__class__.__name__.startswith("EuroBert"):
            return self.eurobert_tokenization_fn
        else:
            return self.standard_tokenization_fn

    def standard_tokenization_fn(self, examples: Dict):
        """
        Standard tokenization function.

        Args:
            examples (Dict): Dictionary containing text examples.

        Returns:
            Dict: Tokenized examples.
        """
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
        )

    def eurobert_tokenization_fn(self, examples: Dict): 
        """
        Tokenization function for EuroBERT models (EOS, no BOS).

        Args:
            examples (Dict): Dictionary containing text examples.

        Returns:
            Dict: Tokenized examples with appropriate special tokens.
        """
        if self.model.clf_pooling == "bos":        
            texts = [self.tokenizer.bos_token + text + self.tokenizer.eos_token for text in examples["text"]]
        elif self.model.clf_pooling in ["mean", "late"]:
            texts = [text + self.tokenizer.eos_token for text in examples["text"]]

        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
    
    def compute_metrics(self, eval_pred):
        """
        Compute accuracy for model predictions.

        Args:
            eval_pred: Tuple of (predictions, labels).

        Returns:
            Dict[str, float]: Dictionary containing accuracy metric.
        """
        predictions, labels = eval_pred
        return {"accuracy": (predictions.argmax(1) == labels).mean().item()}
    
    def compute_metrics_instances(self, eval_pred):
        """
        Compute per-instance accuracy.

        Args:
            eval_pred: Tuple of (predictions, labels).

        Returns:
            Dict[str, List[float]]: Dictionary containing per-instance accuracy values.
        """
        predictions, labels = eval_pred
        return {"accuracy": ((predictions.argmax(1) == labels) * 1).tolist()}
