from typing import Dict, List, Union

from datasets import Dataset
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer

from .abstract_eval import AbstractEval


class SequenceRegressionEval(AbstractEval):
    """
    A class for training and evaluating sequence regression models.
    Supports both score-based regression (e.g., rating prediction)
    and pairwise preference evaluation (e.g., ranking tasks).
    """

    def train(self) -> None:
        """
        Train the regression model on the training set.
        Applies tokenization, initializes a HuggingFace Trainer,
        and performs training. Saves the model if not evaluating.
        """
        print("Tokenizing training dataset")
        tokenization_fn = self.get_tokenization_fn()

        # Tokenize and clean training dataset
        train_dataset = self.dataset["train"].map(
            lambda examples: tokenization_fn(examples, "text"), 
            batched=True, 
            load_from_cache_file=False,
        )
        train_dataset = train_dataset.remove_columns(
            [f for f in train_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
        )

        # Tokenize and clean validation dataset if evaluation is enabled
        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"].map(
                lambda examples: tokenization_fn(examples, "text"), 
                batched=True, 
                load_from_cache_file=False,
            )
            val_dataset = val_dataset.remove_columns(
                [f for f in val_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
            )
        else:
            val_dataset = None

        print("==== Training Arguments ====")
        print(self.tr_args)
        print("=============================")

        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

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
        torch.set_default_dtype(self.model.dtype)
        trainer.train()

        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """Run evaluation on the validation set."""
        print("Evaluating on validation dataset")
        return self.evaluate("validation")
    
    def test(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """Run evaluation on the test set."""
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        General evaluation wrapper for regression tasks.

        Automatically chooses between score-based and preference-based
        evaluation based on dataset fields.

        Args:
            split (str): Dataset split ("validation" or "test").

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Evaluation results.
        """     
        if "chosen" in self.dataset[split].column_names and "rejected" in self.dataset[split].column_names:
            return self.evaluate_on_preferences(split)
        else:
            return self.evaluate_on_scores(split)

    def evaluate_on_scores(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluate the model on datasets with scalar scores (e.g., sentiment regression).

        Computes average Spearman correlation and per-instance predictions/labels.
        """
        print(f"Tokenizing {split} dataset")
        tokenization_fn = self.get_tokenization_fn()

        eval_dataset = self.dataset[split].map(
            lambda examples: tokenization_fn(examples, "text"), 
            batched=True, 
            load_from_cache_file=False,
        )

        subsets = eval_dataset["subset"] if "subset" in eval_dataset.column_names else None
        eval_dataset = eval_dataset.remove_columns(
            [f for f in eval_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
        )

        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                predictions.append(output.logits.cpu())

        predictions = torch.cat(predictions)

        results = {
            "average": self.compute_metrics((predictions, torch.tensor(eval_dataset["label"]).view(-1,1))),
            "per_instance": {
                "prediction": predictions.flatten().tolist(),
                "label": eval_dataset["label"],
            },
        }
        if subsets is not None:
            results["per_instance"]["subset"] = subsets

        return results

    def evaluate_on_preferences(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluate the model on preference-labeled datasets using pairwise comparisons.

        For each (chosen, rejected) pair, checks whether model assigns higher score to the chosen.
        """
        print(f"Tokenizing {split} dataset")
        tokenization_fn = self.get_tokenization_fn()
        eval_dataset = self.dataset[split]

        # Tokenize chosen and rejected examples separately
        chosen_dataset = eval_dataset.map(
            lambda examples: tokenization_fn(examples, "chosen"),
            batched=True, 
            load_from_cache_file=False,
            remove_columns=[f for f in eval_dataset.features if f not in ["input_ids", "attention_mask"]],
        )
        rejected_dataset = eval_dataset.map(
            lambda examples: tokenization_fn(examples, "rejected"),
            batched=True, 
            load_from_cache_file=False,
            remove_columns=[f for f in eval_dataset.features if f not in ["input_ids", "attention_mask"]],
        )

        subsets = eval_dataset["subset"] if "subset" in eval_dataset.column_names else None
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # Loaders for chosen/rejected sequences
        chosen_dataloader = DataLoader(
            chosen_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )
        rejected_dataloader = DataLoader(
            rejected_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

        self.model.eval()
        chosen_scores, rejected_scores = [], []

        with torch.no_grad():
            for batch in tqdm(chosen_dataloader, desc="Computing chosen scores"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                chosen_scores.append(self.model(**batch).logits.cpu())

            for batch in tqdm(rejected_dataloader, desc="Computing rejected scores"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                rejected_scores.append(self.model(**batch).logits.cpu())

        chosen_scores, rejected_scores = torch.cat(chosen_scores), torch.cat(rejected_scores)

        results = {
            "average": self.compute_preference_metrics((chosen_scores, rejected_scores)),
            "per_instance": self.compute_preference_metrics_instances((chosen_scores, rejected_scores)),
        }
        if subsets is not None:
            results["per_instance"]["subset"] = subsets

        return results

    def get_tokenization_fn(self):
        """
        Return the appropriate tokenization function based on model type.
        """
        if self.model.__class__.__name__.startswith("EuroBert"):
            return self.eurobert_tokenization_fn
        else:
            return self.standard_tokenization_fn

    def standard_tokenization_fn(self, examples: Dict, col_name: str):
        """
        Standard tokenization function.

        Args:
            examples (Dict): Example dictionary with sequences.
            col_name (str): Column to tokenize ("text", "chosen", or "rejected").
        """
        return self.tokenizer(
            examples[col_name], 
            truncation=True, 
            max_length=self.max_length,
        )

    def eurobert_tokenization_fn(self, examples: Dict, col_name: str): 
        """
        Tokenization function for EuroBERT models (EOS, no BOS).

        Args:
            examples (Dict): Input text dictionary.
            col_name (str): Text field to tokenize.

        Returns:
            Dict: Tokenized text dictionary.
        """
        if self.model.clf_pooling == "bos":
            texts = [self.tokenizer.bos_token + text + self.tokenizer.eos_token for text in examples[col_name]]
        elif self.model.clf_pooling in ["mean", "late"]:
            texts = [text + self.tokenizer.eos_token for text in examples[col_name]]

        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

    def compute_metrics(self, eval_pred):
        """
        Compute Spearman correlation between predictions and gold labels.

        Args:
            eval_pred (Tuple): (predictions, labels)

        Returns:
            Dict[str, float]: Spearman correlation.
        """
        predictions, labels = eval_pred
        predictions = predictions.flatten().tolist()
        labels = labels.flatten().tolist()
        return {"spearman": spearmanr(predictions, labels)[0]}

    def compute_preference_metrics(self, eval_pred):
        """
        Compute accuracy based on pairwise preference (chosen > rejected).

        Args:
            eval_pred (Tuple): (chosen_scores, rejected_scores)

        Returns:
            Dict[str, float]: Accuracy over pairs.
        """
        chosen_scores, rejected_scores = eval_pred
        chosen_scores = chosen_scores.flatten()
        rejected_scores = rejected_scores.flatten()
        return {"accuracy": ((chosen_scores > rejected_scores) * 1.).mean().item()}

    def compute_preference_metrics_instances(self, eval_pred):
        """
        Compute per-instance preference accuracy.

        Args:
            eval_pred (Tuple): (chosen_scores, rejected_scores)

        Returns:
            Dict[str, List[float]]: Binary accuracy for each pair.
        """
        chosen_scores, rejected_scores = eval_pred
        chosen_scores = chosen_scores.flatten()
        rejected_scores = rejected_scores.flatten()
        return {"accuracy": ((chosen_scores > rejected_scores) * 1).tolist()}
