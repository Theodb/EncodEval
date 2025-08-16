import os
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer

from .abstract_eval import AbstractEval

from encodeval.datasets import apply_data_percentage

class SequenceClassificationEval(AbstractEval):
    """
    Evaluation class for sequence classification models.

    Handles tokenization, training, and evaluation for classification tasks,
    leveraging the HuggingFace Trainer API.
    """

    def train(self) -> None:
        """
        Fine-tunes the sequence classification model on the training dataset.

        If evaluation is enabled, also uses the validation split during training.
        Saves the model to the output directory upon completion if `do_predict` is False.
        """
        print("Tokenizing training dataset")
        tokenization_fn = self.get_tokenization_fn()

        # Tokenize and retain only required columns
        # Get train dataset and apply percentage if needed
        train_dataset = self.dataset["train"]
        data_percentage = os.environ.get("DATA_PERCENTAGE")
        if data_percentage and data_percentage.isdigit() and 0 < int(data_percentage) < 100:
            print(f"Using {data_percentage}% of training data")
            train_dataset = train_dataset.select(range(len(train_dataset) * int(data_percentage) // 100))
        
        train_dataset = train_dataset.map(tokenization_fn, batched=True, load_from_cache_file=False)        
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

        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # Initialize HuggingFace Trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=self.callbacks,
            args=self.tr_args,
        )

        print("Training model")
        trainer.train()

        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluates the model on the validation set.

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Average and per-instance evaluation metrics.
        """
        print("Evaluating on validation dataset")
        return self.evaluate("validation")

    def test(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluates the model on the test set.

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Average and per-instance evaluation metrics.
        """
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split: str) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        General evaluation logic shared by validation and test routines.

        Args:
            split (str): Dataset split to evaluate on ('validation' or 'test').

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Dictionary of classification metrics.
        """
        print(f"Tokenizing {split} dataset")
        tokenization_fn = self.get_tokenization_fn()

        eval_dataset = self.dataset[split].map(tokenization_fn, batched=True, load_from_cache_file=False)
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

        metrics_per_instance = self.compute_metrics_instances(
            (torch.cat(predictions), torch.tensor(eval_dataset["label"]))
        )

        if subsets is not None:
            metrics_per_instance["subset"] = subsets

        return metrics_per_instance


    def get_tokenization_fn(self):
        """
        Selects the appropriate tokenization strategy based on the model type.

        Returns:
            Callable: Tokenization function.
        """
        if self.model.__class__.__name__.startswith("EuroBert"):
            return self.eurobert_tokenization_fn
        else:
            return self.standard_tokenization_fn

    def standard_tokenization_fn(self, examples: Dict) -> Dict:
        """
        Applies standard tokenization to input examples.

        Args:
            examples (Dict): Dictionary containing a 'text' field.

        Returns:
            Dict: Tokenized output.
        """
        # Ensure we get proper tokenization output with input_ids
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # Return lists for batched processing
        )
        
        # Ensure all required fields are present
        if "input_ids" not in tokenized:
            raise ValueError("Tokenizer did not return input_ids")
            
        return tokenized

    def eurobert_tokenization_fn(self, examples: Dict) -> Dict:
        """
        Tokenization function tailored for EuroBERT models.

        Adds EOS (and optionally BOS) tokens based on classification pooling strategy.

        Args:
            examples (Dict): Dictionary with a 'text' field.

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

    def compute_metrics_instances(self, eval_pred) -> Dict[str, List[float]]:
        """
        Computes per-instance classification accuracy.

        Args:
            eval_pred: A tuple containing model predictions and ground-truth labels.

        Returns:
            Dict[str, List[float]]: Dictionary of per-instance accuracy values.
        """
        predictions, labels = eval_pred
        return {"accuracy": ((predictions.argmax(1) == labels) * 1).tolist()}
