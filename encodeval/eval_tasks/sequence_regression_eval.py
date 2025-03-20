from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer

from .abstract_eval import AbstractEval


class SequenceRegressionEval(AbstractEval):
    """
    Evaluation class for sequence regression models.

    Supports both scalar regression (e.g., score prediction) and
    preference-based tasks. Provides tokenization, training, and evaluation routines.
    """

    def train(self) -> None:
        """
        Fine-tunes the regression model using the training dataset.

        Applies tokenization, initializes the HuggingFace Trainer, and performs training.
        Saves the model to the output directory if evaluation is not performed.
        """
        print("Tokenizing training dataset")
        tokenization_fn = self.get_tokenization_fn()

        train_dataset = self.dataset["train"].map(
            lambda examples: tokenization_fn(examples, "text"), 
            batched=True, 
            load_from_cache_file=False,
        )
        train_dataset = train_dataset.remove_columns(
            [f for f in train_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
        )

        # Tokenize validation dataset if evaluation is enabled
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
        """
        Evaluates the model on the validation split.

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Per-instance predictions and labels.
        """
        print("Evaluating on validation dataset")
        return self.evaluate("validation")
    
    def test(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluates the model on the test split.

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Per-instance predictions and labels.
        """
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluates the model on scalar regression tasks (e.g., rating prediction).

        Tokenizes the dataset, computes model predictions, and returns predictions and labels.

        Args:
            split (str): Dataset split to evaluate ('validation' or 'test').

        Returns:
            Dict[str, Dict[str, Union[float, List[float]]]]: Evaluation outputs including predictions and labels.
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
            "prediction": predictions.flatten().tolist(),
            "label": eval_dataset["label"],
        }

        if subsets is not None:
            results["per_instance"] = {"subset": subsets}

        return results

    def get_tokenization_fn(self):
        """
        Selects the appropriate tokenization strategy based on model type.

        Returns:
            Callable: Tokenization function.
        """
        if self.model.__class__.__name__.startswith("EuroBert"):
            return self.eurobert_tokenization_fn
        else:
            return self.standard_tokenization_fn

    def standard_tokenization_fn(self, examples: Dict, col_name: str):
        """
        Applies standard tokenization to the given column.

        Args:
            examples (Dict): Dictionary containing input sequences.
            col_name (str): Name of the column to tokenize (e.g., "text").

        Returns:
            Dict: Tokenized output.
        """
        return self.tokenizer(
            examples[col_name], 
            truncation=True, 
            max_length=self.max_length,
        )

    def eurobert_tokenization_fn(self, examples: Dict, col_name: str): 
        """
        Tokenization function customized for EuroBERT models.

        Adds special tokens (EOS, optionally BOS) depending on the classification pooling strategy.

        Args:
            examples (Dict): Dictionary containing input sequences.
            col_name (str): Column name to tokenize.

        Returns:
            Dict: Tokenized output with appropriate special tokens.
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
