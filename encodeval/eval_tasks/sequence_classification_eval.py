from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer, EarlyStoppingCallback

from .abstract_eval import AbstractEval


class SequenceClassificationEval(AbstractEval):
    def train(self) -> None:
        """
        Adapts the model for a given training configuration by tokenizing datasets, setting up the trainer,
        and performing training.
        """
        # Tokenize training dataset
        print("Tokenizing training dataset")
        tokenization_fn = self.get_tokenization_fn()
        train_dataset = self.dataset["train"].map(tokenization_fn, batched=True, load_from_cache_file=False)        
        train_dataset = train_dataset.remove_columns(
            [f for f in train_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
        )
        
        # Load and tokenize validation dataset
        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"].map(tokenization_fn, batched=True, load_from_cache_file=False)
            val_dataset = val_dataset.remove_columns(
                [f for f in val_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
            )
        else:
            val_dataset = None

        # Print training args
        print("==== Training Arguments ====")
        print(self.tr_args)
        print("=============================")

        # Set up collator
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # Set up Trainer instance
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

        # Train the model
        print("Training model")
        trainer.train()

        # Save model if not evaluated
        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        print("Evaluating on validation dataset")
        return self.evaluate("validation")
    
    def test(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        print("Evaluating on test dataset")
        return self.evaluate("test")
    
    def evaluate(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Returns:
            Dict[str, Dict[str, Union[float, List]]]: A dictionary of computed metrics (e.g., accuracy), averaged and for each instance.
        """
        # Tokenize evaluation dataset
        print(f"Tokenizing {split} dataset")
        tokenization_fn = self.get_tokenization_fn()
        eval_dataset = self.dataset[split].map(tokenization_fn, batched=True, load_from_cache_file=False)
        subsets = eval_dataset["subset"] if "subset" in eval_dataset.column_names else None
        eval_dataset = eval_dataset.remove_columns(
            [f for f in eval_dataset.features if f not in ["input_ids", "attention_mask", "label"]]
        )

        # Set up data collator
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # Get data loader
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

        # Evaluate model
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                predictions.append(output.logits.cpu())

        # Compute and return metrics
        metrics_per_instance = self.compute_metrics_instances(
            (torch.cat(predictions), torch.tensor(eval_dataset["label"]))
        )
        if subsets is not None:
            metrics_per_instance["subset"] = subsets

        return {
            "average": {k: np.mean(v) for k, v in metrics_per_instance.items() if k != "subset"},
            "per_instance": metrics_per_instance,
        }
    
    def get_tokenization_fn(self):
        if self.model.__class__.__name__.startswith("Optimus"):
            return self.optimus_tokenization_fn
        else:
            return self.vanilla_tokenization_fn

    def vanilla_tokenization_fn(self, examples: Dict):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
        )

    def optimus_tokenization_fn(self, examples: Dict): 
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
        predictions, labels = eval_pred
        return {"accuracy": (predictions.argmax(1) == labels).mean().item()}
    
    def compute_metrics_instances(self, eval_pred):
        predictions, labels = eval_pred
        return {"accuracy": ((predictions.argmax(1) == labels) * 1).tolist()}
