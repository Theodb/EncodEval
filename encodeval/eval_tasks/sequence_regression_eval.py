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
    def train(self) -> None:
        """
        Adapts the model for a given training configuration by tokenizing datasets, setting up the trainer,
        and performing training.
        """
        # Tokenize training dataset
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
        
        # Load and tokenize validation dataset
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

        # Train model
        print("Training model")
        torch.set_default_dtype(self.model.dtype)
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
        if "chosen" in self.dataset[split].column_names and "rejected" in self.dataset[split].column_names:
            return self.evaluate_on_preferences(split)
        else:
            return self.evaluate_on_scores(split)

    def evaluate_on_scores(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        # Tokenize evaluation dataset
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
        # Tokenize evaluation dataset
        print(f"Tokenizing {split} dataset")
        tokenization_fn = self.get_tokenization_fn()
        eval_dataset = self.dataset[split]
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

        # Set up data collator
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # Get data loaders
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

        # Evaluate model
        self.model.eval()
        chosen_scores, rejected_scores = [], []

        with torch.no_grad():
            for batch in tqdm(chosen_dataloader, desc="Computing chosen scores"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                chosen_scores.append(self.model(**batch).logits.cpu())
            for batch in tqdm(rejected_dataloader, desc="Computing rejected scores"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                rejected_scores.append(self.model(**batch).logits.cpu())
                
        # Compute and return metrics
        chosen_scores, rejected_scores = torch.cat(chosen_scores), torch.cat(rejected_scores) 
        results = {
            "average": self.compute_preference_metrics((chosen_scores, rejected_scores)),
            "per_instance": self.compute_preference_metrics_instances((chosen_scores, rejected_scores)),
        }
        if subsets is not None:
            results["per_instance"]["subset"] = subsets
        
        return results        

    def get_tokenization_fn(self):
        if self.model.__class__.__name__.startswith("Optimus"):
            return self.optimus_tokenization_fn
        else:
            return self.vanilla_tokenization_fn
    
    def vanilla_tokenization_fn(self, examples: Dict, col_name: str):
        return self.tokenizer(
            examples[col_name], 
            truncation=True, 
            max_length=self.max_length,
        )
    
    def optimus_tokenization_fn(self, examples: Dict, col_name: str): 
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
        predictions, labels = eval_pred
        predictions = predictions.flatten().tolist()
        labels = labels.flatten().tolist()
        return {"spearman": spearmanr(predictions, labels)[0]}
    
    def compute_preference_metrics(self, eval_pred):
        chosen_scores, rejected_scores = eval_pred
        chosen_scores = chosen_scores.flatten()
        rejected_scores = rejected_scores.flatten()
        return {"accuracy": ((chosen_scores > rejected_scores) * 1.).mean().item()}
    
    def compute_preference_metrics_instances(self, eval_pred):
        chosen_scores, rejected_scores = eval_pred
        chosen_scores = chosen_scores.flatten()
        rejected_scores = rejected_scores.flatten()
        return {"accuracy": ((chosen_scores > rejected_scores) * 1).tolist()}
    