

from typing import Dict, List, Union

import numpy as np
import torch
# from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification, Trainer

from .abstract_eval import AbstractEval


class TokenClassificationEval(AbstractEval):
    def train(self) -> None:
        """
        Adapts the model for a given training configuration by tokenizing datasets,
        setting up the trainer, and performing training.
        """
        # Tokenize training dataset
        print("Tokenizing training dataset")
        tokenization_fn = self.get_tokenization_fn()
        train_dataset = self.dataset["train"].map(tokenization_fn, batched=True, load_from_cache_file=False)
        label_list = train_dataset.features["tags"].feature.names
        train_dataset = train_dataset.remove_columns(
            [f for f in train_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
        )

        # Load and tokenize validation dataset
        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"]
            val_dataset = val_dataset.map(tokenization_fn, batched=True, load_from_cache_file=False)
            val_dataset = val_dataset.remove_columns(
                [f for f in val_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
            )
        else:
            val_dataset = None

        # Print training args
        print("==== Training Arguments ====")
        print(self.tr_args)
        print("=============================")

        # Set up collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        # Set up Trainer instance
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
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
        label_list = eval_dataset.features["tags"].feature.names
        subsets = eval_dataset["subset"] if "subset" in eval_dataset.column_names else None
        token_ids = eval_dataset["token_ids"]
        eval_dataset = eval_dataset.remove_columns(
            [f for f in eval_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
        )

        # Set up collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        # Get data loader
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

        # Evaluate model
        self.model.eval()
        predictions, labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                logits = output.logits.cpu()
                preds = logits[0].argmax(2) if isinstance(logits, tuple) else logits.argmax(2)
                predictions += preds.tolist()
                labels += batch["labels"].cpu().tolist()

        predictions_subtoken, labels_subtoken, predictions_token, labels_token = (
            self.sanitize_predictions_labels(predictions, labels, token_ids)
        )
        metrics_per_instance = {
            "prediction_token": predictions_token, 
            "labels_token": labels_token,
            "prediction_subtoken": predictions_subtoken, 
            "labels_subtoken": labels_subtoken,
        }
        if subsets is not None:
            metrics_per_instance["subset"] = subsets
        
        return {
            "average": None, ### TODO: ADD DATASET LEVEL METRICS
            "per_instance": metrics_per_instance,
        }

    def get_tokenization_fn(self):
        if self.model.__class__.__name__.startswith("Optimus"):
            return self.optimus_tokenization_fn
        else:
            return self.vanilla_tokenization_fn

    def vanilla_tokenization_fn(self, examples):
        sentences = [" ".join(tokens) for tokens in examples["tokens"]]
        tokenized_inputs = self.tokenizer(
            sentences,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        aligned_labels = []
        token_ids = []

        for offsets, tokens, tags in zip(
            tokenized_inputs["offset_mapping"], examples["tokens"], examples["tags"]
        ):
            label_ids = []
            token_id_per_subtoken = []
            word_idx = 0
            char_pos = 0
            current_word = tokens[word_idx]
            current_label = tags[word_idx]

            for offset in offsets:
                if offset == (0, 0):
                    label_ids.append(-100)
                    token_id_per_subtoken.append(-100)
                    continue

                while offset[0] >= char_pos + len(current_word):
                    char_pos += len(current_word) + 1
                    word_idx += 1
                    if word_idx >= len(tokens):
                        break
                    current_word = tokens[word_idx]
                    current_label = tags[word_idx]

                if word_idx < len(tags):
                    label_ids.append(current_label)
                    token_id_per_subtoken.append(word_idx)
                else:
                    label_ids.append(-100)
                    token_id_per_subtoken.append(-100)

            aligned_labels.append(label_ids)
            token_ids.append(token_id_per_subtoken)

        tokenized_inputs["labels"] = aligned_labels
        tokenized_inputs["token_ids"] = token_ids
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs
    
    def optimus_tokenization_fn(self, examples):
        sentences = [" ".join(tokens) for tokens in examples["tokens"]]
        tokenized_inputs = self.tokenizer(
            [sentence + self.tokenizer.eos_token for sentence in sentences],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        aligned_labels = []
        token_ids = []

        for offsets, tokens, tags in zip(
            tokenized_inputs["offset_mapping"], examples["tokens"], examples["tags"]
        ):
            label_ids = []
            token_id_per_subtoken = []
            word_idx = 0
            char_pos = 0
            current_word = tokens[word_idx]
            current_label = tags[word_idx]

            for offset in offsets:
                if offset == (0, 0):
                    label_ids.append(-100)
                    token_id_per_subtoken.append(-100)
                    continue

                while offset[0] >= char_pos + len(current_word):
                    char_pos += len(current_word) + 1
                    word_idx += 1
                    if word_idx >= len(tokens):
                        break
                    current_word = tokens[word_idx]
                    current_label = tags[word_idx]

                if word_idx < len(tags):
                    label_ids.append(current_label)
                    token_id_per_subtoken.append(word_idx)
                else:
                    label_ids.append(-100)
                    token_id_per_subtoken.append(-100)

            aligned_labels.append(label_ids)
            token_ids.append(token_id_per_subtoken)

        tokenized_inputs["labels"] = aligned_labels
        tokenized_inputs["token_ids"] = token_ids
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs
    
    def sanitize_predictions_labels(self, predictions, labels, token_ids):
        predictions_subtoken, labels_subtoken = [], []
        predictions_token, labels_token = [], []

        for preds, labs, tok_ids in zip(predictions, labels, token_ids):
            # Subtoken level
            predictions_subtoken.append([pred for pred, lab in zip(preds, labs) if lab != -100])
            labels_subtoken.append([lab for lab in labs if lab != -100])

            # Token level
            unique_tok_ids = sorted(list(set(tok_ids) - {-100}))
            preds_token, labs_token = [], []

            for tok_id in unique_tok_ids:
                preds_id = [pred for pred, _id in zip(preds, tok_ids) if _id == tok_id]
                preds_token.append(max(set(preds_id), key=preds_id.count))
                labs_token.append([lab for lab, _id in zip(labs, tok_ids) if _id == tok_id][0])
            
            predictions_token.append(preds_token)
            labels_token.append(labs_token)
        
        return predictions_subtoken, labels_subtoken, predictions_token, labels_token
    
    # def vanilla_tokenization_fn(self, examples):
    #     sentences = [" ".join(tokens) for tokens in examples["tokens"]]
    #     tokenized_inputs = self.tokenizer(
    #         sentences,
    #         truncation=True,
    #         max_length=self.max_length,
    #         return_offsets_mapping=True,
    #     )
    #     aligned_labels = []
    #     entity_indices = []
        
    #     for offsets, tokens, tags in zip(
    #         tokenized_inputs["offset_mapping"], examples["tokens"], examples["tags"]
    #     ):
    #         label_ids = []
    #         entity_id_per_token = []
    #         word_idx = 0
    #         char_pos = 0
    #         current_word = tokens[word_idx]
    #         current_label = tags[word_idx]
    #         entity_counter = -1

    #         for offset in offsets:
    #             if offset == (0, 0):
    #                 label_ids.append(-100)
    #                 entity_id_per_token.append(-100)
    #                 continue

    #             while offset[0] >= char_pos + len(current_word):
    #                 char_pos += len(current_word) + 1 
    #                 word_idx += 1
    #                 if word_idx >= len(tokens):
    #                     break
    #                 current_word = tokens[word_idx]
    #                 current_label = tags[word_idx]

    #             if word_idx < len(tags):
    #                 label_ids.append(current_label)
                    
    #                 if current_label != 0:
    #                     if len(entity_id_per_token) == 0 or entity_id_per_token[-1] != entity_counter:
    #                         entity_counter += 1
    #                     entity_id_per_token.append(entity_counter)
    #                 else:
    #                     entity_id_per_token.append(-100)
                
    #             else:
    #                 label_ids.append(-100)
    #                 entity_id_per_token.append(-100)

    #         aligned_labels.append(label_ids)
    #         entity_indices.append(entity_id_per_token)

    #     tokenized_inputs["labels"] = aligned_labels
    #     tokenized_inputs["entity_indices"] = entity_indices 
    #     tokenized_inputs.pop("offset_mapping")
    #     return tokenized_inputs
    
    # def optimus_tokenization_fn(self, examples):
    #     sentences = [" ".join(tokens) for tokens in examples["tokens"]]
    #     tokenized_inputs = self.tokenizer(
    #         [sentence + self.tokenizer.eos_token for sentence in sentences],
    #         truncation=True,
    #         max_length=self.max_length,
    #         add_special_tokens=False,
    #         return_offsets_mapping=True,
    #     )
    #     aligned_labels = []
    #     entity_indices = []
        
    #     for offsets, tokens, tags in zip(
    #         tokenized_inputs["offset_mapping"], examples["tokens"], examples["tags"]
    #     ):
    #         label_ids = []
    #         entity_id_per_token = []
    #         word_idx = 0
    #         char_pos = 0
    #         current_word = tokens[word_idx]
    #         current_label = tags[word_idx]
    #         entity_counter = -1

    #         for offset in offsets:
    #             if offset == (0, 0):
    #                 label_ids.append(-100)
    #                 entity_id_per_token.append(-100)
    #                 continue

    #             while offset[0] >= char_pos + len(current_word):
    #                 char_pos += len(current_word) + 1 
    #                 word_idx += 1
    #                 if word_idx >= len(tokens):
    #                     break
    #                 current_word = tokens[word_idx]
    #                 current_label = tags[word_idx]

    #             if word_idx < len(tags):
    #                 label_ids.append(current_label)
                    
    #                 if current_label != 0:
    #                     if len(entity_id_per_token) == 0 or entity_id_per_token[-1] != entity_counter:
    #                         entity_counter += 1
    #                     entity_id_per_token.append(entity_counter)
    #                 else:
    #                     entity_id_per_token.append(-100)
                
    #             else:
    #                 label_ids.append(-100)
    #                 entity_id_per_token.append(-100)

    #         aligned_labels.append(label_ids)
    #         entity_indices.append(entity_id_per_token)

    #     tokenized_inputs["labels"] = aligned_labels
    #     tokenized_inputs["entity_indices"] = entity_indices 
    #     tokenized_inputs.pop("offset_mapping")
    #     return tokenized_inputs
    
    # def compute_metrics(self, eval_pred, label_list):
    #     predictions, labels = eval_pred
    #     predictions = (
    #         predictions[0].argmax(2) if isinstance(predictions, tuple)
    #         else predictions.argmax(2)
    #     )
    #     true_predictions = [
    #         label_list[pred] for prediction, label in zip(predictions, labels)
    #         for (pred, lab) in zip(prediction, label) if lab != -100
    #     ]
    #     true_labels = [
    #         label_list[lab] for prediction, label in zip(predictions, labels)
    #         for (_, lab) in zip(prediction, label) if lab != -100
    #     ]
    #     return {
    #         "precision": precision_score(
    #             true_labels, true_predictions, labels=label_list, average="micro",
    #         ),
    #         "recall": recall_score(
    #             true_labels, true_predictions, labels=label_list, average="micro",
    #         ),
    #         "f1": f1_score(
    #             true_labels, true_predictions, labels=label_list, average="micro",
    #         ),
    #     }
    