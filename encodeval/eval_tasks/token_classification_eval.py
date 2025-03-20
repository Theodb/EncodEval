from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification, Trainer

from .abstract_eval import AbstractEval


class TokenClassificationEval(AbstractEval):
    """
    Evaluation and training class for token-level classification tasks 
    such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
    """

    def train(self) -> None:
        """
        Fine-tunes the token classification model using the training dataset.

        If evaluation is enabled, the validation set is also used. The model is saved to disk 
        unless `do_predict` is True.
        """
        print("Tokenizing training dataset")
        tokenization_fn = self.get_tokenization_fn()
        train_dataset = self.dataset["train"].map(tokenization_fn, batched=True, load_from_cache_file=False)

        train_dataset = train_dataset.remove_columns(
            [f for f in train_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
        )

        # Tokenize validation data if enabled
        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"]
            val_dataset = val_dataset.map(tokenization_fn, batched=True, load_from_cache_file=False)
            val_dataset = val_dataset.remove_columns(
                [f for f in val_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
            )
        else:
            val_dataset = None

        print("==== Training Arguments ====")
        print(self.tr_args)
        print("=============================")

        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

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
            Dict containing per-token and per-subtoken predictions and labels.
        """
        print("Evaluating on validation dataset")
        return self.evaluate("validation")

    def test(self) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluates the model on the test set.

        Returns:
            Dict containing per-token and per-subtoken predictions and labels.
        """
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluates the model on a given dataset split.

        Args:
            split (str): Either "validation" or "test".

        Returns:
            Dict containing per-instance predictions and labels at both token and subtoken levels.
        """
        print(f"Tokenizing {split} dataset")
        tokenization_fn = self.get_tokenization_fn()

        eval_dataset = self.dataset[split].map(tokenization_fn, batched=True, load_from_cache_file=False)
        subsets = eval_dataset["subset"] if "subset" in eval_dataset.column_names else None
        token_ids = eval_dataset["token_ids"]

        eval_dataset = eval_dataset.remove_columns(
            [f for f in eval_dataset.features if f not in ["input_ids", "attention_mask", "labels"]]
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.tr_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            pin_memory=True,
        )

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
            "average": None,  # TODO: Add dataset-level metrics (e.g., F1)
            "per_instance": metrics_per_instance,
        }

    def get_tokenization_fn(self):
        """
        Returns the appropriate tokenization function depending on the model class.

        Returns:
            Callable: Tokenization function.
        """
        if self.model.__class__.__name__.startswith("EuroBert"):
            return self.eurobert_tokenization_fn
        else:
            return self.standard_tokenization_fn

    def standard_tokenization_fn(self, examples):
        """
        Standard tokenization for token-level tasks, including subtoken-to-token label alignment.

        Args:
            examples (Dict): Dictionary with 'tokens' and 'tags'.

        Returns:
            Dict: Tokenized inputs with aligned labels and token indices.
        """
        sentences = [" ".join(tokens) for tokens in examples["tokens"]]
        tokenized_inputs = self.tokenizer(
            sentences,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        aligned_labels, token_ids = [], []

        for offsets, tokens, tags in zip(
            tokenized_inputs["offset_mapping"], examples["tokens"], examples["tags"]
        ):
            label_ids, token_id_per_subtoken = [], []
            word_idx, char_pos = 0, 0
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

    def eurobert_tokenization_fn(self, examples):
        """
        Tokenization for EuroBERT models with EOS tokens and label alignment.

        Args:
            examples (Dict): Dictionary with 'tokens' and 'tags'.

        Returns:
            Dict: Tokenized inputs with aligned labels and token indices.
        """
        sentences = [" ".join(tokens) for tokens in examples["tokens"]]
        tokenized_inputs = self.tokenizer(
            [sentence + self.tokenizer.eos_token for sentence in sentences],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        aligned_labels, token_ids = [], []

        for offsets, tokens, tags in zip(
            tokenized_inputs["offset_mapping"], examples["tokens"], examples["tags"]
        ):
            label_ids, token_id_per_subtoken = [], []
            word_idx, char_pos = 0, 0
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
        """
        Cleans and aggregates model predictions and labels.

        Filters out ignored subtoken positions and aggregates subtoken predictions to token-level predictions
        using majority vote.

        Args:
            predictions (List[List[int]]): Subtoken-level predicted labels.
            labels (List[List[int]]): Subtoken-level gold labels.
            token_ids (List[List[int]]): Mapping of subtokens to token indices.

        Returns:
            Tuple:
                - predictions_subtoken (List[List[int]])
                - labels_subtoken (List[List[int]])
                - predictions_token (List[List[int]])
                - labels_token (List[List[int]])
        """
        predictions_subtoken, labels_subtoken = [], []
        predictions_token, labels_token = [], []

        for preds, labs, tok_ids in zip(predictions, labels, token_ids):
            predictions_subtoken.append([pred for pred, lab in zip(preds, labs) if lab != -100])
            labels_subtoken.append([lab for lab in labs if lab != -100])

            unique_tok_ids = sorted(list(set(tok_ids) - {-100}))
            preds_token, labs_token = [], []

            for tok_id in unique_tok_ids:
                preds_for_token = [pred for pred, _id in zip(preds, tok_ids) if _id == tok_id]
                labs_for_token = [lab for lab, _id in zip(labs, tok_ids) if _id == tok_id]
                preds_token.append(max(set(preds_for_token), key=preds_for_token.count))
                labs_token.append(labs_for_token[0])

            predictions_token.append(preds_token)
            labels_token.append(labs_token)

        return predictions_subtoken, labels_subtoken, predictions_token, labels_token
