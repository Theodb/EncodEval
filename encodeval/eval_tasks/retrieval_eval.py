from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sklearn.metrics import ndcg_score
import torch
from tqdm import tqdm

from .abstract_eval import AbstractEval


class RetrievalEval(AbstractEval):
    """
    Evaluation class for training and evaluating sentence retrieval models.
    Inherits from AbstractEval and implements training and evaluation for information retrieval tasks.
    """

    def train(self) -> None:
        """
        Train the retrieval model using the training dataset and optional validation dataset.
        Saves the model to the output directory after training (if not evaluating).
        """
        train_dataset = self.dataset["train"]

        # Remove the 'subset' column if it exists (used only for per-subset metrics)
        if "subset" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns("subset")

        # Load validation dataset only if evaluation during training is enabled
        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"]
            if "subset" in val_dataset.column_names:
                val_dataset = val_dataset.remove_columns("subset")
        else:
            val_dataset = None

        self.model.tokenizer = self.tokenizer
        tokenization_fn = self.get_tokenization_fn()
        data_collator = SentenceTransformerDataCollator(tokenization_fn)

        # Instantiate loss function with optional keyword arguments
        loss = (
            self.loss_fn(model=self.model, **self.loss_kwargs)
            if self.loss_kwargs is not None else self.loss_fn(model=self.model)
        )

        print("==== Training Arguments ====")
        print(self.tr_args)
        print("============================")

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self.tr_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            loss=loss,
        )

        print("Training model")
        trainer.train()

        # Save model after training if evaluation is not run
        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, List]:
        """
        Evaluate the model on the validation split.

        Returns:
            Dict[str, List]: Average and per-instance evaluation metrics.
        """
        print("Evaluating on validation dataset")
        return self.evaluate("validation")

    def test(self) -> Dict[str, List]:
        """
        Evaluate the model on the test split.

        Returns:
            Dict[str, List]: Average and per-instance evaluation metrics.
        """
        print("Evaluating on test dataset")
        return self.evaluate("test")

    def evaluate(self, split: str) -> Dict[str, List]:
        """
        General evaluation logic used by both validate and test.

        Args:
            split (str): Dataset split to evaluate on, either 'validation' or 'test'.

        Returns:
            Dict[str, List]: Dictionary with averaged metrics and per-instance results.
        """
        eval_dataset = self.dataset[split]
        queries, corpus, qrels = self.get_queries_corpus_qrels(eval_dataset)

        self.model.eval()
        queries_enc = self.encode_sequences(queries)
        docs_enc = self.encode_sequences(corpus)

        # Compute similarity scores between query and corpus embeddings
        similarity_matrix = self.model.similarity(queries_enc, docs_enc).numpy()

        # Compute NDCG metrics per instance
        metrics_per_instance = self.compute_metrics_instances(qrels, similarity_matrix)

        # If the dataset includes subsets, tag each query with its subset
        if "subset" in eval_dataset.column_names:
            query_to_subset = (
                eval_dataset.to_pandas()[["subset", "anchor"]].drop_duplicates()
                .set_index("anchor").to_dict()["subset"]
            )
            metrics_per_instance["subset"] = [query_to_subset[query] for query in queries]

        return {
            "average": {k: np.mean(v) for k, v in metrics_per_instance.items() if k != "subset"},
            "per_instance": metrics_per_instance,
        }

    def get_tokenization_fn(self):
        """
        Choose tokenization function based on model type.

        Returns:
            Callable: Tokenization function.
        """
        if self.model.__class__.__name__.startswith("EuroBert"):
            return self.eurobert_tokenization_fn
        else:
            return self.vanilla_tokenization_fn

    def standard_tokenization_fn(self, sequences: List) -> torch.Tensor:
        """
        Standard tokenization function.

        Args:
            sequences (List): List of string sequences.

        Returns:
            torch.Tensor: Tokenized tensor input.
        """
        return self.tokenizer(
            sequences,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def eurobert_tokenization_fn(self, sequences: List) -> torch.Tensor:
        """
        Tokenization function for EuroBERT models (EOS, no BOS).

        Args:
            sequences (List): List of string sequences.

        Returns:
            torch.Tensor: Tokenized tensor input with EOS tokens.
        """
        return self.tokenizer(
            [seq + self.tokenizer.eos_token for seq in sequences],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def get_queries_corpus_qrels(self, eval_dataset):
        """
        Construct queries, corpus, and relevance labels (qrels) from dataset.

        Args:
            eval_dataset: HuggingFace dataset with 'anchor', 'positive', and optional 'negative' fields.

        Returns:
            Tuple[List[str], List[str], Dict[int, Set[int]]]: queries, corpus, qrels
        """
        queries = sorted(list(set(eval_dataset["anchor"])))
        query_to_id = {query: _id for _id, query in enumerate(queries)}

        # Extract unique corpus documents from positives
        if isinstance(eval_dataset[0]["positive"], str):
            corpus = sorted(list(set(eval_dataset["positive"])))
        elif isinstance(eval_dataset[0]["positive"], list):
            corpus = sorted(list(set(doc for docs in eval_dataset["positive"] for doc in docs)))

        # Add negatives to corpus if available
        if "negative" in eval_dataset.column_names:
            if isinstance(eval_dataset[0]["negative"], str):
                corpus = sorted(list(set(corpus) | set(eval_dataset["negative"])))
            elif isinstance(eval_dataset[0]["negative"], list):
                corpus = sorted(list(set(corpus) | set(doc for docs in eval_dataset["negative"] for doc in docs)))

        doc_to_id = {doc: _id for _id, doc in enumerate(corpus)}
        qrels = {query_id: set() for query_id in range(len(queries))}

        # Populate qrels from positives
        if isinstance(eval_dataset[0]["positive"], str):
            for example in eval_dataset:
                qrels[query_to_id[example["anchor"]]].add(doc_to_id[example["positive"]])
        elif isinstance(eval_dataset[0]["positive"], list):
            for example in eval_dataset:
                qrels[query_to_id[example["anchor"]]] |= set(doc_to_id[doc] for doc in example["positive"])

        return queries, corpus, qrels

    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Encode a list of text sequences into embeddings using the model.

        Args:
            sequences (List[str]): List of input texts.

        Returns:
            np.ndarray: 2D array of embeddings.
        """
        sequences_enc = []
        for i in tqdm(
            range(0, len(sequences), self.tr_args.per_device_eval_batch_size),
            desc="Encoding sequences"
        ):
            sequences_enc.append(
                self.model.encode(
                    sequences[i:i + self.tr_args.per_device_eval_batch_size],
                    batch_size=self.tr_args.per_device_eval_batch_size,
                )
            )
        sequences_enc = np.concatenate(sequences_enc)
        return sequences_enc

    def compute_metrics_instances(self, qrels: Dict[int, set], similarity_matrix: np.ndarray) -> Dict[str, List[float]]:
        """
        Compute per-query NDCG@10 based on the similarity matrix and ground-truth qrels.

        Args:
            qrels (Dict[int, set]): Mapping from query IDs to relevant doc IDs.
            similarity_matrix (np.ndarray): Query-to-document similarity scores.

        Returns:
            Dict[str, List[float]]: Per-query NDCG@10 scores.
        """
        ndcgs = []
        for query_id, query_similarities in tqdm(
            list(enumerate(similarity_matrix)), desc="Evaluating"
        ):
            # Binary relevance labels
            query_relevant_docs = [
                1 if doc_id in qrels[query_id]
                else 0 for doc_id in range(len(query_similarities))
            ]
            ndcgs.append(
                ndcg_score([query_relevant_docs], [query_similarities.tolist()], k=10)
            )
        return {"ndcg": ndcgs}
