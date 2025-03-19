from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sklearn.metrics import ndcg_score
import torch
from tqdm import tqdm

from .abstract_eval import AbstractEval


class RetrievalEval(AbstractEval):    
    def train(self) -> None:
        train_dataset = self.dataset["train"]
        if "subset" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns("subset")
        
        if self.tr_args.eval_strategy != "no":
            val_dataset = self.dataset["validation"]
            if "subset" in val_dataset.column_names:
                val_dataset = val_dataset.remove_columns("subset")
        else:
            val_dataset = None

        self.model.tokenizer = self.tokenizer
        tokenization_fn = self.get_tokenization_fn()
        data_collator = SentenceTransformerDataCollator(tokenization_fn)
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

        # Save model if not evaluated
        if not self.tr_args.do_predict:
            print(f"Saving model at {self.tr_args.output_dir}")
            trainer.save_model(self.tr_args.output_dir)

    def validate(self) -> Dict[str, List]:
        print("Evaluating on validation dataset")
        return self.evaluate("validation")
    
    def test(self) -> Dict[str, List]:
        print("Evaluating on test dataset")
        return self.evaluate("test")
    
    def evaluate(self, split) -> Dict[str, List]:
        eval_dataset = self.dataset[split]
        queries, corpus, qrels = self.get_queries_corpus_qrels(eval_dataset)
        self.model.eval()
        queries_enc = self.encode_sequences(queries)
        docs_enc = self.encode_sequences(corpus)
        similarity_matrix = self.model.similarity(queries_enc, docs_enc).numpy()
        metrics_per_instance = self.compute_metrics_instances(qrels, similarity_matrix)
        
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
        if self.model.__class__.__name__.startswith("Optimus"):
            return self.optimus_tokenization_fn
        else:
            return self.vanilla_tokenization_fn

    def vanilla_tokenization_fn(self, sequences: List) -> torch.Tensor:
        return self.tokenizer(
            sequences,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )        
    
    def optimus_tokenization_fn(self, sequences: List) -> torch.Tensor:
        return self.tokenizer(
            [seq + self.tokenizer.eos_token for seq in sequences],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
    
    def get_queries_corpus_qrels(self, eval_dataset):
        queries = sorted(list(set(eval_dataset["anchor"])))      
        query_to_id = {query: _id for _id, query in enumerate(queries)}  
        
        if isinstance(eval_dataset[0]["positive"], str):
            corpus = sorted(list(set(eval_dataset["positive"])))
        elif isinstance(eval_dataset[0]["positive"], list):
            corpus = sorted(list(set(doc for docs in eval_dataset["positive"] for doc in docs)))
        
        if "negative" in eval_dataset.column_names:
            if isinstance(eval_dataset[0]["negative"], str):
                corpus = sorted(list(set(corpus) | set(eval_dataset["negative"])))
            elif isinstance(eval_dataset[0]["negative"], list):
                corpus = sorted(list(set(corpus) | set(doc for docs in eval_dataset["negative"] for doc in docs)))
        
        doc_to_id = {doc: _id for _id, doc in enumerate(corpus)}
        qrels = {query_id: set() for query_id in range(len(queries))}
        
        if isinstance(eval_dataset[0]["positive"], str):            
            for example in eval_dataset:
                qrels[query_to_id[example["anchor"]]].add(doc_to_id[example["positive"]])
        elif isinstance(eval_dataset[0]["positive"], list):  
            for example in eval_dataset:
                qrels[query_to_id[example["anchor"]]] |= set(doc_to_id[doc] for doc in example["positive"])
        
        return queries, corpus, qrels

    def encode_sequences(self, sequences):
        sequences_enc = []
        for i in tqdm(
            range(0, len(sequences), self.tr_args.per_device_eval_batch_size), 
            desc="Encoding sequences"
        ):
            sequences_enc.append(
                self.model.encode(
                    sequences[i:i+self.tr_args.per_device_eval_batch_size],
                    batch_size=self.tr_args.per_device_eval_batch_size,
                )
            )
        sequences_enc = np.concatenate(sequences_enc)
        return sequences_enc

    def compute_metrics_instances(self, qrels, similarity_matrix):
        ndcgs = []
        for query_id, query_similarities in tqdm(
            list(enumerate(similarity_matrix)), desc="Evaluating"
        ):
            query_relevant_docs = [
                1 if doc_id in qrels[query_id] 
                else 0 for doc_id in range(len(query_similarities))
            ]
            ndcgs.append(
                ndcg_score([query_relevant_docs], [query_similarities.tolist()], k=10)
            )
        return {"ndcg": ndcgs}
