import os
import random
import time
from collections import defaultdict
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import torch
from Stemmer import Stemmer
from bm25_retriever import load_bm25_index
from datasets import Dataset
from sentence_transformers import (
	SentenceTransformer,
	SentenceTransformerTrainer,
	SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses

DATA_DIR = "./data"
MODEL_OUTPUT_DIR = "./models/bge-small-finetuned"
FINETUNED_INDEX_DIR = "./indexes/dense_finetuned"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 256
EMBED_BATCH_SIZE = 128
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)


def prepare_training_data():
	print("--- Step 1: Loading Corpus, Queries, and Relevance Labels ---")
	corpus_df = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	queries_df = pd.read_parquet(f"{DATA_DIR}/queries.parquet")
	qrels_df = pd.read_parquet(f"{DATA_DIR}/qrels.parquet")

	corpus_df["_id"] = corpus_df["_id"].astype(str)
	queries_df["_id"] = queries_df["_id"].astype(str)
	qrels_df["query-id"] = qrels_df["query-id"].astype(str)
	qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)

	corpus_lookup = dict(zip(corpus_df["_id"], corpus_df["text"].astype(str)))
	query_lookup = dict(zip(queries_df["_id"], queries_df["text"].astype(str)))

	query_to_positives = defaultdict(set)
	for _, row in qrels_df.iterrows():
		query_to_positives[row["query-id"]].add(row["corpus-id"])

	unique_query_ids = sorted(list(query_to_positives.keys()))
	random.Random(SEED).shuffle(unique_query_ids)

	# 80/20 train/test split
	split_idx = int(len(unique_query_ids) * 0.80)
	train_qids = set(unique_query_ids[:split_idx])
	test_qids = set(unique_query_ids[split_idx:])

	print(f"Total labeled queries: {len(unique_query_ids)}")
	print(f"  Training queries:   {len(train_qids)}")
	print(f"  Held-out test queries: {len(test_qids)}")

	# Save test query IDs for reproducible evaluation
	os.makedirs("./data/splits", exist_ok=True)
	pd.DataFrame({"query_id": list(test_qids)}).to_csv("./data/splits/test_query_ids.csv", index=False)
	pd.DataFrame({"query_id": list(train_qids)}).to_csv("./data/splits/train_query_ids.csv", index=False)

	# Mine Hard Negatives using BM25
	print("\n--- Step 2: Mining Hard Negatives via BM25 ---")
	bm25_retriever = load_bm25_index()
	stemmer = Stemmer("english")

	anchors = []
	positives = []
	negatives = []

	for qid in train_qids:
		query_text = query_lookup.get(qid)
		if not query_text:
			continue

		pos_ids = query_to_positives[qid]

		# Silent BM25 retrieve
		query_tokens = bm25s.tokenize(query_text, stopwords="en", stemmer=stemmer, show_progress=False)
		results, _ = bm25_retriever.retrieve(query_tokens, k=25, show_progress=False)

		hard_neg_ids = []
		for i in range(results.shape[1]):
			doc = results[0, i]
			doc_id = doc.get("id") if isinstance(doc, dict) else str(doc)
			if doc_id and doc_id not in pos_ids:
				hard_neg_ids.append(doc_id)

		if not hard_neg_ids:
			continue

		for pos_id in pos_ids:
			pos_text = corpus_lookup.get(pos_id)
			if not pos_text:
				continue

			# Take up to 2 hard negatives per positive pair
			for neg_id in hard_neg_ids[:2]:
				neg_text = corpus_lookup.get(neg_id)
				if neg_text:
					anchors.append(f"{QUERY_INSTRUCTION}{query_text}")
					positives.append(pos_text)
					negatives.append(neg_text)

	print(f"Constructed {len(anchors):,} training triplets (Anchor + Positive + Hard Negative).")
	train_dataset = Dataset.from_dict({
		"anchor": anchors,
		"positive": positives,
		"negative": negatives,
	})
	return train_dataset, corpus_df, test_qids


def train_biencoder(train_dataset: Dataset):
	print("\n--- Step 3: Fine-Tuning BAAI/bge-small-en-v1.5 ---")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

	model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
	model.max_seq_length = MAX_SEQ_LENGTH
	loss = losses.MultipleNegativesRankingLoss(model)

	os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

	training_args = SentenceTransformerTrainingArguments(
		output_dir=MODEL_OUTPUT_DIR,
		num_train_epochs=3,
		per_device_train_batch_size=BATCH_SIZE,
		learning_rate=2e-5,
		warmup_ratio=0.1,
		fp16=torch.cuda.is_available(),
		save_strategy="no",
		logging_steps=25,
		report_to="none",
		seed=SEED,
	)

	trainer = SentenceTransformerTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		loss=loss,
	)

	print("Starting training loop...")
	start_time = time.perf_counter()
	trainer.train()
	train_duration = time.perf_counter() - start_time
	print(f"Fine-tuning complete in {train_duration:.1f}s ({train_duration / 60:.1f} minutes)!")

	# Save final model locally
	model.save_pretrained(MODEL_OUTPUT_DIR)
	print(f"Fine-tuned model successfully saved locally to: {MODEL_OUTPUT_DIR}")
	return model


def build_finetuned_index(model: SentenceTransformer, corpus_df: pd.DataFrame):
	print("\n--- Step 4: Building Separate Fine-Tuned Dense Embedding Index ---")
	os.makedirs(FINETUNED_INDEX_DIR, exist_ok=True)

	doc_ids = [str(doc_id) for doc_id in corpus_df["_id"].tolist()]
	doc_texts = [str(text) for text in corpus_df["text"].tolist()]

	print(f"Encoding {len(doc_texts):,} documents with the fine-tuned model...")
	all_embeddings = []
	t0 = time.perf_counter()

	for i in range(0, len(doc_texts), EMBED_BATCH_SIZE):
		batch = doc_texts[i: i + EMBED_BATCH_SIZE]
		batch_emb = model.encode(
			batch,
			batch_size=EMBED_BATCH_SIZE,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		).astype(np.float32)
		all_embeddings.append(batch_emb)
		if (i // EMBED_BATCH_SIZE) % 50 == 0:
			print(f"  Encoded {i + len(batch):,}/{len(doc_texts):,} passages...")

	embeddings_matrix = np.vstack(all_embeddings)
	# Normalize L2
	norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
	norms = np.where(norms == 0, 1, norms)
	embeddings_matrix = embeddings_matrix / norms

	# Save to separate index directory
	np.save(f"{FINETUNED_INDEX_DIR}/embeddings.npy", embeddings_matrix)
	np.save(f"{FINETUNED_INDEX_DIR}/doc_ids.npy", np.array(doc_ids))

	total_encode_time = time.perf_counter() - t0
	size_mb = embeddings_matrix.nbytes / 1e6
	print(f"Fine-tuned index built in {total_encode_time:.1f}s!")
	print(f"Saved to: {FINETUNED_INDEX_DIR}/ ({size_mb:.1f} MB)")


if __name__ == "__main__":
	total_start = time.perf_counter()
	train_dataset, corpus_df, test_qids = prepare_training_data()
	ft_model = train_biencoder(train_dataset)
	build_finetuned_index(ft_model, corpus_df)
	total_elapsed = time.perf_counter() - total_start
	print(f"\nAll tasks finished in {total_elapsed / 60:.1f} minutes!")
