import os
import random
import time
from collections import defaultdict

import bm25s
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Stemmer import Stemmer
from bm25_retriever import load_bm25_index
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = "./data"
MODEL_OUTPUT_DIR = "./models/cross-encoder-finetuned"
BASE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
BATCH_SIZE = 16
MAX_LENGTH = 256
EPOCHS = 3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)


class CrossEncoderDataset(Dataset):
	def __init__(self, pairs_with_labels: list[tuple[str, str, float]]):
		self.data = pairs_with_labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


def collate_pairs(batch):
	queries = [item[0] for item in batch]
	texts = [item[1] for item in batch]
	labels = [item[2] for item in batch]
	return queries, texts, labels


def prepare_crossencoder_data():
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

	# Read train split
	train_split_path = "./data/splits/train_query_ids.csv"
	if os.path.exists(train_split_path):
		train_qids = set(pd.read_csv(train_split_path)["query_id"].astype(str).tolist())
		print(f"Loaded {len(train_qids)} training queries from {train_split_path}")
	else:
		unique_query_ids = sorted(list(query_to_positives.keys()))
		random.Random(SEED).shuffle(unique_query_ids)
		split_idx = int(len(unique_query_ids) * 0.80)
		train_qids = set(unique_query_ids[:split_idx])
		test_qids = set(unique_query_ids[split_idx:])
		os.makedirs("./data/splits", exist_ok=True)
		pd.DataFrame({"query_id": list(test_qids)}).to_csv("./data/splits/test_query_ids.csv", index=False)
		pd.DataFrame({"query_id": list(train_qids)}).to_csv(train_split_path, index=False)
		print(f"Created train/test split: {len(train_qids)} train, {len(test_qids)} test")

	# Mine Hard Negatives via BM25
	print("\n--- Step 2: Mining BM25 Hard Negatives for Binary Cross-Encoder Pairs ---")
	bm25_retriever = load_bm25_index()
	stemmer = Stemmer("english")

	train_pairs = []
	pos_count = 0
	neg_count = 0

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

		# Add positive samples (label = 1.0)
		for pos_id in pos_ids:
			pos_text = corpus_lookup.get(pos_id)
			if pos_text:
				train_pairs.append((query_text, pos_text, 1.0))
				pos_count += 1

		# Add up to 3 hard negative samples (label = 0.0)
		for neg_id in hard_neg_ids[:3]:
			neg_text = corpus_lookup.get(neg_id)
			if neg_text:
				train_pairs.append((query_text, neg_text, 0.0))
				neg_count += 1

	random.Random(SEED).shuffle(train_pairs)
	print(f"Constructed {len(train_pairs):,} pairs: {pos_count:,} Positives (1.0) and {neg_count:,} Hard Negatives (0.0).")
	return train_pairs


def train_crossencoder(train_pairs: list[tuple[str, str, float]]):
	print(f"\n--- Step 3: Fine-Tuning Cross-Encoder ({BASE_MODEL_NAME}) ---")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

	tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
	model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=1).to(device)

	dataset = CrossEncoderDataset(train_pairs)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairs)

	optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
	loss_fn = nn.BCEWithLogitsLoss()
	scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

	total_steps = len(dataloader) * EPOCHS
	print(f"Starting PyTorch loop (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Total Steps: {total_steps})...")
	start_time = time.perf_counter()

	for epoch in range(EPOCHS):
		model.train()
		total_loss = 0.0
		t_epoch = time.perf_counter()

		for step, (queries, texts, labels) in enumerate(dataloader, start=1):
			inputs = tokenizer(
				queries,
				texts,
				padding=True,
				truncation=True,
				max_length=MAX_LENGTH,
				return_tensors="pt"
			).to(device)

			targets = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(1)

			optimizer.zero_grad()
			with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
				outputs = model(**inputs)
				logits = outputs.logits
				loss = loss_fn(logits, targets)

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			total_loss += loss.item()

			if step % 35 == 0 or step == len(dataloader):
				avg_loss = total_loss / step
				print(f"  Epoch {epoch + 1}/{EPOCHS} | Step {step:3d}/{len(dataloader)} | Running Loss: {avg_loss:.4f}")

		epoch_duration = time.perf_counter() - t_epoch
		print(f"Epoch {epoch + 1} finished in {epoch_duration:.1f}s | Average Loss: {total_loss / len(dataloader):.4f}\n")

	train_duration = time.perf_counter() - start_time
	print(f"Cross-Encoder fine-tuning completed in {train_duration:.1f}s ({train_duration / 60:.1f} minutes)!")

	# Save locally in format compatible with CrossEncoder
	os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
	model.save_pretrained(MODEL_OUTPUT_DIR)
	tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
	print(f"Fine-tuned model successfully saved locally to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
	total_start = time.perf_counter()
	pairs = prepare_crossencoder_data()
	train_crossencoder(pairs)
	total_elapsed = time.perf_counter() - total_start
	print(f"\nAll tasks finished in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} minutes)!")
