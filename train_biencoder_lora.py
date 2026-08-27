import os
import random
import time
from collections import defaultdict

import bm25s
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Stemmer import Stemmer
from bm25_retriever import load_bm25_index
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

DATA_DIR = "./data"
MODEL_OUTPUT_DIR = "./models/bge-small-lora"
LORA_INDEX_DIR = "./indexes/dense_lora"
BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
EMBED_BATCH_SIZE = 128
EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
TEMPERATURE = 0.05
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)


class TripletDataset(Dataset):
	def __init__(self, triplets: list[tuple[str, str, str]]):
		self.triplets = triplets

	def __len__(self):
		return len(self.triplets)

	def __getitem__(self, idx):
		return self.triplets[idx]


def collate_fn(batch):
	anchors = [item[0] for item in batch]
	positives = [item[1] for item in batch]
	negatives = [item[2] for item in batch]
	return anchors, positives, negatives


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

	# Read the exact same train split
	train_split_path = "./data/splits/train_query_ids.csv"
	if os.path.exists(train_split_path):
		train_qids = set(pd.read_csv(train_split_path)["query_id"].astype(str).tolist())
		print(f"Loaded {len(train_qids)} training queries from {train_split_path}")
	else:
		unique_query_ids = sorted(list(query_to_positives.keys()))
		random.Random(SEED).shuffle(unique_query_ids)
		split_idx = int(len(unique_query_ids) * 0.80)
		train_qids = set(unique_query_ids[:split_idx])

	print(f"Total training queries: {len(train_qids)}")

	# Step 2: Mining Hard Negatives via BM25 (1 balanced hard negative per positive)
	print("\n--- Step 2: Mining Balanced Hard Negatives via BM25 ---")
	bm25_retriever = load_bm25_index()
	stemmer = Stemmer("english")

	triplets = []
	for qid in train_qids:
		query_text = query_lookup.get(qid)
		if not query_text:
			continue

		pos_ids = query_to_positives[qid]

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

			for neg_id in hard_neg_ids[:1]:
				neg_text = corpus_lookup.get(neg_id)
				if neg_text:
					triplets.append((f"{QUERY_INSTRUCTION}{query_text}", pos_text, neg_text))

	random.Random(SEED).shuffle(triplets)
	print(f"Constructed {len(triplets):,} balanced training triplets.")
	return triplets, corpus_df


def encode_batch(model, tokenizer, texts, device):
	inputs = tokenizer(
		texts,
		padding=True,
		truncation=True,
		max_length=MAX_SEQ_LENGTH,
		return_tensors="pt"
	).to(device)
	outputs = model(**inputs)
	# CLS token pooling (BGE standard)
	cls_embeddings = outputs[0][:, 0]
	# Normalize embeddings
	normalized = F.normalize(cls_embeddings, p=2, dim=1)
	return normalized


def train_lora_biencoder(triplets: list[tuple[str, str, str]]):
	print(f"\n--- Step 3: Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}) & Fine-Tuning ---")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

	tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
	base_hf_model = AutoModel.from_pretrained(BASE_MODEL_NAME)

	peft_config = LoraConfig(
		r=LORA_R,
		lora_alpha=LORA_ALPHA,
		target_modules=["query", "value"],
		lora_dropout=0.05,
		bias="none",
	)

	model = get_peft_model(base_hf_model, peft_config).to(device)
	print("LoRA Parameter Summary:")
	model.print_trainable_parameters()

	dataset = TripletDataset(triplets)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

	optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
	scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
	cross_entropy = nn.CrossEntropyLoss()

	total_steps = len(dataloader) * EPOCHS
	print(f"\nStarting PyTorch LoRA loop (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Total Steps: {total_steps})...")
	start_time = time.perf_counter()

	for epoch in range(EPOCHS):
		model.train()
		total_loss = 0.0
		t_epoch = time.perf_counter()

		for step, (anchors, positives, negatives) in enumerate(dataloader, start=1):
			optimizer.zero_grad()

			with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
				q_embs = encode_batch(model, tokenizer, anchors, device)
				p_embs = encode_batch(model, tokenizer, positives, device)
				n_embs = encode_batch(model, tokenizer, negatives, device)

				# Candidates: all positives in batch + all mined negatives in batch
				# Shape: (2 * B, D)
				candidates = torch.cat([p_embs, n_embs], dim=0)

				# Cosine similarity logits scaled by temperature: (B, 2 * B)
				logits = torch.matmul(q_embs, candidates.T) / TEMPERATURE

				# Ground truth target: index of matching positive (0 to B - 1)
				targets = torch.arange(len(anchors), device=device)

				loss = cross_entropy(logits, targets)

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			total_loss += loss.item()

			if step % 15 == 0 or step == len(dataloader):
				avg_loss = total_loss / step
				print(f"  Epoch {epoch + 1}/{EPOCHS} | Step {step:2d}/{len(dataloader)} | Running Loss: {avg_loss:.4f}")

		epoch_duration = time.perf_counter() - t_epoch
		print(f"Epoch {epoch + 1} finished in {epoch_duration:.1f}s | Average Loss: {total_loss / len(dataloader):.4f}\n")

	train_duration = time.perf_counter() - start_time
	print(f"LoRA fine-tuning completed in {train_duration:.1f}s ({train_duration / 60:.1f} minutes)!")

	# Merge LoRA adapters back into base model
	print("Merging LoRA adapter weights into base model...")
	merged_model = model.merge_and_unload()

	# Save locally in standard format
	os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
	merged_model.save_pretrained(MODEL_OUTPUT_DIR)
	tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
	print(f"Merged LoRA model saved locally to: {MODEL_OUTPUT_DIR}")
	return merged_model, tokenizer


def build_lora_index(model, tokenizer, corpus_df: pd.DataFrame):
	print("\n--- Step 4: Building Separate LoRA Dense Embedding Index ---")
	os.makedirs(LORA_INDEX_DIR, exist_ok=True)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.eval()

	doc_ids = [str(doc_id) for doc_id in corpus_df["_id"].tolist()]
	doc_texts = [str(text) for text in corpus_df["text"].tolist()]

	print(f"Encoding {len(doc_texts):,} documents with LoRA fine-tuned model...")
	all_embeddings = []
	t0 = time.perf_counter()

	with torch.no_grad():
		for i in range(0, len(doc_texts), EMBED_BATCH_SIZE):
			batch = doc_texts[i: i + EMBED_BATCH_SIZE]
			with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
				batch_emb = encode_batch(model, tokenizer, batch, device).cpu().numpy()
			all_embeddings.append(batch_emb)
			if (i // EMBED_BATCH_SIZE) % 50 == 0:
				print(f"  Encoded {i + len(batch):,}/{len(doc_texts):,} passages...")

	embeddings_matrix = np.vstack(all_embeddings)

	# Save to separate index directory
	np.save(f"{LORA_INDEX_DIR}/embeddings.npy", embeddings_matrix)
	np.save(f"{LORA_INDEX_DIR}/doc_ids.npy", np.array(doc_ids))

	total_encode_time = time.perf_counter() - t0
	size_mb = embeddings_matrix.nbytes / 1e6
	print(f"LoRA index built in {total_encode_time:.1f}s ({total_encode_time / 60:.1f} minutes)!")
	print(f"Saved to: {LORA_INDEX_DIR}/ ({size_mb:.1f} MB)")


if __name__ == "__main__":
	total_start = time.perf_counter()
	triplets, corpus_df = prepare_training_data()
	lora_model, tokenizer = train_lora_biencoder(triplets)
	build_lora_index(lora_model, tokenizer, corpus_df)
	total_elapsed = time.perf_counter() - total_start
	print(f"\nAll LoRA tasks completed in {total_elapsed / 60:.1f} minutes!")
