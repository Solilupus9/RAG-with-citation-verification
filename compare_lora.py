import os
import time

import numpy as np
import pandas as pd
from evaluate import ndcg_at_k, mrr_at_k, recall_at_k
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

DATA_DIR = "./data"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BASE_INDEX_DIR = "./indexes/dense"
FULL_FT_INDEX_DIR = "./indexes/dense_finetuned"
LORA_INDEX_DIR = "./indexes/dense_lora"
LORA_MODEL_DIR = "./models/bge-small-lora"


def load_dense_index(embed_dir: str):
	embeddings = np.load(f"{embed_dir}/embeddings.npy")
	doc_ids = np.load(f"{embed_dir}/doc_ids.npy", allow_pickle=True).tolist()
	return embeddings, doc_ids


def dense_search_vector(query_vector: np.ndarray, embeddings: np.ndarray, doc_ids: list, k: int = 10):
	scores = embeddings @ query_vector
	if k >= len(scores):
		top_k_indices = np.argsort(scores)[::-1]
	else:
		top_partition = np.argpartition(scores, -k)[-k:]
		top_k_indices = top_partition[np.argsort(scores[top_partition])[::-1]]
	return [doc_ids[idx] for idx in top_k_indices]


def run_lora_comparison(k: int = 10):
	print("=" * 88)
	print("       DENSE RETRIEVAL COMPARISON: BASE vs FULL FINE-TUNED vs LoRA")
	print("=" * 88)

	# 1. Load Data
	queries = pd.read_parquet(f"{DATA_DIR}/queries.parquet")
	qrels = pd.read_parquet(f"{DATA_DIR}/qrels.parquet")

	queries["_id"] = queries["_id"].astype(str)
	qrels["query-id"] = qrels["query-id"].astype(str)
	qrels["corpus-id"] = qrels["corpus-id"].astype(str)

	query_lookup = dict(zip(queries["_id"], queries["text"].astype(str)))

	split_file = "./data/splits/test_query_ids.csv"
	if os.path.exists(split_file):
		test_qids = pd.read_csv(split_file)["query_id"].astype(str).tolist()
		print(f"Loaded {len(test_qids)} held-out test queries from {split_file}")
	else:
		test_qids = qrels["query-id"].unique().tolist()

	# 2. Load Models & Indexes
	print("\n--- Loading Models and Indexes ---")
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Model 1: Base Pre-trained
	print("Loading 1. Base Model (BAAI/bge-small-en-v1.5)...")
	base_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
	base_embs, base_doc_ids = load_dense_index(BASE_INDEX_DIR)

	# Model 2: Full Fine-Tuned
	print("Loading 2. Full Fine-Tuned Model (./models/bge-small-finetuned)...")
	full_ft_model = SentenceTransformer("./models/bge-small-finetuned", device=device)
	full_ft_embs, full_ft_doc_ids = load_dense_index(FULL_FT_INDEX_DIR)

	# Model 3: LoRA Model
	print(f"Loading 3. LoRA Model ({LORA_MODEL_DIR})...")
	lora_tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_DIR)
	lora_model = AutoModel.from_pretrained(LORA_MODEL_DIR).to(device).eval()
	lora_embs, lora_doc_ids = load_dense_index(LORA_INDEX_DIR)

	methods = [
		"1. Base Pre-Trained (bge-small)",
		"2. Full Fine-Tuned (All 33M Params)",
		"3. LoRA Fine-Tuned (Adapters Merged)",
	]
	scores = {m: {"ndcg": [], "mrr": [], "recall": []} for m in methods}

	print(f"\nBenchmarking {len(test_qids)} held-out test queries (k={k})...\n")
	t0 = time.perf_counter()

	for idx, qid in enumerate(test_qids, start=1):
		query_text = query_lookup.get(qid)
		if not query_text:
			continue

		relevant_ids = set(qrels[qrels["query-id"] == qid]["corpus-id"].tolist())
		if not relevant_ids:
			continue

		query_with_inst = f"{QUERY_INSTRUCTION}{query_text}"

		# 1. Base query vector
		base_vec = base_model.encode(
			[query_with_inst],
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)[0].astype(np.float32)
		base_hits = dense_search_vector(base_vec, base_embs, base_doc_ids, k=k)

		# 2. Full FT query vector
		full_ft_vec = full_ft_model.encode(
			[query_with_inst],
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False,
		)[0].astype(np.float32)
		full_ft_hits = dense_search_vector(full_ft_vec, full_ft_embs, full_ft_doc_ids, k=k)

		# 3. LoRA query vector
		with torch.no_grad():
			inputs = lora_tokenizer(
				[query_with_inst],
				padding=True,
				truncation=True,
				max_length=256,
				return_tensors="pt"
			).to(device)
			outputs = lora_model(**inputs)
			cls_emb = outputs[0][:, 0]
			lora_vec = F.normalize(cls_emb, p=2, dim=1).cpu().numpy()[0].astype(np.float32)
		lora_hits = dense_search_vector(lora_vec, lora_embs, lora_doc_ids, k=k)

		runs = {
			"1. Base Pre-Trained (bge-small)": base_hits,
			"2. Full Fine-Tuned (All 33M Params)": full_ft_hits,
			"3. LoRA Fine-Tuned (Adapters Merged)": lora_hits,
		}

		for m, retrieved in runs.items():
			scores[m]["ndcg"].append(ndcg_at_k(retrieved, relevant_ids, k=k))
			scores[m]["mrr"].append(mrr_at_k(retrieved, relevant_ids, k=k))
			scores[m]["recall"].append(recall_at_k(retrieved, relevant_ids, k=k))

	eval_time = time.perf_counter() - t0
	print(f"Benchmark completed in {eval_time:.1f}s!\n")

	# Print Summary Table
	print("=" * 88)
	print(f"{'Method / Configuration':<44} {'NDCG@10':>12} {'MRR@10':>12} {'Recall@10':>14}")
	print("=" * 88)

	base_ndcg = np.mean(scores["1. Base Pre-Trained (bge-small)"]["ndcg"]) * 100
	base_mrr = np.mean(scores["1. Base Pre-Trained (bge-small)"]["mrr"]) * 100
	base_rec = np.mean(scores["1. Base Pre-Trained (bge-small)"]["recall"]) * 100

	fft_ndcg = np.mean(scores["2. Full Fine-Tuned (All 33M Params)"]["ndcg"]) * 100
	fft_mrr = np.mean(scores["2. Full Fine-Tuned (All 33M Params)"]["mrr"]) * 100
	fft_rec = np.mean(scores["2. Full Fine-Tuned (All 33M Params)"]["recall"]) * 100

	lora_ndcg = np.mean(scores["3. LoRA Fine-Tuned (Adapters Merged)"]["ndcg"]) * 100
	lora_mrr = np.mean(scores["3. LoRA Fine-Tuned (Adapters Merged)"]["mrr"]) * 100
	lora_rec = np.mean(scores["3. LoRA Fine-Tuned (Adapters Merged)"]["recall"]) * 100

	print(f"{'1. Base Pre-Trained (bge-small)':<44} {base_ndcg:>11.2f}% {base_mrr:>11.2f}% {base_rec:>13.2f}%")
	print(f"{'2. Full Fine-Tuned (All 33M Params)':<44} {fft_ndcg:>11.2f}% {fft_mrr:>11.2f}% {fft_rec:>13.2f}%")
	print(f"{'3. LoRA Fine-Tuned (Adapters Merged)':<44} {lora_ndcg:>11.2f}% {lora_mrr:>11.2f}% {lora_rec:>13.2f}%")
	print("-" * 88)

	d_lora_vs_base_ndcg = lora_ndcg - base_ndcg
	d_lora_vs_base_mrr = lora_mrr - base_mrr
	d_lora_vs_base_rec = lora_rec - base_rec

	d_lora_vs_fft_ndcg = lora_ndcg - fft_ndcg
	d_lora_vs_fft_mrr = lora_mrr - fft_mrr
	d_lora_vs_fft_rec = lora_rec - fft_rec

	print(f"{'Delta: LoRA vs Base Pre-Trained':<44} {f'{d_lora_vs_base_ndcg:+6.2f}%':>12} {f'{d_lora_vs_base_mrr:+6.2f}%':>12} {f'{d_lora_vs_base_rec:+6.2f}%':>14}")
	print(f"{'Delta: LoRA vs Full Fine-Tuned':<44} {f'{d_lora_vs_fft_ndcg:+6.2f}%':>12} {f'{d_lora_vs_fft_mrr:+6.2f}%':>12} {f'{d_lora_vs_fft_rec:+6.2f}%':>14}")
	print("=" * 88)


if __name__ == "__main__":
	run_lora_comparison(k=10)
