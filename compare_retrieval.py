import os
import random
import time
from typing import Optional

import numpy as np
import pandas as pd
from bm25_retriever import load_bm25_index, bm25_search
from dense_retriever import dense_search
from evaluate import ndcg_at_k, mrr_at_k, recall_at_k
from reranker import rerank_results
from rrf import weighted_rrf
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
BASE_INDEX_DIR = "./indexes/dense"
FINETUNED_INDEX_DIR = "./indexes/dense_finetuned"
FINETUNED_MODEL_DIR = "./models/bge-small-finetuned"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_index(index_dir: str):
	embeddings = np.load(f"{index_dir}/embeddings.npy")
	doc_ids = np.load(f"{index_dir}/doc_ids.npy", allow_pickle=True).tolist()
	return embeddings, doc_ids


def dense_search_with_model(
		query: str,
		embeddings: np.ndarray,
		doc_ids: list,
		model: SentenceTransformer,
		k: int = 10,
) -> list[dict]:
	query_with_instruction = f"{QUERY_INSTRUCTION}{query}"
	query_vector = model.encode(
		[query_with_instruction],
		convert_to_numpy=True,
		normalize_embeddings=True,
		show_progress_bar=False,
	)[0].astype(np.float32)

	norm = np.linalg.norm(query_vector)
	if norm > 0:
		query_vector = query_vector / norm

	scores = embeddings @ query_vector

	if k >= len(scores):
		top_k_indices = np.argsort(scores)[::-1]
	else:
		top_partition = np.argpartition(scores, -k)[-k:]
		top_k_indices = top_partition[np.argsort(scores[top_partition])[::-1]]

	return [
		{"id": doc_ids[idx], "score": float(scores[idx]), "rank": rank + 1}
		for rank, idx in enumerate(top_k_indices)
	]


def run_comparison(k: int = 10, pool_size: int = 20):
	print("=" * 75)
	print("       PRE- vs POST-FINE-TUNING RETRIEVAL BENCHMARK")
	print("=" * 75)

	# 1. Load Data
	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	queries = pd.read_parquet(f"{DATA_DIR}/queries.parquet")
	qrels = pd.read_parquet(f"{DATA_DIR}/qrels.parquet")

	corpus["_id"] = corpus["_id"].astype(str)
	queries["_id"] = queries["_id"].astype(str)
	qrels["query-id"] = qrels["query-id"].astype(str)
	qrels["corpus-id"] = qrels["corpus-id"].astype(str)

	corpus_lookup = dict(zip(corpus["_id"], corpus["text"].astype(str)))
	query_lookup = dict(zip(queries["_id"], queries["text"].astype(str)))

	# Load held-out test split
	split_file = "./data/splits/test_query_ids.csv"
	if os.path.exists(split_file):
		test_qids = pd.read_csv(split_file)["query_id"].astype(str).tolist()
		print(f"Loaded {len(test_qids)} held-out test queries from {split_file}")
	else:
		print("Test split file not found, evaluating on all labeled queries...")
		test_qids = qrels["query-id"].unique().tolist()

	# 2. Load Models & Indexes
	print("\nLoading Base Model and Index...")
	base_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
	base_embeddings, base_doc_ids = load_index(BASE_INDEX_DIR)

	print("Loading Fine-Tuned Model and Index...")
	if not os.path.exists(f"{FINETUNED_INDEX_DIR}/embeddings.npy"):
		print(f"Error: Fine-tuned index not found at {FINETUNED_INDEX_DIR}. Run train_biencoder.py first.")
		return

	ft_model = SentenceTransformer(FINETUNED_MODEL_DIR)
	ft_embeddings, ft_doc_ids = load_index(FINETUNED_INDEX_DIR)

	print("Loading BM25 index...")
	bm25_retriever = load_bm25_index()

	# 3. Evaluation Setup
	eval_modes = [
		"Dense (Base Pre-FT)",
		"Dense (Fine-Tuned Post-FT)",
		"Hybrid RRF (Base Pre-FT)",
		"Hybrid RRF (Fine-Tuned Post-FT)",
		"Hybrid + Rerank (Base Pre-FT)",
		"Hybrid + Rerank (Fine-Tuned Post-FT)",
	]

	scores = {mode: {"ndcg": [], "mrr": [], "recall": []} for mode in eval_modes}

	print(f"\nBenchmarking {len(test_qids)} held-out queries (k={k}, candidate pool={pool_size})...")
	t0 = time.perf_counter()

	for idx, qid in enumerate(test_qids, start=1):
		query_text = query_lookup.get(qid)
		if not query_text:
			continue

		relevant_ids = set(qrels[qrels["query-id"] == qid]["corpus-id"].tolist())
		if not relevant_ids:
			continue

		# BM25 Search
		bm25_hits = bm25_search(query_text, bm25_retriever, k=pool_size)
		bm25_ids = [h["id"] for h in bm25_hits]

		# Dense Search (Base)
		dense_base_hits = dense_search_with_model(query_text, base_embeddings, base_doc_ids, base_model, k=pool_size)
		dense_base_ids = [h["id"] for h in dense_base_hits]

		# Dense Search (Fine-Tuned)
		dense_ft_hits = dense_search_with_model(query_text, ft_embeddings, ft_doc_ids, ft_model, k=pool_size)
		dense_ft_ids = [h["id"] for h in dense_ft_hits]

		# Hybrid Fusions (50/50 RRF)
		hybrid_base_fused = [
			doc_id for doc_id, _ in weighted_rrf([bm25_ids, dense_base_ids], [0.50, 0.50])[:pool_size]
		]
		hybrid_ft_fused = [
			doc_id for doc_id, _ in weighted_rrf([bm25_ids, dense_ft_ids], [0.50, 0.50])[:pool_size]
		]

		# Re-ranked
		base_cand_texts = [corpus_lookup.get(doc_id, "") for doc_id in hybrid_base_fused]
		reranked_base = rerank_results(query_text, hybrid_base_fused, base_cand_texts, top_n=k)
		reranked_base_ids = [r["id"] for r in reranked_base]

		ft_cand_texts = [corpus_lookup.get(doc_id, "") for doc_id in hybrid_ft_fused]
		reranked_ft = rerank_results(query_text, hybrid_ft_fused, ft_cand_texts, top_n=k)
		reranked_ft_ids = [r["id"] for r in reranked_ft]

		# Record metrics
		runs = {
			"Dense (Base Pre-FT)": dense_base_ids[:k],
			"Dense (Fine-Tuned Post-FT)": dense_ft_ids[:k],
			"Hybrid RRF (Base Pre-FT)": hybrid_base_fused[:k],
			"Hybrid RRF (Fine-Tuned Post-FT)": hybrid_ft_fused[:k],
			"Hybrid + Rerank (Base Pre-FT)": reranked_base_ids[:k],
			"Hybrid + Rerank (Fine-Tuned Post-FT)": reranked_ft_ids[:k],
		}

		for mode, retrieved in runs.items():
			scores[mode]["ndcg"].append(ndcg_at_k(retrieved, relevant_ids, k=k))
			scores[mode]["mrr"].append(mrr_at_k(retrieved, relevant_ids, k=k))
			scores[mode]["recall"].append(recall_at_k(retrieved, relevant_ids, k=k))

		if idx % 25 == 0 or idx == len(test_qids):
			print(f"  Evaluated {idx}/{len(test_qids)} test queries...")

	total_eval_time = time.perf_counter() - t0
	print(f"\nEvaluation completed in {total_eval_time:.1f}s!\n")

	# 4. Print Summary Table
	print("=" * 85)
	print(f"{'Method / Configuration':<38} {'NDCG@10':>12} {'MRR@10':>12} {'Recall@10':>14}")
	print("=" * 85)

	def print_pair(base_name: str, ft_name: str, label: str):
		base_ndcg = np.mean(scores[base_name]["ndcg"]) * 100
		base_mrr = np.mean(scores[base_name]["mrr"]) * 100
		base_rec = np.mean(scores[base_name]["recall"]) * 100

		ft_ndcg = np.mean(scores[ft_name]["ndcg"]) * 100
		ft_mrr = np.mean(scores[ft_name]["mrr"]) * 100
		ft_rec = np.mean(scores[ft_name]["recall"]) * 100

		d_ndcg = ft_ndcg - base_ndcg
		d_mrr = ft_mrr - base_mrr
		d_rec = ft_rec - base_rec

		print(f"[{label}]")
		print(f"  Pre-Fine-Tuned (Base)             {base_ndcg:>11.2f}% {base_mrr:>11.2f}% {base_rec:>13.2f}%")
		print(f"  Post-Fine-Tuned                   {ft_ndcg:>11.2f}% {ft_mrr:>11.2f}% {ft_rec:>13.2f}%")
		print(
			f"  Delta Gain                        {f'{d_ndcg:+6.2f}%':>12} {f'{d_mrr:+6.2f}%':>12} {f'{d_rec:+6.2f}%':>14}")
		print("-" * 85)

	print_pair("Dense (Base Pre-FT)", "Dense (Fine-Tuned Post-FT)", "1. Pure Dense Retrieval")
	print_pair("Hybrid RRF (Base Pre-FT)", "Hybrid RRF (Fine-Tuned Post-FT)", "2. Hybrid BM25 + Dense (RRF)")
	print_pair("Hybrid + Rerank (Base Pre-FT)", "Hybrid + Rerank (Fine-Tuned Post-FT)",
			   "3. Full Pipeline (Hybrid + Cross-Encoder)")
	print("=" * 85)


if __name__ == "__main__":
	run_comparison(k=10, pool_size=20)
