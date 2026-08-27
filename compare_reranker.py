import os
import time

import bm25s
import numpy as np
import pandas as pd
from Stemmer import Stemmer
from dense_retriever import dense_search, load_dense_index
from evaluate import ndcg_at_k, mrr_at_k, recall_at_k
from rrf import weighted_rrf
from sentence_transformers import CrossEncoder

DATA_DIR = "./data"
BASE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
FINETUNED_MODEL_DIR = "./models/cross-encoder-finetuned"
DENSE_INDEX_DIR = "./indexes/dense"


def rerank_with_model(
		query: str,
		candidate_ids: list[str],
		candidate_texts: list[str],
		model: CrossEncoder,
		top_n: int = 10,
) -> list[dict]:
	pairs = [(query, text) for text in candidate_texts]
	scores = model.predict(pairs, show_progress_bar=False)
	ranked = sorted(
		enumerate(scores),
		key=lambda item: item[1],
		reverse=True
	)[:top_n]
	return [
		{"id": candidate_ids[original_idx], "score": float(score)}
		for original_idx, score in ranked
	]


def run_reranker_comparison(k: int = 10, pool_size: int = 20):
	print("=" * 85)
	print("       CROSS-ENCODER RERANKER BENCHMARK: PRE- vs POST-FINE-TUNING")
	print("=" * 85)

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

	# 2. Load Models
	print(f"Loading Base Cross-Encoder ({BASE_MODEL_NAME})...")
	base_reranker = CrossEncoder(BASE_MODEL_NAME)

	print(f"Loading Fine-Tuned Cross-Encoder from {FINETUNED_MODEL_DIR}...")
	if not os.path.exists(FINETUNED_MODEL_DIR):
		print(f"Error: Fine-tuned model not found at {FINETUNED_MODEL_DIR}. Run train_crossencoder.py first.")
		return
	ft_reranker = CrossEncoder(FINETUNED_MODEL_DIR)

	print("Loading retrieval indexes (BM25 + Dense Base)...")
	bm25_retriever = bm25s.BM25.load("./indexes/bm25", load_corpus=True)
	stemmer = Stemmer("english")
	dense_embeddings, dense_doc_ids = load_dense_index()

	# 3. Setup evaluation
	methods = [
		"1. Initial Retrieval (Hybrid RRF Top-20)",
		"2. Base Cross-Encoder (Pre-FT)",
		"3. Fine-Tuned Cross-Encoder (Post-FT)",
	]
	scores = {m: {"ndcg": [], "mrr": [], "recall": []} for m in methods}

	print(f"\nBenchmarking {len(test_qids)} held-out test queries (k={k}, candidate pool={pool_size})...\n")
	t0 = time.perf_counter()

	for idx, qid in enumerate(test_qids, start=1):
		query_text = query_lookup.get(qid)
		if not query_text:
			continue

		relevant_ids = set(qrels[qrels["query-id"] == qid]["corpus-id"].tolist())
		if not relevant_ids:
			continue

		# BM25 Search (silent)
		query_tokens = bm25s.tokenize(query_text, stopwords="en", stemmer=stemmer, show_progress=False)
		bm25_res, _ = bm25_retriever.retrieve(query_tokens, k=pool_size, show_progress=False)
		bm25_ids = [
			(doc.get("id") if isinstance(doc, dict) else str(doc))
			for doc in bm25_res[0]
		]

		# Dense Search
		dense_hits = dense_search(query_text, dense_embeddings, dense_doc_ids, k=pool_size)
		dense_ids = [h["id"] for h in dense_hits]

		hybrid_fused_ids = [
			doc_id for doc_id, _ in weighted_rrf([bm25_ids, dense_ids], [0.50, 0.50])[:pool_size]
		]
		cand_texts = [corpus_lookup.get(doc_id, "") for doc_id in hybrid_fused_ids]

		# 1. Initial Candidates without reranking
		initial_top_k = hybrid_fused_ids[:k]

		# 2. Base Cross-Encoder
		base_ranked = rerank_with_model(query_text, hybrid_fused_ids, cand_texts, base_reranker, top_n=k)
		base_top_k = [r["id"] for r in base_ranked]

		# 3. Fine-Tuned Cross-Encoder
		ft_ranked = rerank_with_model(query_text, hybrid_fused_ids, cand_texts, ft_reranker, top_n=k)
		ft_top_k = [r["id"] for r in ft_ranked]

		runs = {
			"1. Initial Retrieval (Hybrid RRF Top-20)": initial_top_k,
			"2. Base Cross-Encoder (Pre-FT)": base_top_k,
			"3. Fine-Tuned Cross-Encoder (Post-FT)": ft_top_k,
		}

		for m, retrieved in runs.items():
			scores[m]["ndcg"].append(ndcg_at_k(retrieved, relevant_ids, k=k))
			scores[m]["mrr"].append(mrr_at_k(retrieved, relevant_ids, k=k))
			scores[m]["recall"].append(recall_at_k(retrieved, relevant_ids, k=k))

		if idx % 25 == 0 or idx == len(test_qids):
			print(f"  Evaluated {idx:3d}/{len(test_qids)} queries...")

	eval_time = time.perf_counter() - t0
	print(f"\nBenchmark completed in {eval_time:.1f}s!\n")

	# 4. Print Summary Results
	print("=" * 85)
	print(f"{'Method / Configuration':<42} {'NDCG@10':>12} {'MRR@10':>12} {'Recall@10':>14}")
	print("=" * 85)

	init_ndcg = np.mean(scores["1. Initial Retrieval (Hybrid RRF Top-20)"]["ndcg"]) * 100
	init_mrr = np.mean(scores["1. Initial Retrieval (Hybrid RRF Top-20)"]["mrr"]) * 100
	init_rec = np.mean(scores["1. Initial Retrieval (Hybrid RRF Top-20)"]["recall"]) * 100

	base_ndcg = np.mean(scores["2. Base Cross-Encoder (Pre-FT)"]["ndcg"]) * 100
	base_mrr = np.mean(scores["2. Base Cross-Encoder (Pre-FT)"]["mrr"]) * 100
	base_rec = np.mean(scores["2. Base Cross-Encoder (Pre-FT)"]["recall"]) * 100

	ft_ndcg = np.mean(scores["3. Fine-Tuned Cross-Encoder (Post-FT)"]["ndcg"]) * 100
	ft_mrr = np.mean(scores["3. Fine-Tuned Cross-Encoder (Post-FT)"]["mrr"]) * 100
	ft_rec = np.mean(scores["3. Fine-Tuned Cross-Encoder (Post-FT)"]["recall"]) * 100

	print(f"{'1. Initial Hybrid RRF (No Rerank)':<42} {init_ndcg:>11.2f}% {init_mrr:>11.2f}% {init_rec:>13.2f}%")
	print(f"{'2. Base ms-marco Cross-Encoder (Pre-FT)':<42} {base_ndcg:>11.2f}% {base_mrr:>11.2f}% {base_rec:>13.2f}%")
	print(f"{'3. Fine-Tuned Cross-Encoder (Post-FT)':<42} {ft_ndcg:>11.2f}% {ft_mrr:>11.2f}% {ft_rec:>13.2f}%")
	print("-" * 85)

	d_ndcg_vs_base = ft_ndcg - base_ndcg
	d_mrr_vs_base = ft_mrr - base_mrr
	d_rec_vs_base = ft_rec - base_rec

	d_ndcg_vs_init = ft_ndcg - init_ndcg
	d_mrr_vs_init = ft_mrr - init_mrr
	d_rec_vs_init = ft_rec - init_rec

	print(f"{'Delta: Fine-Tuned vs Base Cross-Encoder':<42} {f'{d_ndcg_vs_base:+6.2f}%':>12} {f'{d_mrr_vs_base:+6.2f}%':>12} {f'{d_rec_vs_base:+6.2f}%':>14}")
	print(f"{'Delta: Fine-Tuned vs Initial Retrieval':<42} {f'{d_ndcg_vs_init:+6.2f}%':>12} {f'{d_mrr_vs_init:+6.2f}%':>12} {f'{d_rec_vs_init:+6.2f}%':>14}")
	print("=" * 85)


if __name__ == "__main__":
	run_reranker_comparison(k=10, pool_size=20)
