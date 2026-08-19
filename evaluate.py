import random

import numpy as np
import pandas as pd

from bm25_retriever import load_bm25_index, bm25_search
from dense_retriever import load_dense_index, dense_search
from reranker import rerank_results
from rrf import weighted_rrf

DATA_DIR = "./data"
random.seed(42)  # Reproducibility


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
	"""
	Compute NDCG@k for a single query.
	Args:
		retrieved_ids: Ordered list of retrieved document IDs
		relevant_ids: Set of ground-truth relevant document IDs
		k: Cutoff rank
	"""
	# DCG: sum of relevance / log2(rank + 1) for top-k retrieved docs
	dcg = 0.0
	for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
		if doc_id in relevant_ids:
			dcg += 1.0 / np.log2(rank + 1)
	# Ideal DCG: what we'd get if all relevant docs were ranked first
	ideal_hits = min(len(relevant_ids), k)
	idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
	if idcg == 0:
		return 0.0
	return dcg / idcg


def evaluate_all_methods(n_queries: int = 50) -> dict:
	bm25_retriever = load_bm25_index()
	dense_embeddings, dense_doc_ids = load_dense_index()
	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	queries = pd.read_parquet(f"{DATA_DIR}/queries.parquet")
	qrels = pd.read_parquet(f"{DATA_DIR}/qrels.parquet")
	corpus['_id'] = corpus['_id'].astype(str)
	queries['_id'] = queries['_id'].astype(str)
	qrels['query-id'] = qrels['query-id'].astype(str)
	qrels['corpus-id'] = qrels['corpus-id'].astype(str)
	corpus_lookup = dict(zip(corpus['_id'], corpus['text'].astype(str)))

	valid_query_ids = set(qrels['query-id'].unique())
	eval_queries = queries[queries['_id'].isin(valid_query_ids)]
	sampled = eval_queries.sample(n=min(n_queries, len(eval_queries)), random_state=42)

	weight_config = {
		"rrf_w20_80": (0.20, 0.80),
		"rrf_w50_50": (0.50, 0.50),
	}
	POOL_SIZE = 20  # single source of truth for retrieval AND rerank pool

	scores = {"bm25_only": [], "dense_only": []}
	for name in weight_config:
		scores[f"{name}"] = []
		scores[f"{name}_rerank"] = []

	for i, (_, row) in enumerate(sampled.iterrows()):
		query_id = row['_id']
		query_text = row['text']

		relevant_ids: set[str] = set()
		for doc_id in qrels[qrels['query-id'] == query_id]['corpus-id'].tolist():
			relevant_ids.add(str(doc_id))
		if not relevant_ids:
			continue

		bm25_results = bm25_search(query_text, bm25_retriever, k=10)
		bm25_ids = [r["id"] for r in bm25_results]
		scores["bm25_only"].append(ndcg_at_k(bm25_ids, relevant_ids))

		dense_results = dense_search(query_text, dense_embeddings, dense_doc_ids, k=10)
		dense_ids = [r["id"] for r in dense_results]
		scores["dense_only"].append(ndcg_at_k(dense_ids, relevant_ids))

		for name, (bm25_w, dense_w) in weight_config.items():
			bm25_pool = [r["id"] for r in bm25_search(query_text, bm25_retriever, k=POOL_SIZE)]
			dense_pool = [r["id"] for r in dense_search(query_text, dense_embeddings, dense_doc_ids, k=POOL_SIZE)]

			fused = [
				doc_id for doc_id, _ in weighted_rrf([bm25_pool, dense_pool], [bm25_w, dense_w])[:10]
			]
			scores[f"{name}"].append(ndcg_at_k(fused, relevant_ids))

			fused_pool = [
				doc_id for doc_id, _ in weighted_rrf([bm25_pool, dense_pool], [bm25_w, dense_w])[:POOL_SIZE]
			]
			candidate_texts = [corpus_lookup.get(doc_id, "") for doc_id in fused_pool]
			reranked = rerank_results(query_text, fused_pool, candidate_texts, top_n=10)
			reranked_ids = [r["id"] for r in reranked]
			scores[f"{name}_rerank"].append(ndcg_at_k(reranked_ids, relevant_ids))

		if (i + 1) % 10 == 0:
			print(f"  Evaluated {i + 1}/{len(sampled)} queries...")

	results = {}
	for method, method_scores in scores.items():
		if not method_scores:
			continue
		arr = np.array(method_scores) * 100
		results[method] = {
			"mean": round(float(np.mean(arr)), 1),
			"std": round(float(np.std(arr)), 1),
			"n": len(arr)
		}
	return results


# --- Run the evaluation ---
if __name__ == "__main__":
	print("Running evaluation on 200 queries...")
	results = evaluate_all_methods(n_queries=200)

	print("\n" + "=" * 50)
	print("EVALUATION RESULTS (NDCG@10, higher = better)")
	print("=" * 50)
	for method, stats in results.items():
		mean = stats["mean"]
		std = stats["std"]
		bar = "█" * int(mean / 2)
		print(f"{method:<25} {mean:5.1f} ± {std:4.1f}  {bar}")
	print("=" * 50)