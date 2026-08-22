import argparse
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from bm25_retriever import load_bm25_index, bm25_search
from citation_checks_simple import validate_citation_numbers
from citation_verifier import verify_answer
from dense_retriever import load_dense_index, dense_search
from generator import generate_answer, self_correct_answer
from hybrid_search import hybrid_search_full
from reranker import rerank_results
from rrf import weighted_rrf

DATA_DIR = "./data"
random.seed(42)


# =====================================================================
# 1. Retrieval Metrics
# =====================================================================

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
	"""
	Compute Normalized Discounted Cumulative Gain (NDCG@k).
	"""
	dcg = 0.0
	for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
		if doc_id in relevant_ids:
			dcg += 1.0 / np.log2(rank + 1)
	ideal_hits = min(len(relevant_ids), k)
	idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
	if idcg == 0:
		return 0.0
	return dcg / idcg


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
	"""
	Compute Mean Reciprocal Rank (MRR@k).
	"""
	for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
		if doc_id in relevant_ids:
			return 1.0 / rank
	return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
	"""
	Compute Recall@k (percentage of ground-truth relevant docs retrieved in top-k).
	"""
	if not relevant_ids:
		return 0.0
	hits = len(set(retrieved_ids[:k]).intersection(relevant_ids))
	return hits / len(relevant_ids)


# =====================================================================
# 2. Retrieval Evaluation Suite
# =====================================================================

def evaluate_retrieval(n_queries: int = 50, k: int = 10) -> dict:
	"""
	Benchmark BM25, Dense, RRF Hybrid, and Re-ranked pipelines on NDCG@k, MRR@k, and Recall@k.
	"""
	print(f"\n--- Loading Corpus and Indexes for Retrieval Benchmark ({n_queries} queries) ---")
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
	POOL_SIZE = 20

	methods = ["bm25_only", "dense_only"]
	for name in weight_config:
		methods.extend([name, f"{name}_rerank"])

	metrics = {m: {"ndcg": [], "mrr": [], "recall": []} for m in methods}

	for i, (_, row) in enumerate(sampled.iterrows()):
		query_id = row['_id']
		query_text = row['text']

		relevant_ids = set(
			str(cid)
			for cid in qrels[qrels['query-id'] == query_id]['corpus-id'].tolist()
		)
		if not relevant_ids:
			continue

		# 1. BM25
		bm25_results = bm25_search(query_text, bm25_retriever, k=POOL_SIZE)
		bm25_ids = [r["id"] for r in bm25_results]
		metrics["bm25_only"]["ndcg"].append(ndcg_at_k(bm25_ids, relevant_ids, k=k))
		metrics["bm25_only"]["mrr"].append(mrr_at_k(bm25_ids, relevant_ids, k=k))
		metrics["bm25_only"]["recall"].append(recall_at_k(bm25_ids, relevant_ids, k=k))

		# 2. Dense
		dense_results = dense_search(query_text, dense_embeddings, dense_doc_ids, k=POOL_SIZE)
		dense_ids = [r["id"] for r in dense_results]
		metrics["dense_only"]["ndcg"].append(ndcg_at_k(dense_ids, relevant_ids, k=k))
		metrics["dense_only"]["mrr"].append(mrr_at_k(dense_ids, relevant_ids, k=k))
		metrics["dense_only"]["recall"].append(recall_at_k(dense_ids, relevant_ids, k=k))

		# 3. Hybrid RRF & Rerank
		for name, (bm25_w, dense_w) in weight_config.items():
			fused = [
				doc_id
				for doc_id, _ in weighted_rrf([bm25_ids, dense_ids], [bm25_w, dense_w])[:POOL_SIZE]
			]
			metrics[name]["ndcg"].append(ndcg_at_k(fused, relevant_ids, k=k))
			metrics[name]["mrr"].append(mrr_at_k(fused, relevant_ids, k=k))
			metrics[name]["recall"].append(recall_at_k(fused, relevant_ids, k=k))

			# Cross-encoder Re-ranking
			candidate_texts = [corpus_lookup.get(doc_id, "") for doc_id in fused]
			reranked = rerank_results(query_text, fused, candidate_texts, top_n=k)
			reranked_ids = [r["id"] for r in reranked]
			metrics[f"{name}_rerank"]["ndcg"].append(ndcg_at_k(reranked_ids, relevant_ids, k=k))
			metrics[f"{name}_rerank"]["mrr"].append(mrr_at_k(reranked_ids, relevant_ids, k=k))
			metrics[f"{name}_rerank"]["recall"].append(recall_at_k(reranked_ids, relevant_ids, k=k))

		if (i + 1) % 10 == 0:
			print(f"  Processed {i + 1}/{len(sampled)} retrieval queries...")

	summary = {}
	for method, scores in metrics.items():
		summary[method] = {
			"NDCG@10": round(float(np.mean(scores["ndcg"])) * 100, 2),
			"MRR@10": round(float(np.mean(scores["mrr"])) * 100, 2),
			"Recall@10": round(float(np.mean(scores["recall"])) * 100, 2),
			"n": len(scores["ndcg"])
		}

	return summary


# =====================================================================
# 3. End-to-End Generation & Verification Evaluation
# =====================================================================

def evaluate_generation_and_verification(
		n_queries: int = 10,
		candidates_k: int = 20,
		final_k: int = 5,
		run_self_correction: bool = True,
) -> dict:
	"""
	Benchmark the full end-to-end RAG pipeline:
	- Retrieval -> LLM Generation -> NLI Verification -> Self-Correction Loop
	Measures faithfulness, citation recall, contradiction rates, and self-correction delta.
	"""
	print(f"\n--- Running End-to-End RAG Benchmark ({n_queries} queries) ---")
	bm25_retriever = load_bm25_index()
	dense_embeddings, dense_doc_ids = load_dense_index()
	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	queries = pd.read_parquet(f"{DATA_DIR}/queries.parquet")
	corpus['_id'] = corpus['_id'].astype(str)
	queries['_id'] = queries['_id'].astype(str)
	corpus_lookup = dict(zip(corpus['_id'], corpus['text'].astype(str)))

	sampled_queries = queries.sample(n=min(n_queries, len(queries)), random_state=42)

	e2e_stats = {
		"total_queries": len(sampled_queries),
		"initial_faithfulness": [],
		"corrected_faithfulness": [],
		"citation_coverage": [],
		"valid_citations_rate": [],
		"contradiction_rate": [],
		"corrections_triggered": 0,
		"retrieval_latencies": [],
		"generation_latencies": [],
		"verification_latencies": [],
		"correction_latencies": [],
	}

	for idx, (_, row) in enumerate(sampled_queries.iterrows(), start=1):
		query_text = row['text']
		print(f"\n[{idx}/{len(sampled_queries)}] Query: {query_text[:70]}...")

		# 1. Retrieval
		t0 = time.perf_counter()
		retrieved_docs = hybrid_search_full(
			query=query_text,
			bm25_retriever=bm25_retriever,
			dense_embeddings=dense_embeddings,
			dense_doc_ids=dense_doc_ids,
			corpus_lookup=corpus_lookup,
			candidates_k=candidates_k,
			final_k=final_k,
			use_reranker=True,
		)
		t_ret = time.perf_counter() - t0
		e2e_stats["retrieval_latencies"].append(t_ret)

		# 2. Generation
		t0 = time.perf_counter()
		result = generate_answer(query=query_text, retrieved_docs=retrieved_docs)
		t_gen = time.perf_counter() - t0
		e2e_stats["generation_latencies"].append(t_gen)

		answer = result["answer"]
		documents = result["documents"]

		# 3. Citation Checks & Verification
		t0 = time.perf_counter()
		citation_status = validate_citation_numbers(answer=answer, number_of_documents=len(documents))
		verification = verify_answer(answer=answer, retrieved_docs=documents)
		t_ver = time.perf_counter() - t0
		e2e_stats["verification_latencies"].append(t_ver)

		total_claims = verification["total_claims"]
		checkable_claims = verification["checkable_claims"]
		supported_claims = verification["supported_claims"]
		contradicted_claims = sum(1 for c in verification["claims"] if c["verdict"] == "CONTRADICTED")
		no_citation_claims = sum(1 for c in verification["claims"] if c["verdict"] == "NO_CITATION")

		# Coverage & Validity
		cov = (total_claims - no_citation_claims) / total_claims if total_claims > 0 else 1.0
		val = 1.0 if citation_status["all_citations_valid"] else 0.0
		init_faith = verification["faithfulness"] if verification["faithfulness"] is not None else 1.0
		contra_rate = contradicted_claims / total_claims if total_claims > 0 else 0.0

		e2e_stats["citation_coverage"].append(cov)
		e2e_stats["valid_citations_rate"].append(val)
		e2e_stats["initial_faithfulness"].append(init_faith)
		e2e_stats["contradiction_rate"].append(contra_rate)

		print(f"  Init Faithfulness: {init_faith:.1%} | Claims: {supported_claims}/{checkable_claims} supported | Gen: {t_gen:.1f}s | Ver: {t_ver:.1f}s")

		# 4. Self-Correction Loop
		unverified_claims = [
			c for c in verification["claims"]
			if c["verdict"] in {"UNSUPPORTED", "CONTRADICTED", "INVALID_CITATION", "NO_CITATION"}
		]

		if run_self_correction and (unverified_claims or init_faith < 0.80):
			e2e_stats["corrections_triggered"] += 1
			t0 = time.perf_counter()
			corrected_res = self_correct_answer(
				query=query_text,
				initial_answer=answer,
				unverified_claims=unverified_claims,
				retrieved_docs=documents,
			)
			t_corr = time.perf_counter() - t0
			e2e_stats["correction_latencies"].append(t_corr)

			corrected_ver = verify_answer(
				answer=corrected_res["answer"],
				retrieved_docs=documents,
			)
			corr_faith = corrected_ver["faithfulness"] if corrected_ver["faithfulness"] is not None else 1.0
			e2e_stats["corrected_faithfulness"].append(corr_faith)
			print(f"  -> Corrected Faithfulness: {corr_faith:.1%} (Gain: {corr_faith - init_faith:+.1%}) in {t_corr:.1f}s")
		else:
			e2e_stats["corrected_faithfulness"].append(init_faith)

	return e2e_stats


# =====================================================================
# 4. Display Formatter & CLI Runner
# =====================================================================

def print_retrieval_table(results: dict):
	print("\n" + "=" * 65)
	print("RETRIEVAL EVALUATION RESULTS")
	print("=" * 65)
	print(f"{'Method':<25} {'NDCG@10':>10} {'MRR@10':>10} {'Recall@10':>12}")
	print("-" * 65)
	for method, metrics in results.items():
		print(f"{method:<25} {metrics['NDCG@10']:>9.2f}% {metrics['MRR@10']:>9.2f}% {metrics['Recall@10']:>11.2f}%")
	print("=" * 65)


def print_e2e_table(stats: dict):
	init_faith = np.mean(stats["initial_faithfulness"]) * 100
	corr_faith = np.mean(stats["corrected_faithfulness"]) * 100
	cov = np.mean(stats["citation_coverage"]) * 100
	val = np.mean(stats["valid_citations_rate"]) * 100
	contra = np.mean(stats["contradiction_rate"]) * 100

	avg_t_ret = np.mean(stats["retrieval_latencies"]) if stats["retrieval_latencies"] else 0.0
	avg_t_gen = np.mean(stats["generation_latencies"]) if stats["generation_latencies"] else 0.0
	avg_t_ver = np.mean(stats["verification_latencies"]) if stats["verification_latencies"] else 0.0
	avg_t_corr = np.mean(stats["correction_latencies"]) if stats["correction_latencies"] else 0.0

	print("\n" + "=" * 65)
	print("END-TO-END RAG & VERIFICATION METRICS")
	print("=" * 65)
	print(f"Evaluated Queries:              {stats['total_queries']}")
	print(f"Corrections Triggered:          {stats['corrections_triggered']}/{stats['total_queries']} ({stats['corrections_triggered'] / stats['total_queries']:.1%})")
	print("-" * 65)
	print(f"Citation Coverage Rate:         {cov:6.2f}%  (claims with citations)")
	print(f"Citation Number Validity:       {val:6.2f}%  (in range 1..K)")
	print(f"Hallucination/Contradiction:    {contra:6.2f}%  (contradicted by source)")
	print("-" * 65)
	print(f"Initial Answer Faithfulness:    {init_faith:6.2f}%")
	print(f"Final Answer Faithfulness:      {corr_faith:6.2f}%  (Change: {corr_faith - init_faith:+5.2f}%)")
	print("-" * 65)
	print("Average Latency Breakdown:")
	print(f"  Retrieval & Re-rank:          {avg_t_ret:6.2f}s")
	print(f"  LLM Answer Generation:        {avg_t_gen:6.2f}s")
	print(f"  NLI Verification:             {avg_t_ver:6.2f}s")
	if stats["corrections_triggered"] > 0:
		print(f"  Self-Correction Pass:         {avg_t_corr:6.2f}s")
	print("=" * 65)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline and Citation Verification")
	parser.add_argument("--mode", choices=["retrieval", "e2e", "all"], default="all", help="Evaluation mode")
	parser.add_argument("--n_retrieval", type=int, default=50, help="Number of queries for retrieval evaluation")
	parser.add_argument("--n_e2e", type=int, default=5, help="Number of queries for end-to-end evaluation")
	args = parser.parse_args()

	if args.mode in {"retrieval", "all"}:
		retrieval_results = evaluate_retrieval(n_queries=args.n_retrieval)
		print_retrieval_table(retrieval_results)

	if args.mode in {"e2e", "all"}:
		e2e_results = evaluate_generation_and_verification(n_queries=args.n_e2e)
		print_e2e_table(e2e_results)
