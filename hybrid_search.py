import numpy as np
import pandas as pd

from bm25_retriever import load_bm25_index, bm25_search
from dense_retriever import load_dense_index, dense_search
from reranker import rerank_results
from rrf import weighted_rrf

DATA_DIR = "./data"


def hybrid_search_full(
		query: str,
		bm25_retriever,
		dense_embeddings: np.ndarray,
		dense_doc_ids: list,
		corpus_lookup: dict,
		candidates_k: int = 50,
		final_k: int = 10,
		use_reranker: bool = True
) -> list[dict]:
	"""
	Full hybrid retrieval pipeline:
	1. BM25 sparse retrieval (top candidates_k)
	2. Dense semantic retrieval (top candidates_k)
	3. RRF fusion of both ranked lists
	4. Optional cross-encoder re-ranking (top final_k)
	"""
	# Stage 1 & 2: Cast a wide net with both methods
	bm25_results = bm25_search(query, bm25_retriever, k=candidates_k)
	dense_results = dense_search(query, dense_embeddings, dense_doc_ids, k=candidates_k)
	bm25_ids = [r["id"] for r in bm25_results]
	dense_ids = [r["id"] for r in dense_results]
	# Stage 3: Merge with RRF
	fused_results = reciprocal_rank_fusion([bm25_ids, dense_ids])
	fused_ids = [doc_id for doc_id, _ in fused_results]
	if not use_reranker:
		# Return RRF results directly
		return [
			{"id": doc_id, "score": score, "text": corpus_lookup.get(doc_id, "")[:300]}
			for doc_id, score in fused_results[:final_k]
		]
	# Stage 4: Re-rank the fused candidates with a cross-encoder
	candidate_ids = fused_ids[:candidates_k]
	candidate_texts = [corpus_lookup.get(doc_id, "") for doc_id in candidate_ids]
	reranked = rerank_results(query, candidate_ids, candidate_texts, top_n=final_k)
	return reranked


# --- Demo ---
if __name__ == "__main__":
	print("Loading indexes...")
	bm25_retriever = load_bm25_index()
	dense_embeddings, dense_doc_ids = load_dense_index()
	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	corpus['_id'] = corpus['_id'].astype(str)
	corpus_lookup = dict(zip(corpus['_id'], corpus['text'].astype(str)))
	query = "Where should I park my rainy day emergency funds?"
	print("\n=== BM25 ONLY (Top 5) ===")
	bm25_only = bm25_search(query, bm25_retriever, k=5)
	for r in bm25_only:
		print(f"  {r['text'][:120]}\n")
	print("\n=== DENSE ONLY (Top 5) ===")
	dense_only = dense_search(query, dense_embeddings, dense_doc_ids, k=5)
	for r in dense_only:
		print(f"  {corpus_lookup.get(r['id'], '')[:120]}\n")
	print("\n=== HYBRID + RRF (Top 5) ===")
	hybrid_rrf = hybrid_search_full(
		query, bm25_retriever,
		dense_embeddings, dense_doc_ids,
		corpus_lookup, use_reranker=False
	)
	for r in hybrid_rrf[:5]:
		print(f"  {r['text'][:120]}\n")
	print("\n=== HYBRID + RRF + RE-RANKER (Top 5) ===")
	hybrid_full = hybrid_search_full(
		query, bm25_retriever,
		dense_embeddings, dense_doc_ids,
		corpus_lookup, use_reranker=True
	)
	for r in hybrid_full[:5]:
		print(f"  Score: {r['relevance_score']:.4f}")
		print(f"  {r['text'][:120]}\n")
