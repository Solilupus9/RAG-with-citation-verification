from typing import Optional

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"
rerank_model: Optional[CrossEncoder] = None


def get_rerank_model() -> CrossEncoder | None:
	global rerank_model
	if rerank_model is None:
		rerank_model = CrossEncoder(RERANK_MODEL)
	return rerank_model


def rerank_results(
		query: str,
		candidate_ids: list[str],
		candidate_texts: list[str],
		top_n: int = 10
) -> list[dict]:
	"""
	Re-rank candidates using a cross-encoder model.
	The cross-encoder looks at (query, document) pairs together,
	enabling much richer scoring than bi-encoder similarity.
	Args:
		query: The user's search query
		candidate_ids: Document IDs for the candidates
		candidate_texts: Document texts for the candidates
		top_n: How many results to return after re-ranking
	Returns:
		Re-ranked list of dicts with id, text, relevance_score
	"""
	model = get_rerank_model()
	pairs = [(query, text) for text in candidate_texts]
	scores = model.predict(pairs)
	ranked = sorted(
		enumerate(scores),
		key=lambda item: item[1],
		reverse=True
	)[:top_n]
	results = []
	for original_idx, score in ranked:
		results.append({
			"id": candidate_ids[original_idx],
			"text": candidate_texts[original_idx],
			"relevance_score": float(score)
		})
	return results
