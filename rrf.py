from collections import defaultdict

RRF_K = 60  # Standard smoothing constant from the original paper


def reciprocal_rank_fusion(
		ranked_lists: list[list[str]],
		k: int = RRF_K
) -> list[tuple[str, float]]:
	"""
	Merge multiple ranked lists using Reciprocal Rank Fusion.
	Args:
		ranked_lists: List of ranked document ID lists.
					 e.g., [bm25_ids, dense_ids]
		k: Smoothing constant (default 60, from original paper)
	Returns:
		Sorted list of (doc_id, rrf_score) tuples, highest score first.
	"""
	scores = defaultdict(float)
	for ranked_list in ranked_lists:
		for rank, doc_id in enumerate(ranked_list, start=1):
			# The core RRF formula
			scores[doc_id] += 1.0 / (k + rank)
	# Sort by descending score
	return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def weighted_rrf(
		ranked_lists: list[list[str]],
		weights: tuple[float],
		k: int = RRF_K
) -> list[tuple[str, float]]:
	"""
	Weighted Reciprocal Rank Fusion.
	Args:
		ranked_lists: List of ranked document ID lists, e.g. [bm25_ids, dense_ids]
		weights: Weight for each list, same length and order as ranked_lists
		k: Smoothing constant
	Returns:
		Sorted list of (doc_id, weighted_rrf_score) tuples, highest first.
	"""
	if len(ranked_lists) != len(weights):
		raise ValueError("ranked_lists and weights must be the same length")

	scores = defaultdict(float)
	for weight, ranked_list in zip(weights, ranked_lists):
		for rank, doc_id in enumerate(ranked_list, start=1):
			scores[doc_id] += weight / (k + rank)

	return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# --- Demo ---
if __name__ == "__main__":
	# Simulate two ranked lists
	bm25_results = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
	dense_results = ["doc_D", "doc_B", "doc_F", "doc_A", "doc_G"]
	fused = reciprocal_rank_fusion([bm25_results, dense_results])
	print("RRF Fusion Results:")
	for rank, (doc_id, score) in enumerate(fused, 1):
		bm25_rank = bm25_results.index(doc_id) + 1 if doc_id in bm25_results else None
		dense_rank = dense_results.index(doc_id) + 1 if doc_id in dense_results else None
		print(f"  Rank {rank}: {doc_id} | RRF={score:.4f} | "
		      f"BM25={bm25_rank or 'N/A'} | Dense={dense_rank or 'N/A'}")
