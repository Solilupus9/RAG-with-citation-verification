import os
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
EMBED_DIR = "./indexes/dense"
os.makedirs(EMBED_DIR, exist_ok=True)
# dense_retriever.py
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
BATCH_SIZE = 256
model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer | None:
	global model
	if model is None:
		model = SentenceTransformer(EMBEDDING_MODEL)
	return model



def create_embeddings(texts: list[object]) -> np.ndarray:
	"""
	Create embeddings for a list of texts in batches.
	Returns a numpy array of shape (n_texts, EMBEDDING_DIM).
	"""
	if not texts:
		return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
	embedding_model = get_embedding_model()
	all_embeddings = []
	for i in range(0, len(texts), BATCH_SIZE):
		batch = [str(text) for text in texts[i:i + BATCH_SIZE]]
		batch_embeddings = embedding_model.encode(
			batch,
			batch_size=BATCH_SIZE,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=False
		).astype(np.float32)
		all_embeddings.append(batch_embeddings)
		if (i // BATCH_SIZE) % 10 == 0:
			print(f"  Embedded {i + len(batch):,}/{len(texts):,} documents...")
	return np.vstack(all_embeddings)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
	"""
	L2-normalize vectors so each has unit length.
	After normalization: cosine_similarity(a, b) == dot_product(a, b)
	This is crucial — it's what makes our matrix multiplication valid.
	"""
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	# Avoid division by zero for zero vectors
	norms = np.where(norms == 0, 1, norms)
	return vectors / norms


def build_dense_index():
	"""Build and persist the dense embedding index from the corpus."""
	corpus = pq.read_table(f"{DATA_DIR}/corpus.parquet")
	doc_ids = [str(doc_id) for doc_id in corpus.column("_id").to_pylist()]
	doc_texts = [str(text) for text in corpus.column("text").to_pylist()]  # pyright: ignore[reportGeneralTypeIssues]
	print(f"Creating embeddings for {len(doc_texts):,} documents...")
	embeddings = create_embeddings(doc_texts)
	# Normalize for efficient cosine similarity via dot product
	embeddings_normalized = normalize_vectors(embeddings)
	# Save both the embeddings matrix and doc IDs
	np.save(f"{EMBED_DIR}/embeddings.npy", embeddings_normalized)
	np.save(f"{EMBED_DIR}/doc_ids.npy", np.array(doc_ids))
	size_mb = embeddings_normalized.nbytes / 1e6
	print(f"Dense index saved. Size: {size_mb:.0f} MB for {len(doc_ids):,} documents.")
	return embeddings_normalized, doc_ids


def load_dense_index():
	"""Load a previously built dense index from disk."""
	embeddings = np.load(f"{EMBED_DIR}/embeddings.npy")
	doc_ids = np.load(f"{EMBED_DIR}/doc_ids.npy", allow_pickle=True).tolist()
	return embeddings, doc_ids


def dense_search(
		query: str,
		embeddings: np.ndarray,
		doc_ids: list,
		k: int = 10
) -> list[dict]:
	"""
	Search the dense index using cosine similarity.
	Because embeddings are normalized, this is just a dot product.
	"""
	# Embed the query
	embedding_model = get_embedding_model()
	query_vector = embedding_model.encode(
		[query],
		convert_to_numpy=True,
		normalize_embeddings=True,
		show_progress_bar=False
	)[0].astype(np.float32)
	# Normalize query vector too
	query_vector = normalize_vectors(query_vector.reshape(1, -1)).flatten()
	# Compute cosine similarity for all documents at once
	# embeddings shape: (n_docs, EMBEDDING_DIM)
	# query_vector shape: (EMBEDDING_DIM,)
	# Result: (n_docs,) — one score per document
	# This is a single matrix-vector multiplication — extremely fast with numpy
	scores = embeddings @ query_vector
	# Get indices of top-k scores (argsort returns ascending, so we reverse)
	top_k_indices = np.argsort(scores)[::-1][:k]
	return [
		{"id": doc_ids[idx], "score": float(scores[idx]), "rank": rank + 1}
		for rank, idx in enumerate(top_k_indices)
	]


# --- Demo ---
if __name__ == "__main__":
	embed_path = f"{EMBED_DIR}/embeddings.npy"
	if not os.path.exists(embed_path):
		embeddings, doc_ids = build_dense_index()
	else:
		print("Loading existing dense index...")
		embeddings, doc_ids = load_dense_index()
	query = "Where should I park my rainy day emergency funds?"
	results = dense_search(query, embeddings, doc_ids, k=5)
	print(f"\nDense results for: '{query}'\n")
	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	for r in results:
		doc_text = corpus[corpus['_id'] == r['id']]['text'].values[0]
		print(f"Rank {r['rank']} | Score: {r['score']:.4f} | ID: {r['id']}")
		print(f"  {doc_text[:200]}\n")
