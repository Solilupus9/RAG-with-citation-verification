import os

import bm25s
import pandas as pd

DATA_DIR = "./data"
INDEX_DIR = "./indexes/bm25"
os.makedirs(INDEX_DIR, exist_ok=True)


def build_bm25_index():
	"""Build and persist the BM25 index from the corpus."""
	print("Loading corpus...")
	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")

	# Convert DataFrame columns to plain Python lists
	# BM25s expects lists, not pandas Series
	doc_ids = corpus['_id'].astype(str).tolist()
	doc_texts = corpus['text'].astype(str).tolist()
	corpus_records = [{"id": doc_id, "text": doc_text} for doc_id, doc_text in zip(doc_ids, doc_texts)]
	print(f"Tokenizing {len(doc_texts):,} documents...")

	# Tokenize the entire corpus
	# stop_words="en" removes words like "the", "is", "at" that add noise
	corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en")

	# Build the retriever using the BM25 algorithm
	# method="lucene" uses the Lucene variant (robust default)
	retriever = bm25s.BM25(method="lucene", corpus=corpus_records)
	retriever.index(corpus_tokens)

	# Persist to disk — no database needed!
	# For ~57k short documents, this is only ~33MB
	retriever.save(INDEX_DIR, corpus=corpus_records)
	print(f"BM25 index saved to {INDEX_DIR}")
	index_size_bytes = sum(
		os.path.getsize(os.path.join(INDEX_DIR, filename))
		for filename in os.listdir(INDEX_DIR)
		if os.path.isfile(os.path.join(INDEX_DIR, filename))
	)
	print(f"Index size: ~{index_size_bytes / 1e6:.1f} MB")
	return retriever, corpus_records


def load_bm25_index():
	"""Load a previously built BM25 index from disk."""
	retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
	return retriever


def bm25_search(query: str, retriever, k: int = 10) -> list[dict]:
	"""
	Search the BM25 index for a given query.
	Returns top-k results as a list of dicts with id, score, text.
	"""
	# Tokenize the query the same way we tokenized the corpus
	# This is deterministic — same input always gives same output
	query_tokens = bm25s.tokenize(query, stopwords="en")

	# Retrieve top-k results
	# results[0] = document objects, results[1] = scores
	results, scores = retriever.retrieve(query_tokens, k=k)
	output = []
	for i in range(results.shape[1]):
		doc = results[0, i]
		score = scores[0, i]
		if isinstance(doc, dict):
			doc_id = doc.get("id")
			doc_text = doc.get("text", "")
		else:
			doc_id = None
			doc_text = doc if isinstance(doc, str) else str(doc)
		output.append({
			"id": doc_id,
			"score": float(score),
			"text": doc_text[:300] + ("..." if len(doc_text) > 300 else "")
		})
	return output


if __name__ == "__main__":
	# First run: build the index
	# Subsequent runs: load from disk
	if not os.path.isdir(INDEX_DIR) or not os.listdir(INDEX_DIR):
		retriever, corpus_records = build_bm25_index()
	else:
		retriever = load_bm25_index()
	query = "Where should I park my rainy day emergency funds?"
	results = bm25_search(query, retriever, k=5)
	print(f"\nBM25 results for: '{query}'\n")
	for rank, r in enumerate(results, 1):
		print(f"Rank {rank} | Score: {r['score']:.4f} | ID: {r['id']}")
		print(f"  {r['text'][:150]}\n")
