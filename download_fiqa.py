from datasets import load_dataset
import os

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
print("Loading BEIR fiqa dataset...")

# Load all three components
corpus_dataset = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries_dataset = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels_dataset = load_dataset("BeIR/fiqa-qrels", split="test")

# Convert to DataFrames and save as Parquet (efficient binary format)
corpus_df = corpus_dataset.to_pandas()
queries_df = queries_dataset.to_pandas()
qrels_df = qrels_dataset.to_pandas()
corpus_df.to_parquet(f"{DATA_DIR}/corpus.parquet", index=False)
queries_df.to_parquet(f"{DATA_DIR}/queries.parquet", index=False)
qrels_df.to_parquet(f"{DATA_DIR}/qrels.parquet", index=False)

print(f"Corpus: {len(corpus_df):,} documents")
print(f"Queries: {len(queries_df):,} queries")
print(f"QRels: {len(qrels_df):,} relevance pairs")