# explore_data.py
import pandas as pd

DATA_DIR = "./data"

# Load the three files
corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
queries = pd.read_parquet(f"{DATA_DIR}/queries.parquet")
qrels = pd.read_parquet(f"{DATA_DIR}/qrels.parquet")

# Normalize ID types across datasets. The parquet files use string IDs for
# the corpus and query payloads, but integer IDs for qrels references.
corpus['_id'] = corpus['_id'].astype(str)
queries['_id'] = queries['_id'].astype(str)
qrels['query-id'] = qrels['query-id'].astype(str)
qrels['corpus-id'] = qrels['corpus-id'].astype(str)

# --- Corpus ---
print("=== CORPUS (first document) ===")
print(corpus['text'].iloc[0])
# Output: A long financial discussion about investment strategies...

# --- Queries ---
print("\n=== SAMPLE QUERIES ===")
for text in queries['text'].sample(5, random_state=42):
    print(f"  • {text}")
# Output:
#   • Where should I park my rainy day/emergency funds?
#   • What is considered a business expense on a business trip?
#   • Starting a new business online
#   • New business owner, how did Texas work for the business vs individual?

# --- QRels (the critical piece) ---
print("\n=== HOW QRELS WORK ===")
# Filter queries that have ground-truth documents.
valid_query_ids = set(qrels['query-id'].unique())
filtered_queries = queries[queries['_id'].isin(valid_query_ids)]
print(f"Queries with ground-truth docs: {len(filtered_queries)} (of {len(queries)} total)")

if filtered_queries.empty:
    print("No queries with ground-truth documents were found.")
    raise SystemExit(0)

# Inspect one query-document relationship
sample_query = filtered_queries.iloc[0]
print(f"\nQuery: '{sample_query['text']}'")
relevant_doc_ids = qrels[qrels['query-id'] == sample_query['_id']]['corpus-id'].tolist()
print(f"Relevant document IDs: {relevant_doc_ids}")

for doc_id in relevant_doc_ids[:2]:
    doc_row = corpus[corpus['_id'] == doc_id]
    if doc_row.empty:
        print(f"\nDoc {doc_id}: not found in corpus")
        continue
    doc_text = doc_row['text'].iloc[0]
    print(f"\nDoc {doc_id}: {doc_text[:200]}...")