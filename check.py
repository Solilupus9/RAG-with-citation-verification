from dense_retriever import load_dense_index, dense_search

dense_embeddings, dense_doc_ids = load_dense_index()
query = "Where should I park my rainy day emergency funds?"

for pool_size in [20, 50, 100]:
    dense_ids = [r["id"] for r in dense_search(query, dense_embeddings, dense_doc_ids, k=pool_size)]
    print(pool_size, dense_ids[:10])