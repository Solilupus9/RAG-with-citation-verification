import pandas as pd

from bm25_retriever import load_bm25_index
from citation_checks_simple import validate_citation_numbers
from citation_verifier import verify_answer
from dense_retriever import load_dense_index
from generator import generate_answer
from hybrid_search import hybrid_search_full

DATA_DIR = "./data"


def main():
    query = input("Question: ").strip()

    bm25_retriever = load_bm25_index()
    dense_embeddings, dense_doc_ids = load_dense_index()

    corpus = pd.read_parquet(
        f"{DATA_DIR}/corpus.parquet"
    )
    corpus["_id"] = corpus["_id"].astype(str)

    corpus_lookup = dict(
        zip(
            corpus["_id"],
            corpus["text"].astype(str),
        )
    )

    retrieved_docs = hybrid_search_full(
        query=query,
        bm25_retriever=bm25_retriever,
        dense_embeddings=dense_embeddings,
        dense_doc_ids=dense_doc_ids,
        corpus_lookup=corpus_lookup,
        candidates_k=20,
        final_k=5,
        use_reranker=True,
    )

    result = generate_answer(
        query=query,
        retrieved_docs=retrieved_docs,
    )

    answer = result["answer"]
    documents = result["documents"]

    citation_status = validate_citation_numbers(
        answer=answer,
        number_of_documents=len(documents),
    )

    verification = verify_answer(
        answer=answer,
        retrieved_docs=documents,
    )

    print("\nANSWER\n")
    print(answer)

    print("\nCITATION VERIFICATION\n")
    faithfulness = verification["faithfulness"]
    if faithfulness is None:
        print("No cited factual claims could be checked.")
    else:
        print(
            f"Faithfulness: {faithfulness:.2%} "
            f"({verification['supported_claims']}/"
            f"{verification['checkable_claims']} supported)"
        )

    if not citation_status["all_citations_valid"]:
        print("WARNING: invalid citation numbers:")
        print(citation_status["invalid_citations"])

    for claim in verification["claims"]:
        verdict = claim["verdict"]
        sentence = claim["sentence"]
        print(f"[{verdict}] {sentence}")

        if "entailment_score" in claim:
            print(
                "  scores: "
                f"entailment={claim['entailment_score']:.3f}, "
                f"contradiction={claim['contradiction_score']:.3f}, "
                f"neutral={claim['neutral_score']:.3f}"
            )

        if "best_document" in claim:
            print(
                "  best document:",
                claim["best_document"],
                "chunk:",
                claim.get("best_chunk"),
            )

        

    if faithfulness is not None and faithfulness < 0.80:
        print(
            "\nWARNING: Some claims were not sufficiently supported "
            "by their cited documents."
        )

    print("\nSOURCES\n")
    for index, document in enumerate(documents, start=1):
        print(f"[Document {index}] {document['id']}")


if __name__ == "__main__":
    main()
