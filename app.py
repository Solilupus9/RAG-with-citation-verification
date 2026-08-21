import pandas as pd

from bm25_retriever import load_bm25_index
from citation_checks_simple import validate_citation_numbers
from citation_verifier import verify_answer
from dense_retriever import load_dense_index
from generator import generate_answer, self_correct_answer
from hybrid_search import hybrid_search_full

DATA_DIR = "./data"


def print_verification_report(verification: dict, citation_status: dict):
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
        print("WARNING: invalid citation numbers:", citation_status["invalid_citations"])

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


def main():
    query = input("Question: ").strip()
    if not query:
        print("Empty query provided. Exiting.")
        return

    print("\nLoading search indexes...")
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

    print("Retrieving and re-ranking documents...")
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

    print("Generating initial answer...")
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

    print("\n" + "=" * 50)
    print("INITIAL ANSWER")
    print("=" * 50)
    print(answer)

    print("\n" + "-" * 50)
    print("INITIAL CITATION VERIFICATION")
    print("-" * 50)
    print_verification_report(verification, citation_status)

    # Check for unverified claims or citation issues
    unverified_claims = [
        claim
        for claim in verification["claims"]
        if claim["verdict"] in {"UNSUPPORTED", "CONTRADICTED", "INVALID_CITATION", "NO_CITATION"}
    ]
    faithfulness = verification["faithfulness"]

    # Trigger self-correction loop if needed
    if unverified_claims or (faithfulness is not None and faithfulness < 0.80):
        print("\n" + "=" * 50)
        print("TRIGGERING SELF-CORRECTION LOOP (Self-Reflective RAG)...")
        print("=" * 50)

        corrected_result = self_correct_answer(
            query=query,
            initial_answer=answer,
            unverified_claims=unverified_claims,
            retrieved_docs=documents,
        )

        corrected_answer = corrected_result["answer"]
        corrected_citation_status = validate_citation_numbers(
            answer=corrected_answer,
            number_of_documents=len(documents),
        )
        corrected_verification = verify_answer(
            answer=corrected_answer,
            retrieved_docs=documents,
        )

        print("\nCORRECTED ANSWER\n")
        print(corrected_answer)

        print("\nCORRECTED CITATION VERIFICATION\n")
        print_verification_report(corrected_verification, corrected_citation_status)

        # Update output variables
        answer = corrected_answer
        verification = corrected_verification

    print("\n" + "=" * 50)
    print("SOURCES")
    print("=" * 50)
    for index, document in enumerate(documents, start=1):
        print(f"[Document {index}] ID: {document['id']}")
        print(f"  {document['text'][:150]}...\n")


if __name__ == "__main__":
    main()
