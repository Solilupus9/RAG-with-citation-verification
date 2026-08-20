import re
from typing import Optional

import torch
from sentence_transformers import CrossEncoder


NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAILMENT_THRESHOLD = 0.60
CONTRADICTION_THRESHOLD = 0.60

nli_model: Optional[CrossEncoder] = None
nli_labels: Optional[list[str]] = None


def get_nli_model() -> CrossEncoder:
    global nli_model, nli_labels

    if nli_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        nli_model = CrossEncoder(
            NLI_MODEL,
            max_length=512,
            device=device,
        )

        id2label = nli_model.model.config.id2label

        nli_labels = [
            id2label[index].lower()
            for index in range(nli_model.num_labels)
        ]

        expected = {
            "contradiction",
            "entailment",
            "neutral",
        }

        if set(nli_labels) != expected:
            raise ValueError(
                f"Unexpected NLI labels: {nli_labels}"
            )

        print(f"NLI verifier device: {device}")
        print(f"NLI label order: {nli_labels}")


    return nli_model


def split_claims(text: str) -> list[str]:
    """
    Split normal paragraphs and Markdown bullet items into claims.
    """
    normalized = text.replace("\r\n", "\n").strip()

    # Treat Markdown bullets as separate claim boundaries.
    normalized = re.sub(
        r"(?m)^\s*[*-]\s+",
        "\n",
        normalized,
    )

    # Split after sentence punctuation, including before Markdown bullets.
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        normalized,
    )

    return [
        re.sub(r"\s+", " ", part).strip()
        for part in parts
        if part.strip()
    ]


def extract_citations(claim: str) -> list[int]:
    """
    Supports:
      [Document 1]
      [Document 1, Document 2]
      [Document 1, Document 2, Document 3]
    """
    citations = []

    pattern = r"\[([^\]]+)\]"

    for bracket_content in re.findall(pattern, claim):
        numbers = re.findall(
            r"Document\s+(\d+)",
            bracket_content,
            flags=re.IGNORECASE,
        )
        citations.extend(int(number) for number in numbers)

    return sorted(set(citations))


def remove_citations(claim: str) -> str:
    """
    Remove citation markup before NLI verification.
    """
    cleaned = re.sub(
        r"\s*\[[^\]]*Document\s+\d+[^\]]*\]",
        "",
        claim,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", cleaned).strip()


def classify_scores(
    scores,
    labels: list[str],
) -> dict:
    """
    Classify NLI softmax scores into a verdict based on the highest-scoring label.
    This avoids overly-strict fixed thresholds that can mark reasonable entailment
    predictions as unsupported when the model's probabilities are more spread out.
    """
    label_to_index = {
        label.lower(): index
        for index, label in enumerate(labels)
    }

    # Read scores for each label (fall back to 0.0 if a label is unexpectedly missing)
    contradiction_score = float(
        scores[label_to_index.get("contradiction", 0)]
    )
    entailment_score = float(
        scores[label_to_index.get("entailment", 0)]
    )
    neutral_score = float(
        scores[label_to_index.get("neutral", 0)]
    )

    # Decide by the highest probability label. If tie or unexpected ordering
    # the fallback is UNSUPPORTED to avoid false positives.
    scores_map = {
        "contradiction": contradiction_score,
        "entailment": entailment_score,
        "neutral": neutral_score,
    }

    best_label = max(scores_map, key=scores_map.get)

    if best_label == "entailment":
        verdict = "SUPPORTED"
    elif best_label == "contradiction":
        verdict = "CONTRADICTED"
    else:
        verdict = "UNSUPPORTED"

    return {
        "verdict": verdict,
        "entailment_score": entailment_score,
        "contradiction_score": contradiction_score,
        "neutral_score": neutral_score,
    }


def verify_sentence(
    sentence: str,
    cited_documents: list[dict],
) -> dict:
    """
    Verify a single claim against cited documents by breaking each document into
    small passages (3-sentence chunks), scoring each passage, and using the
    highest-scoring passage per document to find the best supporting document.
    This improves signal when documents are long and the relevant text is a
    small section that would otherwise be truncated or diluted.
    """
    model = get_nli_model()

    if not cited_documents:
        return {
            "verdict": "MISSING_SOURCE",
            "best_document": None,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
        }

    claim_without_citation = remove_citations(sentence)

    # Build passages per document: split into sentences and group into windows
    passages = []  # list of tuples (doc_index, passage_text)

    for document in cited_documents:
        text = document["text"] or ""
        # Split into sentences (simple heuristic)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        if not sentences:
            # fallback to full text if sentence-splitting failed
            passages.append((document["index"], text))
            continue

        window_size = 3
        for i in range(0, len(sentences), window_size):
            chunk = " ".join(sentences[i : i + window_size])
            passages.append((document["index"], chunk))

    # Prepare model pairs and keep mapping to which doc/passage
    pairs = [[passage, claim_without_citation] for (_idx, passage) in passages]

    if not pairs:
        # Nothing to check; return unsupported by default
        return {
            "verdict": "UNSUPPORTED",
            "best_document": None,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
        }

    scores = model.predict(
        pairs,
        apply_softmax=True,
        batch_size=8,
    )

    if nli_labels is None:
        raise RuntimeError("NLI labels were not initialized")

    entailment_index = nli_labels.index("entailment")

    # For each passage, get its entailment score and group by document
    doc_best = {}  # doc_index -> (best_passage_idx, best_entailment_score)

    for i, (doc_index, _passage) in enumerate(passages):
        ent_score = float(scores[i][entailment_index])
        if doc_index not in doc_best or ent_score > doc_best[doc_index][1]:
            doc_best[doc_index] = (i, ent_score)

    # Choose the document whose best passage has the highest entailment score
    best_doc_index = max(doc_best.items(), key=lambda kv: kv[1][1])[0]
    best_passage_idx = doc_best[best_doc_index][0]

    classification = classify_scores(
        scores[best_passage_idx],
        nli_labels,
    )

    return {
        **classification,
        "best_document": best_doc_index,
    }


def verify_answer(
    answer: str,
    retrieved_docs: list[dict],
) -> dict:
    indexed_docs = [
        {
            "index": index,
            "id": document["id"],
            "text": document["text"],
        }
        for index, document in enumerate(
            retrieved_docs,
            start=1,
        )
    ]

    doc_lookup = {
        document["index"]: document
        for document in indexed_docs
    }

    results = []

    for claim in split_claims(answer):
        citation_numbers = extract_citations(claim)

        # Ignore headings and generic intro lines.
        if (
            not citation_numbers
            and (
                claim.startswith("Here's an answer")
                or claim.lower() in {
                    "answer:",
                    "sources:",
                }
            )
        ):
            continue

        if not citation_numbers:
            results.append({
                "sentence": claim,
                "citations": [],
                "verdict": "NO_CITATION",
            })
            continue

        invalid_citations = [
            number
            for number in citation_numbers
            if number not in doc_lookup
        ]

        if invalid_citations:
            results.append({
                "sentence": claim,
                "citations": citation_numbers,
                "invalid_citations": invalid_citations,
                "verdict": "INVALID_CITATION",
            })
            continue

        cited_documents = [
            doc_lookup[number]
            for number in citation_numbers
        ]

        verification = verify_sentence(
            sentence=claim,
            cited_documents=cited_documents,
        )

        results.append({
            "sentence": claim,
            "citations": citation_numbers,
            **verification,
        })

    checkable = [
        item
        for item in results
        if item["verdict"] in {
            "SUPPORTED",
            "UNSUPPORTED",
            "CONTRADICTED",
        }
    ]

    supported = [
        item
        for item in checkable
        if item["verdict"] == "SUPPORTED"
    ]

    return {
        "claims": results,
        "faithfulness": (
            len(supported) / len(checkable)
            if checkable
            else None
        ),
        "total_claims": len(results),
        "checkable_claims": len(checkable),
        "supported_claims": len(supported),
    }