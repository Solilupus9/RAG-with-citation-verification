import re
from typing import Optional

import torch
from sentence_transformers import CrossEncoder


NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAILMENT_THRESHOLD = 0.60
CONTRADICTION_THRESHOLD = 0.60

nli_model: Optional[CrossEncoder] = None
nli_labels: Optional[list[str]] = None
SOURCE_CHUNK_WORDS = 120
SOURCE_CHUNK_OVERLAP = 30


def chunk_source(text: str) -> list[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(
            start + SOURCE_CHUNK_WORDS,
            len(words)
        )

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - SOURCE_CHUNK_OVERLAP

    return chunks

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
    cleaned = re.sub(
        r"\[[^\]]*Document\s+\d+[^\]]*\]",
        "",
        claim,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned.strip()

def classify_scores(
    scores,
    labels: list[str],
) -> dict:
    label_to_index = {
        label.lower(): index
        for index, label in enumerate(labels)
    }

    contradiction_score = float(
        scores[label_to_index["contradiction"]]
    )
    entailment_score = float(
        scores[label_to_index["entailment"]]
    )
    neutral_score = float(
        scores[label_to_index["neutral"]]
    )

    if (
        entailment_score >= ENTAILMENT_THRESHOLD
        and entailment_score > contradiction_score
        and entailment_score > neutral_score
    ):
        verdict = "SUPPORTED"
    elif (
        contradiction_score >= CONTRADICTION_THRESHOLD
        and contradiction_score > entailment_score
        and contradiction_score > neutral_score
    ):
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

    evidence = []

    for document in cited_documents:
        chunks = chunk_source(document["text"])

        for chunk_index, chunk in enumerate(chunks):
            evidence.append({
                "document_index": document["index"],
                "chunk_index": chunk_index,
                "text": chunk,
            })

    if not evidence:
        return {
            "verdict": "MISSING_SOURCE",
            "best_document": None,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
        }

    pairs = [
        [item["text"], claim_without_citation]
        for item in evidence
    ]

    scores = model.predict(
        pairs,
        apply_softmax=True,
        batch_size=8,
    )

    if nli_labels is None:
        raise RuntimeError(
            "NLI labels were not initialized"
        )

    entailment_index = nli_labels.index("entailment")
    contradiction_index = nli_labels.index("contradiction")
    neutral_index = nli_labels.index("neutral")

    best_index = max(
        range(len(scores)),
        key=lambda index: float(
            scores[index][entailment_index]
        ),
    )

    best_scores = scores[best_index]

    entailment_score = float(
        best_scores[entailment_index]
    )
    contradiction_score = float(
        best_scores[contradiction_index]
    )
    neutral_score = float(
        best_scores[neutral_index]
    )

    if (
        entailment_score >= ENTAILMENT_THRESHOLD
        and entailment_score > contradiction_score
        and entailment_score > neutral_score
    ):
        verdict = "SUPPORTED"
    elif (
        contradiction_score >= CONTRADICTION_THRESHOLD
        and contradiction_score > entailment_score
        and contradiction_score > neutral_score
    ):
        verdict = "CONTRADICTED"
    else:
        verdict = "UNSUPPORTED"

    best_evidence = evidence[best_index]

    return {
        "verdict": verdict,
        "best_document": best_evidence["document_index"],
        "best_chunk": best_evidence["chunk_index"],
        "entailment_score": entailment_score,
        "contradiction_score": contradiction_score,
        "neutral_score": neutral_score,
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