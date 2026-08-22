import re
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAILMENT_THRESHOLD = 0.70
STRICT_ENTAILMENT_THRESHOLD = 0.78
CONTRADICTION_THRESHOLD = 0.60

nli_model: Optional[CrossEncoder] = None
nli_labels: Optional[list[str]] = None

ABBR_PATTERN = (
    r"\b(?:"
    r"e\.g\.|i\.e\.|u\.s\.|u\.k\.|dr\.|mr\.|mrs\.|ms\.|prof\.|"
    r"inc\.|corp\.|ltd\.|co\.|vs\.|etc\.|fig\.|no\.|vol\.|approx\.|"
    r"jan\.|feb\.|mar\.|apr\.|jun\.|jul\.|aug\.|sep\.|sept\.|oct\.|nov\.|dec\."
    r")"
)


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


def split_sentences_robust(text: str) -> list[str]:
    """
    Split text into sentences while avoiding false splits on abbreviations
    (e.g., U.S., e.g., i.e., vs., Dec.), decimal numbers (e.g., 4.5%, $10.50),
    and bracketed citations.
    """
    if not text or not text.strip():
        return []

    temp_text = text.strip()
    placeholder = "___DOT___"

    # Protect decimal numbers (digits.digits)
    temp_text = re.sub(r"(\d+)\.(\d+)", rf"\1{placeholder}\2", temp_text)

    # Protect known abbreviations and months case-insensitively
    def _replace_abbr(match: re.Match) -> str:
        return match.group(0).replace(".", placeholder)

    temp_text = re.sub(ABBR_PATTERN, _replace_abbr, temp_text, flags=re.IGNORECASE)

    # Split on sentence terminals followed by whitespace and beginning of sentence
    splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])", temp_text)

    results = []
    for s in splits:
        restored = s.replace(placeholder, ".")
        cleaned = re.sub(r"\s+", " ", restored).strip()
        if cleaned:
            results.append(cleaned)

    return results if results else [text.strip()]


def split_claims(text: str) -> list[str]:
    """
    Split answer text into granular factual claims.
    Goal: one sentence-level claim per item, not mixed lists.
    """
    normalized = text.replace("\r\n", "\n").strip()

    # Treat Markdown bullets as separate boundaries
    normalized = re.sub(
        r"(?m)^\s*[*-]\s+",
        "\n",
        normalized,
    )

    # Remove emphasis markup before splitting
    normalized = re.sub(r"[*_`]", "", normalized)

    parts: list[str] = []
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]

    for line in lines:
        sentence_parts = split_sentences_robust(line)
        for sentence in sentence_parts:
            # Also split on semicolon-separated independent statements
            semicolon_parts = re.split(r"\s*;\s+", sentence)
            for part in semicolon_parts:
                cleaned = re.sub(r"\s+", " ", part).strip()
                if cleaned:
                    parts.append(cleaned)

    return parts


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
    claim_text: str,
) -> dict:
    """
    Classify NLI softmax scores into a verdict based on the highest-scoring label
    and calibrated probability thresholds.
    """
    label_to_index = {
        label.lower(): index
        for index, label in enumerate(labels)
    }

    contradiction_score = float(
        scores[label_to_index.get("contradiction", 0)]
    )
    entailment_score = float(
        scores[label_to_index.get("entailment", 0)]
    )
    neutral_score = float(
        scores[label_to_index.get("neutral", 0)]
    )

    lower_claim = claim_text.lower()
    has_broad_or_absolute_language = bool(
        re.search(
            r"\b("
            r"always|never|must|guaranteed|risk[- ]?free|100%|only|impossible"
            r")\b",
            lower_claim,
        )
    )
    support_threshold = (
        STRICT_ENTAILMENT_THRESHOLD
        if has_broad_or_absolute_language
        else ENTAILMENT_THRESHOLD
    )

    scores_map = {
        "contradiction": contradiction_score,
        "entailment": entailment_score,
        "neutral": neutral_score,
    }

    best_label = max(scores_map, key=scores_map.get)

    if (
        best_label == "entailment"
        and entailment_score >= support_threshold
    ):
        verdict = "SUPPORTED"
    elif (
        best_label == "contradiction"
        and contradiction_score >= CONTRADICTION_THRESHOLD
    ):
        verdict = "CONTRADICTED"
    else:
        verdict = "UNSUPPORTED"

    return {
        "verdict": verdict,
        "entailment_score": entailment_score,
        "contradiction_score": contradiction_score,
        "neutral_score": neutral_score,
        "support_threshold": support_threshold,
        "strict_claim": has_broad_or_absolute_language,
    }


def verify_sentence(
    sentence: str,
    cited_documents: list[dict],
) -> dict:
    """
    Verify a single claim against cited documents.
    Generates granular individual sentence passages and 2-sentence adjacent windows
    to avoid premise dilution while capturing multi-sentence claims.
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

    # Build granular passages per document:
    # 1) Single sentences (prevents premise dilution on atomic claims)
    # 2) 2-sentence contiguous pairs (captures premises spanning two sentences)
    passages = []  # list of tuples (doc_index, passage_text)

    for document in cited_documents:
        text = document.get("text", "") or ""
        sentences = split_sentences_robust(text)

        if not sentences:
            passages.append((document["index"], text))
            continue

        for s in sentences:
            if s.strip():
                passages.append((document["index"], s.strip()))

        for i in range(len(sentences) - 1):
            pair_chunk = f"{sentences[i]} {sentences[i + 1]}".strip()
            passages.append((document["index"], pair_chunk))

    pairs = [[passage, claim_without_citation] for (_idx, passage) in passages]

    if not pairs:
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
        batch_size=16,
    )

    if nli_labels is None:
        raise RuntimeError("NLI labels were not initialized")

    entailment_index = nli_labels.index("entailment")

    # Group best passage per document by highest entailment score
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
        claim_without_citation,
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

        # Ignore headings and generic intro lines
        if (
            not citation_numbers
            and (
                claim.startswith("Here's an answer")
                or claim.lower() in {
                    "answer:",
                    "sources:",
                }
                or claim.endswith(":")
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
