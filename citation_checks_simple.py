import re


def extract_citations(text: str) -> list[int]:
    citations = []

    for bracket_content in re.findall(
        r"\[([^\]]+)\]",
        text,
    ):
        numbers = re.findall(
            r"Document\s+(\d+)",
            bracket_content,
            flags=re.IGNORECASE,
        )
        citations.extend(int(number) for number in numbers)

    return sorted(set(citations))


def validate_citation_numbers(
    answer: str,
    number_of_documents: int,
) -> dict:
    citations = extract_citations(answer)

    invalid = [
        number
        for number in citations
        if number < 1
        or number > number_of_documents
    ]

    return {
        "citations": citations,
        "invalid_citations": invalid,
        "all_citations_valid": not invalid,
    }