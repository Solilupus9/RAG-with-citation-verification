from typing import Any

import ollama

LLM_MODEL = "gemma3:4b"


def build_context(retrieved_docs: list[dict]) -> str:
	blocks = []

	for index, doc in enumerate(retrieved_docs, start=1):
		blocks.append(
			f"[Document {index}]\n"
			f"ID: {doc['id']}\n"
			f"Text: {doc['text']}"
		)

	return "\n\n".join(blocks)


def generate_answer(
		query: str,
		retrieved_docs: list[dict]
) -> dict[str, Any]:
	context = build_context(retrieved_docs)

	prompt = f"""You are a careful question-answering assistant.

Answer the question using only the provided documents.

Rules:
- Write one factual claim per sentence.
- Put a citation after every factual claim.
- Use citations in the exact format [Document N].
- N must refer to one of the provided documents.
- Do not use outside knowledge.
- Only include claims directly supported by the cited passage.
- If uncertain whether a claim is directly supported, omit that sentence.
- If the documents do not contain enough information, say:
  "The provided documents do not contain enough information to answer this."
- Do not invent figures, dates, names, or recommendations.
- Avoid broad or absolute wording (e.g., "always", "never", "safe", "crucial")
  unless the document explicitly states it.

Question:
{query}

Documents:
{context}

Answer:
"""

	response = ollama.generate(
		model=LLM_MODEL,
		prompt=prompt,
		options={
			"temperature": 0,
		}
	)

	return {
		"answer": response["response"],
		"documents": retrieved_docs,
	}
