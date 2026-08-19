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
- Use only the provided documents.
- Write short, atomic factual claims.
- Each sentence should contain only one main factual claim.
- Put citations at the end of every factual sentence.
- Use citations in the exact format [Document N].
- For multiple sources, use [Document 1, Document 2].
- N must refer to a provided document.
- Do not combine unrelated claims into one sentence.
- Do not invent figures, dates, names, or recommendations.
- If the documents do not support an answer, say:
  "The provided documents do not contain enough information to answer this."

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
