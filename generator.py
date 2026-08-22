from typing import Any, Callable, Optional

import ollama
from citation_verifier import split_sentences_robust
from reranker import get_rerank_model

LLM_MODEL = "qwen3:4b"


def prune_document_text(query: str, doc_text: str, max_sentences: int = 4) -> str:
	"""
	Prune irrelevant sentences from a long document text while preserving
	the original chronological flow and coherent context.
	"""
	sentences = split_sentences_robust(doc_text)
	if len(sentences) <= max_sentences:
		return doc_text.strip()

	reranker = get_rerank_model()
	pairs = [(query, s) for s in sentences]
	scores = reranker.predict(pairs)

	# Select top-scoring sentence indices
	top_indices = sorted(
		sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:max_sentences]
	)

	# Rebuild document text in original sequence with omission markers
	pruned_parts = []
	prev_idx = -1
	for idx in top_indices:
		if prev_idx != -1 and idx > prev_idx + 1:
			pruned_parts.append("[...]")
		pruned_parts.append(sentences[idx])
		prev_idx = idx

	return " ".join(pruned_parts)


def build_context(
		retrieved_docs: list[dict],
		query: Optional[str] = None,
		prune: bool = False,
		max_sentences_per_doc: int = 4,
) -> str:
	"""
	Construct formatted document context blocks for LLM prompt.
	Preserves exact document indexing [Document 1]..[Document N].
	Optionally applies dynamic context pruning per document.
	"""
	blocks = []

	for index, doc in enumerate(retrieved_docs, start=1):
		doc_text = doc.get("text", "")
		if prune and query and doc_text:
			doc_text = prune_document_text(query, doc_text, max_sentences=max_sentences_per_doc)

		blocks.append(
			f"[Document {index}]\n"
			f"ID: {doc.get('id', '')}\n"
			f"Text: {doc_text}"
		)

	return "\n\n".join(blocks)


def generate_answer(
		query: str,
		retrieved_docs: list[dict],
		prune_context: bool = True,
		stream: bool = False,
		on_token: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
	"""
	Generate a grounded answer using retrieved documents.
	Enforces explicit attribution, stand-alone sentence coherence, and exact citations.
	"""
	context = build_context(
		retrieved_docs=retrieved_docs,
		query=query,
		prune=prune_context,
	)

	prompt = f"""You are a precise, grounded factual assistant.

Answer the Question using ONLY the information provided in the Documents below.

CRITICAL RULES:
1. Answer the question directly with complete, coherent, stand-alone sentences.
2. Every sentence MUST end with its specific source citation: [Document N].
3. Do NOT make normative leaps: if a document discusses possibilities, options, or examples, frame them as options (e.g. "One option mentioned is..." or "[Document 3] notes that...") rather than claiming someone "should" or "must" do it.
4. Attribute claims accurately to the specific document that contains that fact. Do not mix up documents.
5. Every sentence must have a clear subject. Never copy verbatim fragments with missing referents (e.g. avoid starting sentences with "It's just..." or "I posted...").
6. If the documents do not contain enough information to answer, state:
   "The provided documents do not contain enough information to answer this."

Question:
{query}

Documents:
{context}

Answer:
"""

	if stream:
		response_stream = ollama.generate(
			model=LLM_MODEL,
			prompt=prompt,
			stream=True,
			options={
				"temperature": 0,
			}
		)
		full_answer_parts = []
		for chunk in response_stream:
			token = chunk.get("response", "")
			full_answer_parts.append(token)
			if on_token:
				on_token(token)
		answer = "".join(full_answer_parts)
	else:
		response = ollama.generate(
			model=LLM_MODEL,
			prompt=prompt,
			options={
				"temperature": 0,
			}
		)
		answer = response["response"]

	return {
		"answer": answer,
		"documents": retrieved_docs,
	}


def self_correct_answer(
		query: str,
		initial_answer: str,
		unverified_claims: list[dict],
		retrieved_docs: list[dict],
		prune_context: bool = True,
		stream: bool = False,
		on_token: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
	"""
	Re-prompt the LLM to resolve verification issues with strict coherence
	and factual precision.
	"""
	context = build_context(
		retrieved_docs=retrieved_docs,
		query=query,
		prune=prune_context,
	)

	feedback_lines = []
	for c in unverified_claims:
		verdict = c.get("verdict", "UNSUPPORTED")
		sentence = c.get("sentence", "")
		citations = c.get("citations", [])
		citations_str = ", ".join(f"Document {n}" for n in citations) if citations else "No Citation"
		feedback_lines.append(f"- [{verdict}] \"{sentence}\" (Cited: {citations_str})")

	feedback_text = "\n".join(feedback_lines)

	prompt = f"""You are a precise, self-correcting factual assistant.

A previous answer was checked against the source documents and had the following verification issues:
{feedback_text}

Original Answer:
{initial_answer}

Original Question:
{query}

Documents:
{context}

REVISION INSTRUCTIONS:
1. Rewrite the answer so that EVERY sentence is a complete, well-formed sentence answering the question.
2. For unverified/unsupported claims: either accurately rephrase them to match what the document explicitly states (e.g. describe choices as options, not commands), or completely delete that sentence.
3. Every single sentence MUST end with its source citation in the format [Document N].
4. Every sentence MUST be self-contained with a clear noun subject (do NOT begin sentences with dangling pronouns like "It's just money..." or raw conversational fragments).
5. Only cite [Document N] if that specific document directly supports that exact statement.

Corrected Answer:
"""

	if stream:
		response_stream = ollama.generate(
			model=LLM_MODEL,
			prompt=prompt,
			stream=True,
			options={
				"temperature": 0,
			}
		)
		full_answer_parts = []
		for chunk in response_stream:
			token = chunk.get("response", "")
			full_answer_parts.append(token)
			if on_token:
				on_token(token)
		answer = "".join(full_answer_parts)
	else:
		response = ollama.generate(
			model=LLM_MODEL,
			prompt=prompt,
			options={
				"temperature": 0,
			}
		)
		answer = response["response"]

	return {
		"answer": answer,
		"documents": retrieved_docs,
	}
