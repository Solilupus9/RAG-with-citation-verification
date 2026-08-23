import sys
import time
from typing import Optional

import pandas as pd
from bm25_retriever import load_bm25_index
from citation_checks_simple import validate_citation_numbers
from citation_verifier import get_nli_model, verify_answer
from dense_retriever import load_dense_index
from generator import generate_answer, self_correct_answer
from hybrid_search import hybrid_search_full
from reranker import get_rerank_model

DATA_DIR = "./data"


def print_banner():
	print("=" * 70)
	print("       RAG with Citation Verification & Self-Correction (CLI REPL)")
	print("=" * 70)
	print("Commands:")
	print("  :q / :exit       -> Exit the application")
	print("  :k <number>      -> Set final number of retrieved documents (e.g. :k 5)")
	print("  :prune on/off    -> Enable/disable dynamic context pruning")
	print("  :rerank on/off   -> Enable/disable CrossEncoder re-ranking")
	print("=" * 70)


def print_verification_report(verification: dict, citation_status: dict):
	faithfulness = verification.get("faithfulness")
	if faithfulness is None:
		print("  [!] No cited factual claims could be checked.")
	else:
		badge = "[PASS]" if faithfulness >= 0.80 else "[WARN]"
		print(
			f"  {badge} Faithfulness: {faithfulness:.1%} "
			f"({verification['supported_claims']}/"
			f"{verification['checkable_claims']} claims supported)"
		)

	if not citation_status.get("all_citations_valid", True):
		print("  [!] Invalid citation indices detected:", citation_status.get("invalid_citations"))

	print()
	for claim in verification.get("claims", []):
		verdict = claim.get("verdict", "UNKNOWN")
		sentence = claim.get("sentence", "")

		if verdict == "SUPPORTED":
			icon = "[+]"
		elif verdict == "CONTRADICTED":
			icon = "[-]"
		elif verdict == "UNSUPPORTED":
			icon = "[?]"
		else:
			icon = "[!]"

		print(f"  {icon} [{verdict}] {sentence}")

		if "entailment_score" in claim:
			print(
				f"      -> Entailment: {claim['entailment_score']:.3f} | "
				f"Neutral: {claim['neutral_score']:.3f} | "
				f"Contradiction: {claim['contradiction_score']:.3f}"
			)


def run_pipeline(
		query: str,
		bm25_retriever,
		dense_embeddings,
		dense_doc_ids,
		corpus_lookup: dict[str, str],
		final_k: int = 5,
		prune_context: bool = True,
		use_reranker: bool = True,
):
	total_start = time.perf_counter()

	# 1. Retrieval
	retrieval_start = time.perf_counter()
	retrieved_docs = hybrid_search_full(
		query=query,
		bm25_retriever=bm25_retriever,
		dense_embeddings=dense_embeddings,
		dense_doc_ids=dense_doc_ids,
		corpus_lookup=corpus_lookup,
		candidates_k=max(20, final_k * 4),
		final_k=final_k,
		use_reranker=use_reranker,
	)
	retrieval_time = time.perf_counter() - retrieval_start

	# 2. Initial Generation (Streaming)
	print("\n" + "=" * 70)
	print("INITIAL ANSWER (Streaming)")
	print("=" * 70)
	gen_start = time.perf_counter()
	result = generate_answer(
		query=query,
		retrieved_docs=retrieved_docs,
		prune_context=prune_context,
		stream=True,
		on_token=lambda tok: print(tok, end="", flush=True),
	)
	print()
	gen_time = time.perf_counter() - gen_start

	answer = result["answer"]
	documents = result["documents"]

	# 3. Citation Verification
	ver_start = time.perf_counter()
	citation_status = validate_citation_numbers(
		answer=answer,
		number_of_documents=len(documents),
	)
	verification = verify_answer(
		answer=answer,
		retrieved_docs=documents,
	)
	ver_time = time.perf_counter() - ver_start

	print("\n" + "-" * 70)
	print("INITIAL CITATION VERIFICATION")
	print("-" * 70)
	print_verification_report(verification, citation_status)

	# 4. Self-Correction Trigger
	unverified_claims = [
		claim
		for claim in verification.get("claims", [])
		if claim.get("verdict") in {"UNSUPPORTED", "CONTRADICTED", "INVALID_CITATION", "NO_CITATION"}
	]
	faithfulness = verification.get("faithfulness")

	if unverified_claims or (faithfulness is not None and faithfulness < 0.80):
		print("\n" + "=" * 70)
		print("TRIGGERING SELF-CORRECTION LOOP (Streaming Corrected Answer)...")
		print("=" * 70)

		corr_start = time.perf_counter()
		corrected_result = self_correct_answer(
			query=query,
			initial_answer=answer,
			unverified_claims=unverified_claims,
			retrieved_docs=documents,
			prune_context=prune_context,
			stream=True,
			on_token=lambda tok: print(tok, end="", flush=True),
		)
		print()
		corr_time = time.perf_counter() - corr_start

		corrected_answer = corrected_result["answer"]
		corrected_citation_status = validate_citation_numbers(
			answer=corrected_answer,
			number_of_documents=len(documents),
		)
		corrected_verification = verify_answer(
			answer=corrected_answer,
			retrieved_docs=documents,
		)

		print("\n" + "-" * 70)
		print("CORRECTED CITATION VERIFICATION")
		print("-" * 70)
		print_verification_report(corrected_verification, corrected_citation_status)

		answer = corrected_answer
		verification = corrected_verification

	# 5. Retrieved Sources Summary
	print("\n" + "=" * 70)
	print("RETRIEVED SOURCES")
	print("=" * 70)
	for index, document in enumerate(documents, start=1):
		score_str = f" | Rerank Score: {document['rerank_score']:.4f}" if "rerank_score" in document else ""
		print(f"  [Document {index}] ID: {document['id']}{score_str}")
		preview = document.get("text", "").replace("\n", " ").strip()
		if len(preview) > 140:
			preview = preview[:140] + "..."
		print(f"    {preview}\n")

	total_time = time.perf_counter() - total_start
	print("-" * 70)
	print(
		f"Latency Breakdown: Retrieval: {retrieval_time:.2f}s | "
		f"Generation: {gen_time:.2f}s | Verification: {ver_time:.2f}s | Total: {total_time:.2f}s"
	)
	print("-" * 70)


def main():
	print_banner()

	print("\nPreloading models and search indexes (one-time setup)...")
	load_start = time.perf_counter()
	bm25_retriever = load_bm25_index()
	dense_embeddings, dense_doc_ids = load_dense_index()

	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	corpus["_id"] = corpus["_id"].astype(str)
	corpus_lookup = dict(zip(corpus["_id"], corpus["text"].astype(str)))

	# Warm up models
	get_rerank_model()
	get_nli_model()
	print(f"Setup complete in {time.perf_counter() - load_start:.2f}s! Ready for questions.\n")

	# Configuration state
	final_k = 5
	prune_context = True
	use_reranker = True

	while True:
		try:
			query = input("\nAsk a question (:q to exit) > ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nExiting. Goodbye!")
			break

		if not query:
			continue

		# Handle CLI commands
		lower = query.lower()
		if lower in {":q", ":exit", "exit", "quit"}:
			print("Exiting. Goodbye!")
			break
		elif lower.startswith(":k "):
			try:
				val = int(query.split()[1])
				if 1 <= val <= 20:
					final_k = val
					print(f"[*] Retrieved documents count (final_k) set to {final_k}")
				else:
					print("[!] Please provide a number between 1 and 20.")
			except ValueError:
				print("[!] Usage: :k <number>")
			continue
		elif lower == ":prune on":
			prune_context = True
			print("[*] Context pruning ENABLED")
			continue
		elif lower == ":prune off":
			prune_context = False
			print("[*] Context pruning DISABLED")
			continue
		elif lower == ":rerank on":
			use_reranker = True
			print("[*] Re-ranking ENABLED")
			continue
		elif lower == ":rerank off":
			use_reranker = False
			print("[*] Re-ranking DISABLED")
			continue

		# Run pipeline
		run_pipeline(
			query=query,
			bm25_retriever=bm25_retriever,
			dense_embeddings=dense_embeddings,
			dense_doc_ids=dense_doc_ids,
			corpus_lookup=corpus_lookup,
			final_k=final_k,
			prune_context=prune_context,
			use_reranker=use_reranker,
		)


if __name__ == "__main__":
	main()
