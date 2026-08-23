import time
from typing import Optional

import pandas as pd
import streamlit as st
from bm25_retriever import load_bm25_index
from citation_checks_simple import validate_citation_numbers
from citation_verifier import get_nli_model, verify_answer
from dense_retriever import load_dense_index
from generator import generate_answer, self_correct_answer
from hybrid_search import hybrid_search_full
from reranker import get_rerank_model

DATA_DIR = "./data"

st.set_page_config(
	page_title="RAG with Citation Verification",
	page_icon="🛡️",
	layout="wide",
	initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Pre-loading indexes and AI models into memory...")
def load_all_resources():
	bm25_retriever = load_bm25_index()
	dense_embeddings, dense_doc_ids = load_dense_index()
	corpus = pd.read_parquet(f"{DATA_DIR}/corpus.parquet")
	corpus["_id"] = corpus["_id"].astype(str)
	corpus_lookup = dict(zip(corpus["_id"], corpus["text"].astype(str)))

	# Warmup models
	get_rerank_model()
	get_nli_model()

	return bm25_retriever, dense_embeddings, dense_doc_ids, corpus_lookup


# Load resources
bm25_retriever, dense_embeddings, dense_doc_ids, corpus_lookup = load_all_resources()

# Sidebar Configuration
st.sidebar.title("⚙️ RAG Configuration")

sample_prompts = [
	"-- Select a Sample Prompt --",
	"Where should a young student put their money?",
	"What is the lender's motivation in short selling?",
	"Accidentally opened a year term CD account, then realized I need the money sooner. What to do?",
	"If I deposit money as cash does it count as direct deposit?",
	"Where should I park my rainy day emergency funds?",
	"Does the bid/ask concept exist in dealer markets?",
	"What are the IRS tax guidelines for staking Ethereum in 2026?",
]

selected_sample = st.sidebar.selectbox("💡 Sample Queries", sample_prompts)

final_k = st.sidebar.slider("Top Documents (final_k)", min_value=1, max_value=10, value=5)
candidates_k = st.sidebar.slider("Search Candidates (candidates_k)", min_value=10, max_value=50, value=20)
faithfulness_threshold = st.sidebar.slider("Faithfulness Threshold (trigger self-correction)", min_value=0.5,
										   max_value=1.0, value=0.80, step=0.05)

prune_context = st.sidebar.toggle("Dynamic Context Pruning", value=True,
								  help="Extract top relevant sentences per document to fit prompt context.")
use_reranker = st.sidebar.toggle("Cross-Encoder Re-ranking", value=True,
								 help="Use ms-marco-MiniLM-L12-v2 to rerank hybrid search results.")

st.sidebar.markdown("---")
st.sidebar.caption("⚡ **Local Models Used:**")
st.sidebar.caption("• Embeddings: `bge-small-en-v1.5`")
st.sidebar.caption("• Reranker: `ms-marco-MiniLM-L12-v2`")
st.sidebar.caption("• NLI Verifier: `nli-deberta-v3-base`")
st.sidebar.caption("• Generator: `Ollama / qwen3:4b`")

# Main Page Header
st.title("🛡️ Grounded RAG with Citation Verification")
st.markdown(
	"A local, hallucination-resistant RAG pipeline featuring **Hybrid Search (BM25 + Dense + RRF)**, "
	"**Cross-Encoder Re-ranking**, **Token Streaming**, **DeBERTa NLI Claim Verification**, and **Automated Self-Correction**."
)

default_query = "" if selected_sample == sample_prompts[0] else selected_sample
query = st.text_input("💬 Ask a Financial Question:", value=default_query,
					  placeholder="e.g., Where should a young student put their money?")

if st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True) and query.strip():
	st.markdown("---")

	# Timings
	t_start = time.perf_counter()

	with st.spinner("🔍 Performing Hybrid Search (BM25 + Dense) & Cross-Encoder Re-ranking..."):
		t_ret_start = time.perf_counter()
		retrieved_docs = hybrid_search_full(
			query=query,
			bm25_retriever=bm25_retriever,
			dense_embeddings=dense_embeddings,
			dense_doc_ids=dense_doc_ids,
			corpus_lookup=corpus_lookup,
			candidates_k=candidates_k,
			final_k=final_k,
			use_reranker=use_reranker,
		)
		retrieval_time = time.perf_counter() - t_ret_start

	col_answer, col_verif = st.columns([1.1, 0.9])

	with col_answer:
		st.subheader("📝 Answer")
		answer_box = st.empty()

		# Stream generation
		t_gen_start = time.perf_counter()
		streamed_tokens = []


		def token_handler(tok: str):
			streamed_tokens.append(tok)
			answer_box.markdown("".join(streamed_tokens) + "▌")


		result = generate_answer(
			query=query,
			retrieved_docs=retrieved_docs,
			prune_context=prune_context,
			stream=True,
			on_token=token_handler,
		)
		initial_answer = result["answer"]
		answer_box.markdown(initial_answer)
		gen_time = time.perf_counter() - t_gen_start

	with col_verif:
		st.subheader("🛡️ Citation Verification Report")
		with st.spinner("Checking factual claims with DeBERTa NLI cross-encoder..."):
			t_ver_start = time.perf_counter()
			citation_status = validate_citation_numbers(initial_answer, len(retrieved_docs))
			verification = verify_answer(initial_answer, retrieved_docs)
			ver_time = time.perf_counter() - t_ver_start

			faithfulness = verification.get("faithfulness")
			if faithfulness is not None:
				st.metric(
					label="Faithfulness Score",
					value=f"{faithfulness:.1%}",
					delta=f"{verification['supported_claims']}/{verification['checkable_claims']} claims supported",
				)
				st.progress(min(1.0, max(0.0, faithfulness)))
			else:
				st.warning("No factual claims with citations detected.")

			for claim in verification.get("claims", []):
				verdict = claim.get("verdict", "UNKNOWN")
				sentence = claim.get("sentence", "")

				if verdict == "SUPPORTED":
					badge_color = "green"
					icon = "✅"
				elif verdict == "CONTRADICTED":
					badge_color = "red"
					icon = "❌"
				elif verdict == "UNSUPPORTED":
					badge_color = "orange"
					icon = "⚠️"
				else:
					badge_color = "gray"
					icon = "ℹ️"

				with st.container(border=True):
					st.markdown(f"**:{badge_color}[{icon} {verdict}]** {sentence}")
					if "entailment_score" in claim:
						c1, c2, c3 = st.columns(3)
						c1.caption(f"Entailment: `{claim['entailment_score']:.3f}`")
						c2.caption(f"Neutral: `{claim['neutral_score']:.3f}`")
						c3.caption(f"Contradiction: `{claim['contradiction_score']:.3f}`")

	# Self-Correction Step
	unverified = [
		c for c in verification.get("claims", [])
		if c.get("verdict") in {"UNSUPPORTED", "CONTRADICTED", "INVALID_CITATION", "NO_CITATION"}
	]

	if unverified or (faithfulness is not None and faithfulness < faithfulness_threshold):
		st.markdown("---")
		st.warning(f"⚠️ Initial Faithfulness ({faithfulness:.1%}) was below threshold ({faithfulness_threshold:.0%}) "
				   f"or contained unverified claims. Triggering **Self-Correction Loop**...")

		corr_col_answer, corr_col_verif = st.columns([1.1, 0.9])

		with corr_col_answer:
			st.subheader("✨ Corrected Answer")
			corr_box = st.empty()
			corr_streamed = []


			def corr_token_handler(tok: str):
				corr_streamed.append(tok)
				corr_box.markdown("".join(corr_streamed) + "▌")


			corrected_result = self_correct_answer(
				query=query,
				initial_answer=initial_answer,
				unverified_claims=unverified,
				retrieved_docs=retrieved_docs,
				prune_context=prune_context,
				stream=True,
				on_token=corr_token_handler,
			)
			corrected_answer = corrected_result["answer"]
			corr_box.markdown(corrected_answer)

		with corr_col_verif:
			st.subheader("🛡️ Corrected Verification")
			corr_verification = verify_answer(corrected_answer, retrieved_docs)
			corr_faith = corr_verification.get("faithfulness")

			if corr_faith is not None:
				st.metric(
					label="Corrected Faithfulness",
					value=f"{corr_faith:.1%}",
					delta=f"{corr_verification['supported_claims']}/{corr_verification['checkable_claims']} claims supported",
				)
				st.progress(min(1.0, max(0.0, corr_faith)))

			for claim in corr_verification.get("claims", []):
				verdict = claim.get("verdict", "UNKNOWN")
				sentence = claim.get("sentence", "")

				badge_color = "green" if verdict == "SUPPORTED" else ("red" if verdict == "CONTRADICTED" else "orange")
				icon = "✅" if verdict == "SUPPORTED" else ("❌" if verdict == "CONTRADICTED" else "⚠️")

				with st.container(border=True):
					st.markdown(f"**:{badge_color}[{icon} {verdict}]** {sentence}")
					if "entailment_score" in claim:
						c1, c2, c3 = st.columns(3)
						c1.caption(f"Entailment: `{claim['entailment_score']:.3f}`")
						c2.caption(f"Neutral: `{claim['neutral_score']:.3f}`")
						c3.caption(f"Contradiction: `{claim['contradiction_score']:.3f}`")

	# Sources Section
	st.markdown("---")
	st.subheader(f"📚 Retrieved Context ({len(retrieved_docs)} Documents)")
	for i, doc in enumerate(retrieved_docs, start=1):
		score_info = f" — Rerank Score: `{doc['rerank_score']:.4f}`" if "rerank_score" in doc else ""
		with st.expander(f"**[Document {i}]** ID: `{doc['id']}`{score_info}"):
			st.write(doc.get("text", ""))

	# Latency Bar
	total_time = time.perf_counter() - t_start
	st.caption(
		f"⏱️ **Latency:** Retrieval: `{retrieval_time:.2f}s` | Generation: `{gen_time:.2f}s` | "
		f"Verification: `{ver_time:.2f}s` | Total: `{total_time:.2f}s`"
	)
