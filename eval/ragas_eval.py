"""
eval/ragas_eval.py

RAGAS evaluation harness for Plan 1 Step 4.

Runs the golden set (eval/golden_set.py) through the current graph pipeline
(Phases 1-4 + Plan 1 Steps 1-3), collects each generated answer and its
retrieved contexts, then scores the whole run with RAGAS:
    faithfulness, answer_relevancy, context_precision, context_recall

NOTE ON BASELINE: this project does not capture a pre-Steps-1-3 baseline
run (Steps 1-3 were already built and verified before this harness was
written). This script therefore produces a single post-Steps-1-3
measurement, not a before/after comparison. Documented as a deliberate,
accepted gap rather than an oversight.

Usage:
    python -m eval.ragas_eval
"""
import uuid
import os
import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# --- Embeddings import ---------------------------------------------------
# This project's embeddings policy is "free and local" (sentence-transformers
# all-MiniLM-L6-v2), same model ingestion/embedder.py already loads.
try:
    from ingestion.embedder import get_embedding_model
    _embedding_model = get_embedding_model()
except ImportError:
    from langchain_huggingface import HuggingFaceEmbeddings
    _embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from graph.build_graph import build_graph
from eval.golden_set import GOLDEN_SET
from llm.task_router import get_llm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ragas_eval")

RESULTS_CSV = "eval/ragas_results.csv"


def run_single_question(app, question: str):
    """
    Invokes the full graph for one question, using RAGState's real field
    names (graph/state.py) and the same full-initial-state pattern used in
    graph/test_cases.py. Each question gets its own unique thread_id, so
    the SqliteSaver checkpointer never leaks chat_history between unrelated
    golden-set questions.

    Returns (answer: str, contexts: list[str]).
    """
    config = {"configurable": {"thread_id": f"ragas-eval-{uuid.uuid4()}"}}

    result = app.invoke({
        "question": question,
        "rewritten_question": "",
        "retrieved_docs": [],
        "answer": "",
        "is_relevant": False,
        "chat_history": [],
        "route_category": "",
        "sub_questions": [],
        "retry_count": 0,
        "grading_passed": False,
    }, config=config)

    answer = result.get("answer", "")
    contexts = [doc.page_content for doc in result.get("retrieved_docs", [])]
    return answer, contexts


def build_ragas_dataset(app):
    """
    Runs every question in GOLDEN_SET through the graph and assembles the
    RAGAS-format dataset (question, answer, contexts, ground_truth).
    Also returns the raw per-question rows (carrying domain_tag), since
    RAGAS's own output doesn't preserve custom columns like domain_tag
    through evaluate() - we re-attach it after scoring for the per-domain
    breakdown.
    """
    rows = []
    for i, item in enumerate(GOLDEN_SET, start=1):
        logger.info(f"[{i}/{len(GOLDEN_SET)}] ({item['domain_tag']}) {item['question']}")
        answer, contexts = run_single_question(app, item["question"])

        # context_precision / context_recall need at least one non-empty
        # context string. Out-of-scope questions correctly retrieve nothing
        # (they route straight to out_of_scope_node) - give them an explicit
        # placeholder rather than an empty list, which would crash scoring.
        if not contexts:
            contexts = ["(no context retrieved - question was routed out_of_scope)"]

        rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "domain_tag": item["domain_tag"],
        })

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])
    return dataset, rows


def get_ragas_judge_models():
    """
    RAGAS's metrics default to OpenAI. This project uses its own multi-provider
    LLMManager (Groq -> NVIDIA -> local LM Studio, via the 'complex' lane) for
    judging instead of a raw ChatGroq call, so a 30-question x 4-metric
    evaluation run - a lot of LLM calls - benefits from the same provider
    failover protection the rest of the project relies on, rather than
    risking a bare Groq quota hit partway through scoring.
    """
    judge_llm = get_llm(task="generate", temperature=0)
    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(_embedding_model)


def run_evaluation():
    app = build_graph()
    dataset, rows = build_ragas_dataset(app)

    ragas_llm, ragas_embeddings = get_ragas_judge_models()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores_df = result.to_pandas()
    scores_df["domain_tag"] = [r["domain_tag"] for r in rows]

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    scores_df.to_csv(RESULTS_CSV, index=False)
    logger.info(f"\nFull per-question results saved to {RESULTS_CSV}")

    print("\n=== Overall RAGAS Scores (all 30 questions) ===")
    print(result)

    print("\n=== Per-Domain Averages ===")
    domain_avg = scores_df.groupby("domain_tag")[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean()
    print(domain_avg)

    return result, scores_df


if __name__ == "__main__":
    run_evaluation()
