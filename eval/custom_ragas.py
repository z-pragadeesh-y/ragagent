"""
eval/custom_ragas.py

Custom implementation of 4 core RAGAS-style metrics, scored via LLM-judge
prompts through this project's own llm_manager (task="generate" lane).
Built after the official `ragas` PyPI package proved incompatible with this
project's LangChain 1.x stack across 3 attempted versions (0.4.3, 0.2.15,
0.1.21) - see git log for that investigation. Same evaluation intent as
RAGAS, zero new dependencies, fully transparent scoring logic.

Metrics (each scored 0.0-1.0 by an LLM judge):
- faithfulness: does the answer only contain claims supported by context?
- answer_relevancy: does the answer actually address the question?
- context_precision: of the retrieved chunks, how many were relevant?
- context_recall: did retrieved context contain enough to support ground_truth?

Usage:
    python -m eval.custom_ragas
"""
import json
import re
import uuid
import logging

from langchain_core.prompts import ChatPromptTemplate
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError
from graph.build_graph import build_graph
from eval.golden_set import GOLDEN_SET

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("custom_ragas")

RESULTS_CSV = "eval/custom_ragas_results.csv"

FAITHFULNESS_PROMPT = """Score how well the ANSWER is supported by the CONTEXT, from 0.0 to 1.0.
1.0 = every claim in the answer is directly supported by the context.
0.0 = the answer contains claims not found in the context at all (hallucination).
If the answer correctly says "I don't have enough information," score 1.0 (that is faithful, not a failure).
Judge ONLY based on the provided context below - do not use your own outside knowledge to second-guess it.

Context:
{context}

Answer:
{answer}

Respond with ONLY a JSON object: {{"score": 0.0, "reason": "brief reason"}}"""

RELEVANCY_PROMPT = """Score how well the ANSWER actually addresses the QUESTION, from 0.0 to 1.0.
A REFERENCE ANSWER is provided so you can check factual alignment - use it as the source of truth,
NOT your own background knowledge, which may be outdated or wrong.
1.0 = directly and substantively addresses the question, consistent with the reference answer's facts.
0.0 = off-topic, evasive, or contradicts the reference answer.
An honest "I don't have enough information" IS relevant if the context genuinely lacked the answer.

Question:
{question}

Reference answer (ground truth):
{ground_truth}

Answer to evaluate:
{answer}

Respond with ONLY a JSON object: {{"score": 0.0, "reason": "brief reason"}}"""

CONTEXT_PRECISION_PROMPT = """You are given a QUESTION and a list of retrieved PASSAGES.
Score what fraction of the passages are actually relevant/useful for answering the question, from 0.0 to 1.0.
1.0 = all passages are relevant. 0.0 = none are relevant.

Question:
{question}

Passages:
{context}

Respond with ONLY a JSON object: {{"score": 0.0, "reason": "brief reason"}}"""

CONTEXT_RECALL_PROMPT = """You are given a GROUND_TRUTH answer and the retrieved CONTEXT that was actually
provided to the system. Score whether the context contains enough information to fully support the
ground truth, from 0.0 to 1.0. 1.0 = context fully supports it. 0.0 = context is missing the key facts entirely.

Ground truth:
{ground_truth}

Context:
{context}

Respond with ONLY a JSON object: {{"score": 0.0, "reason": "brief reason"}}"""


def _judge(prompt_template: str, **kwargs) -> dict:
    """Runs one judge prompt, returns {'score': float, 'reason': str}. Fails safe to 0.0 on any error."""
    llm = get_llm(task="grade", temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    try:
        response = chain.invoke(kwargs)
        raw = response.content.strip()
    except AllProvidersFailedError:
        return {"score": 0.0, "reason": "judge LLM unavailable (all providers failed)"}

    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"score": 0.0, "reason": f"could not parse judge output: {raw[:100]}"}

    try:
        parsed = json.loads(match.group(0))
        return {"score": float(parsed.get("score", 0.0)), "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": 0.0, "reason": f"could not parse judge output: {raw[:100]}"}


def score_question(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    context_text = "\n\n---\n\n".join(contexts)

    faithfulness = _judge(FAITHFULNESS_PROMPT, context=context_text, answer=answer)
    relevancy = _judge(RELEVANCY_PROMPT, question=question, ground_truth=ground_truth, answer=answer)
    precision = _judge(CONTEXT_PRECISION_PROMPT, question=question, context=context_text)
    recall = _judge(CONTEXT_RECALL_PROMPT, ground_truth=ground_truth, context=context_text)

    return {
        "faithfulness": faithfulness["score"],
        "answer_relevancy": relevancy["score"],
        "context_precision": precision["score"],
        "context_recall": recall["score"],
    }


def run_single_question(app, question: str):
    """Invokes the full graph for one question. Same pattern as graph/test_cases.py."""
    config = {"configurable": {"thread_id": f"custom-ragas-{uuid.uuid4()}"}}

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
    if not contexts:
        contexts = ["(no context retrieved - question was routed out_of_scope)"]
    return answer, contexts


def run_evaluation():
    app = build_graph()
    rows = []

    for i, item in enumerate(GOLDEN_SET, start=1):
        logger.info(f"[{i}/{len(GOLDEN_SET)}] ({item['domain_tag']}) {item['question']}")
        answer, contexts = run_single_question(app, item["question"])
        scores = score_question(item["question"], answer, contexts, item["ground_truth"])

        rows.append({
            "question": item["question"],
            "domain_tag": item["domain_tag"],
            "answer": answer,
            **scores,
        })

    import pandas as pd
    import os

    df = pd.DataFrame(rows)
    os.makedirs("eval", exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    logger.info(f"\nFull per-question results saved to {RESULTS_CSV}")

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n=== Overall Averages (all questions) ===")
    print(df[metric_cols].mean())

    print("\n=== Per-Domain Averages ===")
    print(df.groupby("domain_tag")[metric_cols].mean())

    return df


if __name__ == "__main__":
    run_evaluation()
