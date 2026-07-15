"""
eval/golden_set.py

Golden Q&A set for Plan 1 Step 4 (RAGAS Evaluation Harness).

Design notes (per RAG_Project_Roadmap_v2.docx, Plan 1 Step 4):
- Pulled 5 questions per domain (equal weight), NOT proportional to document
  length -- the corpus is domain-imbalanced (Health ~226k chars vs AI Research
  ~114k chars), so a length-proportional golden set would just measure corpus
  size rather than retrieval/generation quality.
- Every ground_truth answer below is a fact directly verified against the
  actual uploaded source documents (document_1..5_*_optimized.md), not
  generated from general knowledge -- this matters for context_recall scoring,
  since RAGAS checks retrieved context against this reference answer.
- 5 deliberately out-of-scope questions are included to exercise Step 3's
  (CRAG) fallback / Step 2's out_of_scope routing during evaluation.
- Total: 30 questions (25 in-scope, 5 out-of-scope).

Each entry:
    question      -- str, the query to feed into the RAG pipeline
    ground_truth  -- str, reference answer (used by RAGAS context_recall /
                     as the "expected" answer signal)
    domain_tag    -- one of: ai_policy, climate, economics, health, ai_research, out_of_scope
    source_file   -- which data/raw/*.md file the fact was pulled from (None for OOS)
    notes         -- optional context for why this question was chosen
"""

GOLDEN_SET = [
    # ------------------------------------------------------------------
    # AI POLICY -- NIST AI RMF 1.0 (document_1_AI_optimized.md)
    # ------------------------------------------------------------------
    {
        "question": "What are the four functions of the NIST AI RMF Core?",
        "ground_truth": "GOVERN, MAP, MEASURE, and MANAGE.",
        "domain_tag": "ai_policy",
        "source_file": "document_1_AI_optimized.md",
        "notes": "Direct factual recall, single-chunk answerable.",
    },
    {
        "question": "What are the seven characteristics of trustworthy AI systems described in the AI RMF?",
        "ground_truth": (
            "Valid and reliable; safe; secure and resilient; accountable and "
            "transparent; explainable and interpretable; privacy-enhanced; "
            "and fair with harmful bias managed."
        ),
        "domain_tag": "ai_policy",
        "source_file": "document_1_AI_optimized.md",
        "notes": "Tests whether retrieval surfaces the full enumerated list, not just a subset.",
    },
    {
        "question": "What law directed NIST to develop the AI Risk Management Framework?",
        "ground_truth": "The National Artificial Intelligence Initiative Act of 2020 (P.L. 116-283).",
        "domain_tag": "ai_policy",
        "source_file": "document_1_AI_optimized.md",
    },
    {
        "question": "By when does NIST expect to conduct a formal review of the AI RMF with input from the AI community?",
        "ground_truth": "No later than 2028.",
        "domain_tag": "ai_policy",
        "source_file": "document_1_AI_optimized.md",
    },
    {
        "question": "How does the AI RMF define reliability?",
        "ground_truth": (
            "Reliability is defined (per ISO/IEC TS 5723:2022) as the ability of an "
            "item to perform as required, without failure, for a given time "
            "interval, under given conditions."
        ),
        "domain_tag": "ai_policy",
        "source_file": "document_1_AI_optimized.md",
    },

    # ------------------------------------------------------------------
    # CLIMATE -- IPCC AR6 Synthesis Report SPM (document_2_Climate_optimized.md)
    # ------------------------------------------------------------------
    {
        "question": "How much higher was global surface temperature in 2011-2020 compared to 1850-1900?",
        "ground_truth": "1.09°C higher (with a likely range of 0.95 to 1.20°C).",
        "domain_tag": "climate",
        "source_file": "document_2_Climate_optimized.md",
    },
    {
        "question": "What were the total historical cumulative net CO2 emissions from 1850 to 2019?",
        "ground_truth": "2400 ± 240 GtCO2, of which more than half (58%) occurred between 1850 and 1989.",
        "domain_tag": "climate",
        "source_file": "document_2_Climate_optimized.md",
    },
    {
        "question": "Approximately how many people live in contexts highly vulnerable to climate change, according to IPCC AR6?",
        "ground_truth": "Approximately 3.3 to 3.6 billion people.",
        "domain_tag": "climate",
        "source_file": "document_2_Climate_optimized.md",
    },
    {
        "question": "What is the estimated remaining carbon budget from the beginning of 2020 for a 50% likelihood of limiting global warming to 1.5°C?",
        "ground_truth": "500 GtCO2 (for a 67% likelihood of limiting warming to 2°C, the budget is 1150 GtCO2).",
        "domain_tag": "climate",
        "source_file": "document_2_Climate_optimized.md",
        "notes": "Tests precision on a specific number that has a very similar nearby figure (1150 GtCO2) -- good check against retrieval conflating the two.",
    },
    {
        "question": "Under the very high emissions scenario (SSP5-8.5), what is the likely range of global mean sea level rise by 2100 relative to 1995-2014?",
        "ground_truth": "0.63 to 1.01 meters.",
        "domain_tag": "climate",
        "source_file": "document_2_Climate_optimized.md",
    },

    # ------------------------------------------------------------------
    # ECONOMICS -- IMF World Economic Outlook, April 2026, Ch. 1 (document_3_Economics_optimized.md)
    # ------------------------------------------------------------------
    {
        "question": "What was global growth in the fourth quarter of 2025, on an annualized basis, per the April 2026 WEO?",
        "ground_truth": "3.9 percent on an annualized basis.",
        "domain_tag": "economics",
        "source_file": "document_3_Economics_optimized.md",
    },
    {
        "question": "What is US public debt projected to reach as a percentage of GDP by 2031, and what was it in 2025?",
        "ground_truth": "US public debt is projected to climb from 124 percent of GDP in 2025 to 142 percent in 2031.",
        "domain_tag": "economics",
        "source_file": "document_3_Economics_optimized.md",
    },
    {
        "question": "What is the reference-forecast global growth rate projected for 2026, assuming a relatively short-lived Middle East conflict?",
        "ground_truth": "3.1 percent for 2026 (and 3.2 percent for 2027).",
        "domain_tag": "economics",
        "source_file": "document_3_Economics_optimized.md",
    },
    {
        "question": "How large was China's merchandise goods trade surplus in 2025?",
        "ground_truth": "A record $1.2 trillion, equivalent to 6 percent of GDP.",
        "domain_tag": "economics",
        "source_file": "document_3_Economics_optimized.md",
    },
    {
        "question": "What is the US effective statutory tariff rate underlying the April 2026 WEO projections, compared to the October 2025 forecast?",
        "ground_truth": "13.5 percent in the April 2026 WEO, compared with 18.7 percent in the October 2025 forecast.",
        "domain_tag": "economics",
        "source_file": "document_3_Economics_optimized.md",
    },

    # ------------------------------------------------------------------
    # HEALTH -- WHO World Health Statistics 2025 (document_4_Health_optimized.md)
    # ------------------------------------------------------------------
    {
        "question": "By how many years did global life expectancy at birth increase between 2000 and 2019?",
        "ground_truth": "6.3 years, from 66.8 years in 2000 to 73.1 years in 2019.",
        "domain_tag": "health",
        "source_file": "document_4_Health_optimized.md",
    },
    {
        "question": "What was the global maternal mortality ratio (MMR) in 2023, and how much has it declined since 2000?",
        "ground_truth": (
            "197 maternal deaths per 100,000 live births in 2023, a 40% decline "
            "from 328 per 100,000 live births in 2000."
        ),
        "domain_tag": "health",
        "source_file": "document_4_Health_optimized.md",
    },
    {
        "question": "How much did the global under-five mortality rate decline between 2000 and 2023?",
        "ground_truth": (
            "It declined by more than half (52%), from 77 deaths per 1,000 live "
            "births in 2000 to 37 per 1,000 live births in 2023."
        ),
        "domain_tag": "health",
        "source_file": "document_4_Health_optimized.md",
    },
    {
        "question": "In 2019, how did the African Region's age-standardized death rate for under-70 deaths compare to the global average?",
        "ground_truth": (
            "The African Region had 665 deaths per 100,000 population, over 80% "
            "higher than the global average of 366 per 100,000."
        ),
        "domain_tag": "health",
        "source_file": "document_4_Health_optimized.md",
    },
    {
        "question": "How much did global healthy life expectancy (HALE) at birth drop between 2019 and 2021 due to COVID-19?",
        "ground_truth": "It dropped from 63.5 years in 2019 to 61.9 years in 2021, a decline of 1.6 years.",
        "domain_tag": "health",
        "source_file": "document_4_Health_optimized.md",
    },

    # ------------------------------------------------------------------
    # AI RESEARCH -- "RAG for LLMs: A Survey" (document_5_Natural_optimized.md)
    # ------------------------------------------------------------------
    {
        "question": "What does HyDE stand for, and what does it embed instead of the raw query?",
        "ground_truth": (
            "HyDE stands for Hypothetical Document Embeddings. Instead of embedding "
            "the raw query, it has the LLM construct a hypothetical document (an "
            "assumed answer to the query) and embeds that, focusing on embedding "
            "similarity from answer to answer rather than query to answer."
        ),
        "domain_tag": "ai_research",
        "source_file": "document_5_Natural_optimized.md",
        "notes": "Directly relevant to your own Plan 2 Step 7 -- good cross-check question.",
    },
    {
        "question": "What are the two types of 'reflection tokens' used in Self-RAG?",
        "ground_truth": "\"Retrieve\" and \"critic\" tokens.",
        "domain_tag": "ai_research",
        "source_file": "document_5_Natural_optimized.md",
    },
    {
        "question": "How does FLARE decide when to trigger retrieval during generation?",
        "ground_truth": (
            "FLARE monitors the probability of generated terms; when that "
            "probability falls below a certain threshold, it activates the "
            "retrieval system to collect relevant information."
        ),
        "domain_tag": "ai_research",
        "source_file": "document_5_Natural_optimized.md",
    },
    {
        "question": "According to the survey, what methods can be used to perform reranking of retrieved document chunks?",
        "ground_truth": (
            "Rule-based methods relying on predefined metrics like diversity, "
            "relevance, and MRR, or model-based approaches such as Encoder-Decoder "
            "models (e.g., SpanBERT), specialized reranking models like Cohere "
            "Rerank or bge-reranker-large, and general large language models like GPT."
        ),
        "domain_tag": "ai_research",
        "source_file": "document_5_Natural_optimized.md",
    },
    {
        "question": "What does CRAG stand for in the RAG survey, and how is it categorized in the paper's method comparison table?",
        "ground_truth": (
            "CRAG stands for Corrective Retrieval Augmented Generation. In the "
            "survey's comparison table it is categorized as using Arxiv data, "
            "Text Doc granularity, applied at Inference stage, with a 'Once' "
            "retrieval pattern."
        ),
        "domain_tag": "ai_research",
        "source_file": "document_5_Natural_optimized.md",
        "notes": "Nice self-referential check since your own pipeline implements a CRAG-style loop in Step 3.",
    },

    # ------------------------------------------------------------------
    # OUT-OF-SCOPE -- to exercise Step 2 routing / Step 3 CRAG fallback
    # ------------------------------------------------------------------
    {
        "question": "What is the capital of France?",
        "ground_truth": "This question is unrelated to the document corpus (AI policy, climate, economics, health, AI research) and should be flagged as out of scope.",
        "domain_tag": "out_of_scope",
        "source_file": None,
    },
    {
        "question": "Who won the most recent Super Bowl?",
        "ground_truth": "This question is unrelated to the document corpus and should be flagged as out of scope.",
        "domain_tag": "out_of_scope",
        "source_file": None,
    },
    {
        "question": "Can you give me a recipe for chocolate cake?",
        "ground_truth": "This question is unrelated to the document corpus and should be flagged as out of scope.",
        "domain_tag": "out_of_scope",
        "source_file": None,
    },
    {
        "question": "What's the best programming language for building a web app?",
        "ground_truth": "This question is unrelated to the document corpus and should be flagged as out of scope.",
        "domain_tag": "out_of_scope",
        "source_file": None,
    },
    {
        "question": "Can you recommend a good science fiction movie to watch tonight?",
        "ground_truth": "This question is unrelated to the document corpus and should be flagged as out of scope.",
        "domain_tag": "out_of_scope",
        "source_file": None,
    },
]


def get_by_domain(domain_tag: str):
    """Filter golden set questions by domain_tag."""
    return [q for q in GOLDEN_SET if q["domain_tag"] == domain_tag]


def summary():
    """Print a quick count per domain -- use this to sanity-check balance."""
    from collections import Counter
    counts = Counter(q["domain_tag"] for q in GOLDEN_SET)
    for domain, count in counts.items():
        print(f"{domain:15s}: {count}")
    print(f"{'TOTAL':15s}: {len(GOLDEN_SET)}")


if __name__ == "__main__":
    summary()
