"""
Debug script: shows full retrieved chunk content + distance scores for a question,
so we can diagnose why generation refused despite relevant-seeming retrieval.
"""
from ingestion.vectorstore import load_vectorstore

QUESTIONS = [
    "What are the key findings of the IPCC AR6 report on climate change?",
    "What economic risks does AI pose according to policy frameworks?",
]

def debug(question):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(question, k=4)

    print(f"\n{'='*80}\nQUESTION: {question}\n{'='*80}")
    for i, (doc, distance) in enumerate(results):
        print(f"\n--- Chunk {i+1} | distance={distance:.4f} | source={doc.metadata.get('source')} ---")
        print(doc.page_content[:400])


if __name__ == "__main__":
    for q in QUESTIONS:
        debug(q)