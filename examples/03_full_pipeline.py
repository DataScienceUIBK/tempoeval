"""
Example 3: Full Pipeline - Retrieval + Generation Evaluation
=============================================================

This example shows a complete evaluation pipeline:
1. Simple keyword retrieval
2. Generate answers with Azure OpenAI
3. Evaluate both retrieval AND generation metrics
4. Compute combined TempoScore

Setup:
    1. Copy examples/.env.example to examples/.env
    2. Fill in your Azure OpenAI credentials
    3. Run this script
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Verify environment is configured
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    print("ERROR: Azure OpenAI credentials not found!")
    print("Please copy examples/.env.example to examples/.env and fill in your credentials.")
    exit(1)

from tempoeval.llm import AzureOpenAIProvider
from tempoeval.metrics import (
    # Retrieval metrics
    TemporalRecall,
    TemporalNDCG,
    TemporalMRR,
    # Generation metrics
    TemporalFaithfulness,
    TemporalCoherence,
    # Composite
    TempoScore,
)
from tempoeval.datasets import TemporalQuery


def simple_bm25_retrieval(query: str, documents: dict, k: int = 5) -> list:
    """Simple keyword-based retrieval (replace with real BM25)."""
    query_terms = set(query.lower().split())
    scores = []
    for doc_id, doc_text in documents.items():
        doc_terms = set(doc_text.lower().split())
        overlap = len(query_terms & doc_terms)
        scores.append((doc_id, overlap))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:k]]


def generate_answer(query: str, contexts: list, llm) -> str:
    """Generate answer using LLM."""
    context_text = "\n".join([f"- {c}" for c in contexts])
    prompt = f"""Answer the question based on these documents:

{context_text}

Question: {query}
Answer (be specific about dates):"""
    
    return llm.generate(prompt)


def main():
    print("=" * 70)
    print("TempoEval - Full RAG Pipeline Evaluation")
    print("=" * 70)
    
    # ================================================================
    # Setup
    # ================================================================
    
    llm = AzureOpenAIProvider()
    print("\n✓ Azure OpenAI connected")
    
    # Document corpus
    documents = {
        "doc_1": "Bitcoin pruning was introduced in version 0.11 in July 2015.",
        "doc_2": "SegWit was activated on Bitcoin mainnet on August 24, 2017.",
        "doc_3": "The Lightning Network launched on Bitcoin mainnet in March 2018.",
        "doc_4": "Taproot upgrade activated on November 14, 2021.",
        "doc_5": "Bitcoin hit $69,000 all-time high in November 2021.",
        "doc_6": "Bitcoin ETFs were approved by SEC in January 2024.",
    }
    
    # Queries with ground truth
    queries = [
        TemporalQuery(
            id="q1",
            query="When was Bitcoin pruning introduced?",
            gold_ids=["doc_1"],
            key_time_anchors=["2015"],
        ),
        TemporalQuery(
            id="q2",
            query="What major Bitcoin upgrades happened in 2017?",
            gold_ids=["doc_2"],
            key_time_anchors=["2017"],
        ),
    ]
    
    # ================================================================
    # Metrics initialization
    # ================================================================
    
    recall_metric = TemporalRecall()
    ndcg_metric = TemporalNDCG()
    mrr_metric = TemporalMRR()
    
    faithfulness_metric = TemporalFaithfulness()
    faithfulness_metric.llm = llm
    
    coherence_metric = TemporalCoherence()
    coherence_metric.llm = llm
    
    # ================================================================
    # Evaluate each query
    # ================================================================
    
    all_results = []
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query.query}")
        print("=" * 50)
        
        # Step 1: Retrieval
        retrieved_ids = simple_bm25_retrieval(query.query, documents, k=3)
        retrieved_docs = [documents[did] for did in retrieved_ids]
        
        print(f"\nRetrieved docs: {retrieved_ids}")
        
        # Step 2: Retrieval metrics
        recall = recall_metric.compute(
            retrieved_ids=retrieved_ids,
            gold_ids=query.gold_ids,
            k=3
        )
        ndcg = ndcg_metric.compute(
            retrieved_ids=retrieved_ids,
            gold_ids=query.gold_ids,
            k=3
        )
        mrr = mrr_metric.compute(
            retrieved_ids=retrieved_ids,
            gold_ids=query.gold_ids,
            k=3
        )
        
        print(f"\nRetrieval Metrics:")
        print(f"  Recall@3:    {recall:.3f}")
        print(f"  NDCG@3:      {ndcg:.3f}")
        print(f"  MRR@3:       {mrr:.3f}")
        
        # Step 3: Generate answer
        answer = generate_answer(query.query, retrieved_docs, llm)
        print(f"\nGenerated Answer: {answer}")
        
        # Step 4: Generation metrics
        faithfulness = faithfulness_metric.compute(
            answer=answer,
            contexts=retrieved_docs
        )
        coherence = coherence_metric.compute(answer=answer)
        
        print(f"\nGeneration Metrics:")
        print(f"  Faithfulness: {faithfulness:.3f}")
        print(f"  Coherence:    {coherence:.3f}")
        
        # Step 5: Combined score
        combined = TempoScore()
        
        score = combined.compute(
            temporal_recall=recall,
            temporal_precision=1.0,  # Placeholder
            temporal_faithfulness=faithfulness,
            temporal_coherence=coherence,
        )
        
        print(f"\n>>> TempoScore (Weighted): {score['tempo_weighted']:.3f}")
        print(f">>> TempoScore (F-Score):  {score['tempo_f_score']:.3f}")
        
        all_results.append({
            "query_id": query.id,
            "recall": recall,
            "ndcg": ndcg,
            "faithfulness": faithfulness,
            "coherence": coherence,
            "tempo_score": score['tempo_weighted'],
        })
    
    # ================================================================
    # Summary
    # ================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_score = sum(r["tempo_score"] for r in all_results) / len(all_results)
    print(f"Average TempoScore: {avg_score:.3f}")


if __name__ == "__main__":
    main()
