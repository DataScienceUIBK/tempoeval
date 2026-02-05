"""
Example 09: BM25 Retrieval on TEMPO HSM (History of Science and Medicine)
=========================================================================

This example demonstrates how to evaluate BM25 retrieval on the TEMPO benchmark.
It uses the EfficiencyTracker to report cost and latency.

Dataset: tempo26/Tempo (subset: hsm)
Retriever: BM25
Metrics: Temporal Recall, NDCG, MRR

No LLM needed for basic retrieval metrics!
"""

import logging
from tempoeval.datasets import load_tempo, load_tempo_documents
from tempoeval.retrieval import BM25Retriever
from tempoeval.core import evaluate
from tempoeval.metrics import (
    TemporalRecall, 
    TemporalNDCG, 
    TemporalMRR,
    TemporalDiversity
)
from tempoeval.efficiency import EfficiencyTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("--- 1. Loading TEMPO HSM Dataset ---")
    # Load queries (limit to 20 for speed in this demo)
    queries = load_tempo(domain="hsm", max_samples=20)
    print(f"Loaded {len(queries)} queries.")
    
    print("\n--- 2. Loading Corpus Documents ---")
    documents = load_tempo_documents(domain="hsm")
    print(f"Loaded {len(documents)} documents.")
    
    # Prepare documents for BM25
    # BM25Retriever expects a dict of id -> text or list of texts
    # We pass the full corpus
    
    print("\n--- 3. initializing BM25 Retriever ---")
    retriever = BM25Retriever(
        index_path="./bm25_hsm_index",
        force_rebuild=True
    )
    
    # Add corpus
    doc_ids = list(documents.keys())
    doc_texts = list(documents.values())
    retriever.index(doc_texts, doc_ids)
    
    print("\n--- 4. Running Retrieval ---")
    # Retrieve top-10 for evaluation
    query_texts = [q.query for q in queries]
    query_ids = [q.id for q in queries]
    
    results = retriever.retrieve(
        queries=query_texts,
        query_ids=query_ids,
        top_k=20
    )
    
    # Attach retrieved results to query objects for evaluation
    eval_data = []
    for q in queries:
        if q.id in results:
            hits = results[q.id]
            # Convert to format expected by metrics
            # (metrics usually take 'retrieved_ids')
            q_dict = q.to_dict()
            q_dict["retrieved_ids"] = [h["id"] for h in hits]
            q_dict["retrieved_docs"] = [h["content"] for h in hits]  # For content-based metrics
            eval_data.append(q_dict)

    print("\n--- 5. initializing Efficiency Tracker ---")
    tracker = EfficiencyTracker(model_name="bm25-cpu")
    
    print("\n--- 6. Running Evaluation ---")
    metrics = [
        TemporalRecall(),
        TemporalNDCG(),
        TemporalMRR(),
        TemporalDiversity()
    ]
    
    evaluation_result = evaluate(
        data=eval_data,
        metrics=metrics,
        k_values=[5, 10, 20],
        efficiency_tracker=tracker
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(evaluation_result.summary_table())
    
    print("\n" + "=" * 50)
    print("EFFICIENCY REPORT")
    print("=" * 50)
    eff = evaluation_result.metadata.get("efficiency", {})
    print(f"Total Operations: {eff.get('total_operations', 0)}")
    print(f"Total Latency:    {eff.get('total_latency_seconds', 0)}s")
    print(f"Avg Latency/Op:   {eff.get('avg_latency_ms', 0)}ms")
    print(f"Est Cost:         ${eff.get('total_cost_usd', 0)}")
    
    # Generate Leaderboard CSV line
    print("\nLeaderboard CSV Line:")
    print(f"BM25,HSM,{evaluation_result.summary().get('temporal_recall@10', 0):.3f},{evaluation_result.summary().get('temporal_ndcg@10', 0):.3f}")


if __name__ == "__main__":
    main()
