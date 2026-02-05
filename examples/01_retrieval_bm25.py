"""
Example 1: Retrieval Evaluation with BM25
=========================================

This example shows how to:
1. Load sample temporal data
2. Simulate BM25 retrieval
3. Evaluate retrieval quality with temporal metrics

No LLM needed for basic retrieval metrics!
"""

from tempoeval.metrics import (
    TemporalRecall, 
    TemporalNDCG, 
    TemporalMRR,
    AnchorCoverage,
    TemporalDiversity,
)
from tempoeval.datasets import TemporalQuery


def simple_bm25_retrieval(query: str, documents: dict, k: int = 5) -> list:
    """Simple keyword-based retrieval (simulates BM25)."""
    query_terms = set(query.lower().split())
    scores = []
    for doc_id, doc_text in documents.items():
        doc_terms = set(doc_text.lower().split())
        overlap = len(query_terms & doc_terms)
        scores.append((doc_id, overlap))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:k]]


def main():
    # ================================================================
    # Step 1: Prepare sample data
    # ================================================================
    
    # Sample documents with temporal content
    documents = {
        "doc_1": "Bitcoin pruning was introduced in version 0.11 released in July 2015.",
        "doc_2": "The Lightning Network was proposed in 2015 and launched on mainnet in 2018.",
        "doc_3": "Ethereum was launched on July 30, 2015 by Vitalik Buterin.",
        "doc_4": "Bitcoin's block size debate intensified in 2016-2017.",
        "doc_5": "The SegWit upgrade was activated on Bitcoin in August 2017.",
        "doc_6": "Bitcoin reached $20,000 for the first time in December 2017.",
        "doc_7": "The Taproot upgrade was activated on Bitcoin in November 2021.",
        "doc_8": "Bitcoin ETFs were approved by the SEC in January 2024.",
    }
    
    # Sample queries with gold temporal document annotations
    queries = [
        TemporalQuery(
            id="q1",
            query="When was Bitcoin pruning introduced?",
            gold_ids=["doc_1"],  # Ground truth
            temporal_focus="specific_time",
            key_time_anchors=["2015"],
        ),
        TemporalQuery(
            id="q2", 
            query="What happened to Bitcoin in 2017?",
            gold_ids=["doc_4", "doc_5", "doc_6"],  # Multiple relevant docs
            temporal_focus="specific_time",
            key_time_anchors=["2017"],
        ),
        TemporalQuery(
            id="q3",
            query="How has Bitcoin evolved since 2015?",
            gold_ids=["doc_1", "doc_2", "doc_4", "doc_5", "doc_7"],
            temporal_focus="change_over_time",
            key_time_anchors=["2015", "2017", "2021"],
            is_cross_period=True,
        ),
    ]
    
    # ================================================================
    # Step 2: Simulate BM25 retrieval
    # ================================================================
    
    # Simulated retrieval results
    retrieval_results = {
        "q1": ["doc_1", "doc_3", "doc_2", "doc_7", "doc_8"],  # doc_1 at rank 1 ✓
        "q2": ["doc_6", "doc_5", "doc_4", "doc_8", "doc_7"],  # All 3 in top 3 ✓
        "q3": ["doc_7", "doc_5", "doc_1", "doc_8", "doc_2"],  # Good but missing some
    }
    
    # ================================================================
    # Step 3: Evaluate with TempoEval metrics
    # ================================================================
    
    print("=" * 60)
    print("TempoEval - Retrieval Metrics Evaluation")
    print("=" * 60)
    
    # Initialize metrics (no LLM needed!)
    recall_metric = TemporalRecall()
    ndcg_metric = TemporalNDCG()
    mrr_metric = TemporalMRR()
    anchor_metric = AnchorCoverage()
    diversity_metric = TemporalDiversity()
    
    k_values = [3, 5, 10]
    
    for query in queries:
        print(f"\nQuery: {query.query}")
        print(f"Gold docs: {query.gold_ids}")
        print(f"Retrieved: {retrieval_results[query.id][:5]}")
        
        for k in k_values:
            recall = recall_metric.compute(
                retrieved_ids=retrieval_results[query.id],
                gold_ids=query.gold_ids,
                k=k
            )
            ndcg = ndcg_metric.compute(
                retrieved_ids=retrieval_results[query.id],
                gold_ids=query.gold_ids,
                k=k
            )
            mrr = mrr_metric.compute(
                retrieved_ids=retrieval_results[query.id],
                gold_ids=query.gold_ids,
                k=k
            )
            
            print(f"  @{k}: Recall={recall:.3f}, NDCG={ndcg:.3f}, MRR={mrr:.3f}")
    
    # ================================================================
    # Step 4: Aggregate results
    # ================================================================
    
    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)
    
    for k in k_values:
        recalls = []
        ndcgs = []
        
        for query in queries:
            recalls.append(recall_metric.compute(
                retrieved_ids=retrieval_results[query.id],
                gold_ids=query.gold_ids,
                k=k
            ))
            ndcgs.append(ndcg_metric.compute(
                retrieved_ids=retrieval_results[query.id],
                gold_ids=query.gold_ids,
                k=k
            ))
        
        avg_recall = sum(recalls) / len(recalls)
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        
        print(f"@{k}: Avg Recall={avg_recall:.3f}, Avg NDCG={avg_ndcg:.3f}")


if __name__ == "__main__":
    main()
