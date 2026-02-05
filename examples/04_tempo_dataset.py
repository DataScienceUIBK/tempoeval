"""
Example 4: Using TEMPO Dataset with Real Data
==============================================

This example shows how to:
1. Load the actual TEMPO benchmark dataset
2. Use your retrieval scores
3. Evaluate with temporal metrics

No LLM needed for basic retrieval metrics!
"""

from tempoeval.datasets import load_tempo, TemporalDataset, TemporalQuery
from tempoeval.metrics import (
    TemporalRecall,
    TemporalNDCG,
    TemporalMRR,
    AnchorCoverage,
)


def load_retrieval_scores(scores_path: str) -> dict:
    """Load retrieval scores from your model.
    
    Expected format:
    {
        "query_id_1": {"doc_id_1": 0.9, "doc_id_2": 0.8, ...},
        "query_id_2": {...}
    }
    """
    import json
    with open(scores_path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("TempoEval - TEMPO Benchmark Evaluation")
    print("=" * 60)
    
    # ================================================================
    # Step 1: Load TEMPO dataset
    # ================================================================
    
    domain = "bitcoin"  # Choose: bitcoin, cardano, economics, etc.
    
    print(f"\nLoading TEMPO/{domain} dataset...")
    
    try:
        dataset = load_tempo(domain=domain)
        print(f"✓ Loaded {len(dataset)} queries")
        
        # Optionally load documents
        # documents = load_tempo_documents(domain=domain)
        # print(f"✓ Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Note: TEMPO loading requires 'datasets' package: {e}")
        print("Using sample data instead...")
        
        # Fall back to sample data
        dataset = TemporalDataset(
            queries=[
                TemporalQuery(
                    id="q1",
                    query="When was Bitcoin pruning introduced?",
                    gold_ids=["doc_1", "doc_2"],
                    temporal_focus="specific_time",
                    key_time_anchors=["2015"],
                ),
                TemporalQuery(
                    id="q2",
                    query="How has Bitcoin block size changed since 2015?",
                    gold_ids=["doc_3", "doc_4", "doc_5"],
                    temporal_focus="change_over_time",
                    key_time_anchors=["2015", "2017"],
                    is_cross_period=True,
                ),
            ],
            name="sample_tempo"
        )
    
    # ================================================================
    # Step 2: Simulated retrieval results (replace with your BM25)
    # ================================================================
    
    # In real usage, you would load from your retrieval system:
    # retrieval_scores = load_retrieval_scores("path/to/scores.json")
    
    simulated_retrieval = {
        "q1": ["doc_1", "doc_3", "doc_2", "doc_5", "doc_4"],
        "q2": ["doc_3", "doc_5", "doc_1", "doc_4", "doc_6"],
    }
    
    # ================================================================
    # Step 3: Initialize metrics
    # ================================================================
    
    metrics = {
        "Temporal Recall": TemporalRecall(),
        "Temporal NDCG": TemporalNDCG(),
        "Temporal MRR": TemporalMRR(),
        "Anchor Coverage": AnchorCoverage(),
    }
    
    k_values = [5, 10, 20]
    
    # ================================================================
    # Step 4: Evaluate
    # ================================================================
    
    print("\n" + "-" * 50)
    print("Evaluation Results")
    print("-" * 50)
    
    # Aggregate scores
    aggregated = {k: {m: [] for m in metrics} for k in k_values}
    
    for query in dataset.queries:
        retrieved = simulated_retrieval.get(query.id, [])
        
        if not retrieved:
            continue
        
        print(f"\nQuery {query.id}: {query.query[:50]}...")
        
        for k in k_values:
            recall = metrics["Temporal Recall"].compute(
                retrieved_ids=retrieved,
                gold_ids=query.gold_ids,
                k=k
            )
            ndcg = metrics["Temporal NDCG"].compute(
                retrieved_ids=retrieved,
                gold_ids=query.gold_ids,
                k=k
            )
            mrr = metrics["Temporal MRR"].compute(
                retrieved_ids=retrieved,
                gold_ids=query.gold_ids,
                k=k
            )
            anchor = metrics["Anchor Coverage"].compute(
                retrieved_anchors=["2015", "2017"],  # Extracted from docs
                required_anchors=query.key_time_anchors,
                k=k
            )
            
            aggregated[k]["Temporal Recall"].append(recall)
            aggregated[k]["Temporal NDCG"].append(ndcg)
            aggregated[k]["Temporal MRR"].append(mrr)
            aggregated[k]["Anchor Coverage"].append(anchor)
            
            print(f"  @{k}: R={recall:.2f}, NDCG={ndcg:.2f}, MRR={mrr:.2f}, Anchor={anchor:.2f}")
    
    # ================================================================
    # Step 5: Print summary
    # ================================================================
    
    print("\n" + "=" * 50)
    print("AGGREGATED RESULTS")
    print("=" * 50)
    
    for k in k_values:
        print(f"\n@{k}:")
        for metric_name, scores in aggregated[k].items():
            if scores:
                import math
                valid_scores = [s for s in scores if not math.isnan(s)]
                if valid_scores:
                    avg = sum(valid_scores) / len(valid_scores)
                    print(f"  {metric_name}: {avg:.3f}")


if __name__ == "__main__":
    main()
