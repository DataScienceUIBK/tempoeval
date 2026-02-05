"""
Example 5: Cross-Period Query Evaluation
=========================================

This example shows how to evaluate queries that require
reasoning across multiple time periods, such as:
- "How has X changed since 2015?"
- "Compare X in 2017 vs 2024"

These queries need special metrics like Temporal Coverage
and Cross-Period Reasoning.

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
    TemporalRecall,
    TemporalCoverage,
    AnchorCoverage,
    CrossPeriodReasoning,
    TemporalFaithfulness,
    TempoScore,
)


def main():
    print("=" * 70)
    print("TempoEval - Cross-Period Query Evaluation")
    print("=" * 70)
    
    llm = AzureOpenAIProvider()
    print("\n✓ Azure OpenAI connected")
    
    # ================================================================
    # Cross-period query scenario
    # ================================================================
    
    query = "How has Bitcoin's energy consumption changed from 2017 to 2024?"
    
    baseline_anchor = "2017"
    comparison_anchor = "2024"
    
    # Retrieved documents (must cover BOTH periods)
    retrieved_docs = [
        # Document covering 2017 (baseline)
        "In 2017, Bitcoin mining consumed approximately 30 TWh annually, equivalent to Ireland's annual electricity consumption.",
        
        # Document covering 2024 (comparison)
        "By 2024, Bitcoin's energy consumption reached 150 TWh annually, though efficiency per transaction improved significantly.",
        
        # Document covering the transition
        "Between 2017 and 2024, Bitcoin's hash rate increased 100x, with mining becoming increasingly industrialized.",
        
        # Irrelevant document (no temporal coverage)
        "Bitcoin uses proof-of-work consensus mechanism which requires computational power.",
    ]
    
    # Simulated gold document IDs
    gold_ids = ["doc_2017", "doc_2024", "doc_trend"]
    retrieved_ids = ["doc_2017", "doc_2024", "doc_trend", "doc_generic"]
    
    # Generated answer (from your RAG system)
    generated_answer = """Bitcoin's energy consumption has increased significantly from 2017 to 2024. 
    In 2017, the network consumed about 30 TWh annually. By 2024, this grew to approximately 
    150 TWh per year - a 5x increase. However, efficiency per transaction improved as the 
    network scaled. The growth was driven by a 100x increase in hash rate as mining became 
    more industrialized."""
    
    print(f"\nQuery: {query}")
    print(f"Baseline: {baseline_anchor}")
    print(f"Comparison: {comparison_anchor}")
    print(f"\nGenerated Answer:\n{generated_answer}")
    
    # ================================================================
    # Evaluate with cross-period metrics
    # ================================================================
    
    print("\n" + "=" * 50)
    print("RETRIEVAL METRICS")
    print("=" * 50)
    
    # Standard retrieval
    recall = TemporalRecall().compute(
        retrieved_ids=retrieved_ids,
        gold_ids=gold_ids,
        k=5
    )
    print(f"\n1. Temporal Recall@5: {recall:.3f}")
    
    # Anchor coverage - do we have docs covering both periods?
    anchor_cov = AnchorCoverage().compute(
        retrieved_anchors=["2017", "2024", "2017-2024"],  # Extracted from docs
        required_anchors=[baseline_anchor, comparison_anchor],
    )
    print(f"\n2. Anchor Coverage (F1): {anchor_cov:.3f}")
    print(f"   Required: {[baseline_anchor, comparison_anchor]}")
    
    # Temporal coverage - LLM judges if both periods are covered
    coverage = TemporalCoverage()
    coverage.llm = llm
    
    cov_score = coverage.compute(
        query=query,
        retrieved_docs=retrieved_docs,
        temporal_focus="change_over_time",
        baseline_anchor=baseline_anchor,
        comparison_anchor=comparison_anchor,
        k=5
    )
    print(f"\n3. Temporal Coverage: {cov_score:.3f}")
    print(f"   (1.0 = both periods covered, 0.5 = one period, 0.0 = neither)")
    
    # ================================================================
    # Generation metrics for cross-period
    # ================================================================
    
    print("\n" + "=" * 50)
    print("GENERATION METRICS (Cross-Period)")
    print("=" * 50)
    
    # Cross-period reasoning
    cross_period = CrossPeriodReasoning()
    cross_period.llm = llm
    
    cp_score = cross_period.compute(
        query=query,
        answer=generated_answer,
        baseline_anchor=baseline_anchor,
        comparison_anchor=comparison_anchor,
    )
    print(f"\n4. Cross-Period Reasoning: {cp_score:.3f}")
    print("   Evaluates: period coverage, explicit comparison, quantification")
    
    # Faithfulness
    faithful = TemporalFaithfulness()
    faithful.llm = llm
    
    faith_score = faithful.compute(
        answer=generated_answer,
        contexts=retrieved_docs,
    )
    print(f"\n5. Temporal Faithfulness: {faith_score:.3f}")
    
    # ================================================================
    # Combined TempoScore for cross-period
    # ================================================================
    
    print("\n" + "=" * 50)
    print("COMPOSITE SCORE")
    print("=" * 50)
    
    cross_period_scorer = TempoScore()
    
    final_result = cross_period_scorer.compute(
        temporal_recall=recall,
        temporal_precision=anchor_cov,  # Use anchor coverage as precision proxy
        temporal_faithfulness=faith_score,
        temporal_coherence=cp_score,  # Use cross-period reasoning as coherence
    )
    
    print(f"\n>>> Cross-Period TempoScore (Weighted): {final_result['tempo_weighted']:.3f}")
    print(f">>> Cross-Period TempoScore (F-Score):  {final_result['tempo_f_score']:.3f}")
    
    # Interpretation
    if final_result['tempo_weighted'] >= 0.8:
        interpretation = "Excellent - comprehensive cross-period analysis"
    elif final_result['tempo_weighted'] >= 0.6:
        interpretation = "Good - covers both periods with some gaps"
    elif final_result['tempo_weighted'] >= 0.4:
        interpretation = "Fair - missing coverage of one period"
    else:
        interpretation = "Poor - inadequate cross-period reasoning"
    
    print(f"    Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
