"""
Example 2: RAG Evaluation with Generation Metrics
==================================================

This example shows how to:
1. Generate answers using Azure OpenAI
2. Evaluate the temporal quality of generated answers
3. Use LLM-as-judge for faithfulness, hallucination, coherence

For RAG evaluation you need:
- The query
- Retrieved documents (context)
- Generated answer from your RAG system
- (Optional) Gold answer for comparison

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
    TemporalFaithfulness,
    TemporalHallucination,
    TemporalCoherence,
    AnswerTemporalAlignment,
    TempoScore,
)


def generate_rag_answer(query: str, contexts: list, llm) -> str:
    """Generate answer using RAG approach."""
    
    context_text = "\n\n".join([f"[Doc {i+1}]: {c}" for i, c in enumerate(contexts)])
    
    prompt = f"""Based on the following documents, answer the question.
Be specific about dates and temporal information.

Documents:
{context_text}

Question: {query}

Answer:"""
    
    return llm.generate(prompt)


def main():
    print("=" * 60)
    print("TempoEval - RAG Generation Metrics")
    print("=" * 60)
    
    # ================================================================
    # Step 1: Initialize LLM Provider
    # ================================================================
    
    llm = AzureOpenAIProvider()
    print("\n✓ Azure OpenAI connected")
    
    # ================================================================
    # Step 2: Sample RAG scenario
    # ================================================================
    
    query = "When was Bitcoin pruning introduced and what version included it?"
    
    # Retrieved documents (from your retrieval system)
    retrieved_contexts = [
        "Bitcoin Core version 0.11.0 was released on July 12, 2015. This version introduced pruning mode, which allows nodes to delete old block data while still validating the blockchain.",
        "The pruning feature allows Bitcoin nodes to run with significantly less disk space. Before pruning, a full node required over 40GB of storage.",
        "Version 0.12.0 released in February 2016 improved the pruning feature with better performance.",
    ]
    
    # Gold answer (if available)
    gold_answer = "Bitcoin pruning was introduced in version 0.11.0, released on July 12, 2015."
    
    # ================================================================
    # Step 3: Generate answer with your RAG system
    # ================================================================
    
    print(f"\nQuery: {query}")
    print("\nGenerating answer...")
    
    generated_answer = generate_rag_answer(query, retrieved_contexts, llm)
    print(f"\nGenerated Answer: {generated_answer}")
    
    # ================================================================
    # Step 4: Evaluate generation quality
    # ================================================================
    
    print("\n" + "-" * 40)
    print("Evaluating temporal quality...")
    print("-" * 40)
    
    # Initialize metrics with LLM
    faithfulness = TemporalFaithfulness()
    faithfulness.llm = llm
    
    hallucination = TemporalHallucination()
    hallucination.llm = llm
    
    coherence = TemporalCoherence()
    coherence.llm = llm
    
    alignment = AnswerTemporalAlignment()
    alignment.llm = llm
    
    # Compute metrics
    faith_score = faithfulness.compute(
        answer=generated_answer,
        contexts=retrieved_contexts
    )
    print(f"\n1. Temporal Faithfulness: {faith_score:.3f}")
    print("   (Are temporal claims supported by context?)")
    
    halluc_score = hallucination.compute(
        answer=generated_answer,
        contexts=retrieved_contexts,
        gold_answer=gold_answer
    )
    print(f"\n2. Temporal Hallucination Rate: {halluc_score:.3f}")
    print("   (Lower is better - % of fabricated temporal claims)")
    
    coher_score = coherence.compute(answer=generated_answer)
    print(f"\n3. Temporal Coherence: {coher_score:.3f}")
    print("   (Are temporal statements internally consistent?)")
    
    align_score = alignment.compute(
        query=query,
        answer=generated_answer,
        temporal_focus="specific_time",
        target_anchors=["2015", "0.11"]
    )
    print(f"\n4. Answer Temporal Alignment: {align_score:.3f}")
    print("   (Does answer focus on the right time period?)")
    
    # ================================================================
    # Step 5: Compute TempoScore
    # ================================================================
    
    tempo_scorer = TempoScore()
    
    # For hallucination, lower is better, so we invert
    tempo_result = tempo_scorer.compute(
        temporal_precision=0.8,  # Placeholder for retrieval
        temporal_recall=0.8,    # Placeholder for retrieval
        temporal_faithfulness=faith_score,
        temporal_coherence=coher_score,
    )
    
    print("\n" + "=" * 40)
    print(f"TEMPOSCORE (Weighted): {tempo_result['tempo_weighted']:.3f}")
    print(f"TEMPOSCORE (F-Score):  {tempo_result['tempo_f_score']:.3f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
