"""
Generate Temporal Guidance for Custom Datasets
================================================

This example shows how to generate temporal annotations for datasets
that don't have pre-existing guidance (unlike TEMPO).

The TemporalGuidanceGenerator can:
1. Analyze queries to extract temporal intent, signals, and retrieval plans
2. Annotate passages with temporal information (signals, events, time scope)

This is useful when you want to use TempoEval metrics on your own dataset.

Requirements:
    pip install openai python-dotenv

Setup:
    1. Copy examples/.env.example to examples/.env
    2. Fill in your Azure OpenAI credentials
    3. Run this script
"""

import os
import json
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
from tempoeval.guidance import TemporalGuidanceGenerator


def main():
    print("=" * 60)
    print("Temporal Guidance Generator Demo")
    print("=" * 60)
    
    # Initialize LLM provider
    print("\n[1] Initializing Azure OpenAI...")
    llm = AzureOpenAIProvider()
    print("  ✓ Connected")
    
    # Initialize guidance generator
    print("\n[2] Creating TemporalGuidanceGenerator...")
    generator = TemporalGuidanceGenerator(
        llm=llm,
        max_retries=2,
        concurrency=10,
        max_passage_chars=6000,
    )
    print("  ✓ Ready")
    
    # Example queries to analyze
    queries = [
        "When did the French Revolution begin?",
        "How long did the Roman Empire last?",
        "What events led to World War I?",
        "Who was the president of the United States in 1960?",
    ]
    
    # Example passages to annotate
    passages = [
        {
            "doc_id": "doc_001",
            "text": "The French Revolution began in 1789 with the storming of the Bastille on July 14. This pivotal event marked the beginning of a decade of political upheaval that would reshape France and influence revolutions worldwide."
        },
        {
            "doc_id": "doc_002", 
            "text": "The Roman Empire traditionally began with the reign of Augustus in 27 BC and ended with the fall of Constantinople in 1453 AD, spanning nearly 1,500 years of history across Europe and the Mediterranean."
        },
    ]
    
    # ================================================================
    # Demo 1: Generate Query Guidance
    # ================================================================
    print("\n" + "=" * 60)
    print("DEMO 1: Query Guidance Generation")
    print("=" * 60)
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        guidance = generator.generate_query_guidance(query)
        
        print(f"  Is Temporal: {guidance.is_temporal_query}")
        print(f"  Intent: {guidance.temporal_intent}")
        print(f"  Time Anchors: {guidance.key_time_anchors}")
        print(f"  Signals: {guidance.query_temporal_signals}")
        print(f"  Granularity: {guidance.expected_granularity}")
        print(f"  Reasoning Class: {guidance.temporal_reasoning_class_primary}")
        print(f"  Summary: {guidance.query_summary[:80]}...")
    
    # ================================================================
    # Demo 2: Passage Annotation
    # ================================================================
    print("\n" + "=" * 60)
    print("DEMO 2: Passage Annotation")
    print("=" * 60)
    
    query = "When did the French Revolution begin?"
    
    for p in passages:
        print(f"\n📄 Document: {p['doc_id']}")
        print(f"   Text: {p['text'][:80]}...")
        print("-" * 40)
        
        annotation = generator.annotate_passage(
            query=query,
            passage=p["text"],
            doc_id=p["doc_id"],
        )
        
        print(f"  Confidence: {annotation.confidence:.2f}")
        print(f"  Temporal Signals: {annotation.passage_temporal_signals}")
        print(f"  Temporal Events: {annotation.passage_temporal_events}")
        print(f"  Time Mentions: {annotation.time_mentions}")
        print(f"  Time Scope: {annotation.time_scope_guess}")
        print(f"  Tense: {annotation.tense_guess}")
    
    # ================================================================
    # Demo 3: Full Annotation (Query + Passages)
    # ================================================================
    print("\n" + "=" * 60)
    print("DEMO 3: Full Annotation Output")
    print("=" * 60)
    
    full_annotation = generator.generate_full_annotation(
        query_id="q_001",
        query="When did the French Revolution begin?",
        gold_passages=[
            ("doc_001", passages[0]["text"]),
        ],
        negative_passages=[
            ("doc_002", passages[1]["text"]),
        ],
    )
    
    print("\nGenerated Annotation (TEMPO-compatible format):")
    print(json.dumps(full_annotation, indent=2, default=str))
    
    # Save to file
    output_path = "generated_guidance.json"
    with open(output_path, "w") as f:
        json.dump(full_annotation, f, indent=2, default=str)
    print(f"\n✓ Saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("✓ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
