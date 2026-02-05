"""
Complete Evaluation Pipeline with Generated Guidance
=====================================================

This example demonstrates a full evaluation pipeline where temporal
guidance is GENERATED using LLM (for datasets without pre-existing annotations).

Pipeline:
1. Load dataset using HuggingFace load_dataset (TimeQA)
2. Generate temporal guidance using TemporalGuidanceGenerator
3. Build simple corpus and retrieve with BM25
4. Generate answers with GPT-4
5. Evaluate all temporal metrics
6. Compute TempoScore

This is useful for custom datasets or HF datasets without TEMPO-style annotations.

Requirements:
    pip install datasets pyserini gensim openai python-dotenv

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

from tqdm import tqdm

# TempoEval imports
from tempoeval.llm import AzureOpenAIProvider
from tempoeval.guidance import TemporalGuidanceGenerator
from tempoeval.metrics import (
    # Retrieval metrics
    TemporalRecall,
    TemporalPrecision,
    TemporalNDCG,
    TemporalMRR,
    TemporalCoverage,
    AnchorCoverage,
    TemporalDiversity,
    TemporalRelevance,
    NDCGFullCoverage,
    # Generation metrics
    TemporalFaithfulness,
    TemporalHallucination,
    TemporalCoherence,
    AnswerTemporalAlignment,
    # Reasoning metrics
    EventOrdering,
    DurationAccuracy,
    CrossPeriodReasoning,
    # Composite
    TempoScore,
)


# ================================================================
# Helper Functions
# ================================================================
def to_list(val, default=None):
    """Convert value to Python list, handling numpy arrays."""
    if val is None:
        return default if default is not None else []
    if hasattr(val, 'tolist'):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    try:
        return [val]
    except:
        return default if default is not None else []


def simple_bm25_retrieval(queries, query_ids, documents, doc_ids, k=10):
    """Simple BM25-like keyword retrieval."""
    try:
        from pyserini import analysis
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity
        
        analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        corpus = [analyzer.analyze(x) for x in documents]
        dictionary = Dictionary(corpus)
        model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
        bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
        bm25_index = SparseMatrixSimilarity(
            bm25_corpus, 
            num_docs=len(corpus), 
            num_terms=len(dictionary),
            normalize_queries=False, 
            normalize_documents=False
        )
        
        all_results = {}
        for query_id, query_text in zip(query_ids, queries):
            query_tokens = analyzer.analyze(query_text)
            bm25_query = model[dictionary.doc2bow(query_tokens)]
            similarities = bm25_index[bm25_query].tolist()
            
            scored = list(zip(doc_ids, similarities))
            scored.sort(key=lambda x: x[1], reverse=True)
            all_results[str(query_id)] = [doc_id for doc_id, _ in scored[:k]]
        
        return all_results
        
    except ImportError:
        # Fallback to simple keyword matching
        all_results = {}
        for query_id, query_text in zip(query_ids, queries):
            query_terms = set(query_text.lower().split())
            scores = []
            for doc_id, doc_text in zip(doc_ids, documents):
                doc_terms = set(doc_text.lower().split())
                overlap = len(query_terms & doc_terms)
                scores.append((doc_id, overlap))
            scores.sort(key=lambda x: x[1], reverse=True)
            all_results[str(query_id)] = [doc_id for doc_id, _ in scores[:k]]
        return all_results


def generate_answer(query: str, contexts: list, llm) -> str:
    """Generate answer using GPT-4."""
    context_text = "\n\n".join([f"[Document {i+1}]:\n{c[:1500]}" for i, c in enumerate(contexts[:5])])
    
    prompt = f"""Based on the following documents, answer the question.
Be specific about dates, time periods, and temporal information.
If the documents don't contain enough information, say so.

Documents:
{context_text}

Question: {query}

Answer (include specific dates and temporal details):"""
    
    try:
        return llm.generate(prompt)
    except Exception as e:
        return f"Error generating answer: {e}"


# ================================================================
# Main Pipeline
# ================================================================
def main():
    print("=" * 70)
    print("TempoEval - Pipeline with Generated Guidance")
    print("=" * 70)
    
    # ================================================================
    # Step 1: Load Dataset using load_dataset
    # ================================================================
    print("\n[1/7] Loading TimeQA dataset...")
    
    try:
        from datasets import load_dataset
        
        # Load TimeQA - a temporal QA dataset
        ds = load_dataset("hugosousa/TimeQA", split="validation")
        
        print(f"  ✓ Loaded {len(ds)} samples from TimeQA")
        
        # Extract queries and create simple corpus from contexts
        queries = []
        query_ids = []
        documents = []
        doc_ids = []
        gold_doc_map = {}  # query_id -> list of gold doc_ids
        
        # Process first N samples for demo
        max_samples = 10
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            
            qid = f"q_{i}"
            query = sample.get("question", "")
            
            # TimeQA has contexts
            contexts = sample.get("context", []) or []
            if isinstance(contexts, str):
                contexts = [contexts]
            
            # Add to corpus
            gold_ids = []
            for j, ctx in enumerate(contexts[:3]):  # Limit contexts
                doc_id = f"doc_{i}_{j}"
                documents.append(ctx)
                doc_ids.append(doc_id)
                gold_ids.append(doc_id)
            
            if query and gold_ids:
                queries.append(query)
                query_ids.append(qid)
                gold_doc_map[qid] = gold_ids
        
        print(f"  ✓ Created corpus with {len(documents)} documents")
        print(f"  ✓ Processing {len(queries)} queries")
        
    except Exception as e:
        print(f"  ⚠ Error loading TimeQA: {e}")
        print("  Using sample data...")
        
        # Sample temporal QA data
        queries = [
            "When did World War I begin?",
            "How long did the Roman Empire last?", 
            "Who was the president of the United States in 1960?",
            "What happened after the fall of the Berlin Wall?",
            "When was the first iPhone released?",
        ]
        query_ids = ["q_0", "q_1", "q_2", "q_3", "q_4"]
        
        documents = [
            "World War I began on July 28, 1914, when Austria-Hungary declared war on Serbia following the assassination of Archduke Franz Ferdinand.",
            "The war lasted until November 11, 1918, when the Armistice was signed.",
            "The Roman Empire was founded in 27 BC by Augustus and lasted until 476 AD in the West.",
            "The Eastern Roman Empire continued until 1453 AD when Constantinople fell.",
            "John F. Kennedy was elected president of the United States in 1960, defeating Richard Nixon.",
            "The Berlin Wall fell on November 9, 1989, leading to German reunification in 1990.",
            "The first iPhone was announced by Steve Jobs on January 9, 2007 and released on June 29, 2007.",
        ]
        doc_ids = ["doc_0_0", "doc_0_1", "doc_1_0", "doc_1_1", "doc_2_0", "doc_3_0", "doc_4_0"]
        
        gold_doc_map = {
            "q_0": ["doc_0_0", "doc_0_1"],
            "q_1": ["doc_1_0", "doc_1_1"],
            "q_2": ["doc_2_0"],
            "q_3": ["doc_3_0"],
            "q_4": ["doc_4_0"],
        }
    
    doc_map = {doc_id: doc for doc_id, doc in zip(doc_ids, documents)}
    
    # ================================================================
    # Step 2: Initialize LLM and Generator
    # ================================================================
    print("\n[2/7] Initializing Azure OpenAI and Guidance Generator...")
    
    llm = AzureOpenAIProvider()
    print("  ✓ Azure OpenAI connected")
    
    guidance_generator = TemporalGuidanceGenerator(
        llm=llm,
        max_retries=2,
        concurrency=5,
    )
    print("  ✓ TemporalGuidanceGenerator ready")
    
    # ================================================================
    # Step 3: Generate Temporal Guidance for Queries
    # ================================================================
    print("\n[3/7] Generating temporal guidance for queries...")
    
    guidance_map = {}
    for qid, query in tqdm(zip(query_ids, queries), total=len(queries), desc="Generating guidance"):
        guidance = guidance_generator.generate_query_guidance(query)
        guidance_map[qid] = guidance.to_dict()
    
    print(f"  ✓ Generated guidance for {len(guidance_map)} queries")
    
    # Show sample guidance
    sample_qid = query_ids[0]
    sample_guidance = guidance_map[sample_qid]
    print(f"\n  Sample guidance for: '{queries[0][:50]}...'")
    print(f"    - Is Temporal: {sample_guidance['is_temporal_query']}")
    print(f"    - Intent: {sample_guidance['temporal_intent']}")
    print(f"    - Time Anchors: {sample_guidance['key_time_anchors']}")
    
    # ================================================================
    # Step 4: BM25 Retrieval
    # ================================================================
    print("\n[4/7] Running BM25 retrieval...")
    
    k = 5
    retrieval_results = simple_bm25_retrieval(queries, query_ids, documents, doc_ids, k=k)
    print(f"  ✓ Retrieved top-{k} documents for each query")
    
    # ================================================================
    # Step 5: Generate Answers
    # ================================================================
    print("\n[5/7] Generating answers with GPT-4...")
    
    generated_answers = {}
    for qid, query in tqdm(zip(query_ids, queries), total=len(queries), desc="Generating answers"):
        retrieved_doc_ids = retrieval_results.get(qid, [])[:5]
        contexts = [doc_map.get(did, "") for did in retrieved_doc_ids if did in doc_map]
        
        if contexts:
            answer = generate_answer(query, contexts, llm)
        else:
            answer = "No relevant documents found."
        
        generated_answers[qid] = answer
    
    print(f"  ✓ Generated {len(generated_answers)} answers")
    
    # ================================================================
    # Step 6: Initialize Metrics and Evaluate
    # ================================================================
    print("\n[6/7] Evaluating all metrics...")
    
    # Initialize metrics
    retrieval_metrics = {
        "temporal_recall": TemporalRecall(),
        "temporal_ndcg": TemporalNDCG(),
        "temporal_mrr": TemporalMRR(),
        "anchor_coverage": AnchorCoverage(),
        "temporal_diversity": TemporalDiversity(),
    }
    
    temporal_precision = TemporalPrecision()
    temporal_precision.llm = llm
    
    temporal_relevance = TemporalRelevance()
    temporal_relevance.llm = llm
    
    ndcg_full_coverage = NDCGFullCoverage()
    ndcg_full_coverage.llm = llm
    
    gen_metrics = {
        "temporal_faithfulness": TemporalFaithfulness(),
        "temporal_hallucination": TemporalHallucination(),
        "temporal_coherence": TemporalCoherence(),
        "answer_alignment": AnswerTemporalAlignment(),
    }
    for m in gen_metrics.values():
        m.llm = llm
    
    reasoning_metrics = {
        "event_ordering": EventOrdering(),
    }
    for m in reasoning_metrics.values():
        m.llm = llm
    
    # Evaluate
    all_results = []
    
    for qid, query in tqdm(zip(query_ids, queries), total=len(queries), desc="Evaluating"):
        retrieved_ids = retrieval_results.get(qid, [])
        retrieved_docs = [doc_map.get(did, "") for did in retrieved_ids if did in doc_map]
        gold_ids = gold_doc_map.get(qid, [])
        answer = generated_answers.get(qid, "")
        guidance = guidance_map.get(qid, {})
        
        # Extract from generated guidance
        temporal_focus = guidance.get("temporal_reasoning_class_primary", "specific_time") or "specific_time"
        key_anchors = to_list(guidance.get("key_time_anchors"), [])
        is_cross_period = "cross_period" in temporal_focus.lower() or "trend" in temporal_focus.lower()
        
        scores = {"query_id": qid, "query": query[:50]}
        
        # --- Retrieval Metrics ---
        scores["temporal_recall@5"] = retrieval_metrics["temporal_recall"].compute(
            retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=5
        )
        scores["temporal_ndcg@5"] = retrieval_metrics["temporal_ndcg"].compute(
            retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=5
        )
        scores["temporal_mrr@5"] = retrieval_metrics["temporal_mrr"].compute(
            retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=5
        )
        scores["anchor_coverage"] = retrieval_metrics["anchor_coverage"].compute(
            retrieved_anchors=key_anchors,
            required_anchors=key_anchors
        )
        scores["temporal_diversity"] = retrieval_metrics["temporal_diversity"].compute(
            retrieved_docs=retrieved_docs, k=5
        )
        
        if retrieved_docs:
            scores["temporal_precision@3"] = temporal_precision.compute(
                query=query,
                retrieved_docs=retrieved_docs[:3],
                temporal_focus=temporal_focus,
                k=3
            )
            
            scores["temporal_relevance@5"] = temporal_relevance.compute(
                query=query,
                retrieved_docs=retrieved_docs[:5],
                temporal_focus=temporal_focus,
                k=5
            )
            
            # NDCGFullCoverage for cross-period queries
            if is_cross_period and len(key_anchors) >= 2:
                scores["ndcg_full_coverage@5"] = ndcg_full_coverage.compute(
                    query=query,
                    retrieved_ids=retrieved_ids[:5],
                    gold_ids=gold_ids,
                    retrieved_docs=retrieved_docs[:5],
                    baseline_anchor=key_anchors[0] if key_anchors else "",
                    comparison_anchor=key_anchors[1] if len(key_anchors) > 1 else "present",
                    k=5
                )
        
        # --- Generation Metrics ---
        if retrieved_docs and answer and not answer.startswith("Error"):
            try:
                scores["temporal_faithfulness"] = gen_metrics["temporal_faithfulness"].compute(
                    answer=answer, contexts=retrieved_docs[:5]
                )
            except:
                pass
            
            try:
                scores["temporal_hallucination"] = gen_metrics["temporal_hallucination"].compute(
                    answer=answer, contexts=retrieved_docs[:5]
                )
            except:
                pass
            
            try:
                scores["temporal_coherence"] = gen_metrics["temporal_coherence"].compute(
                    answer=answer
                )
            except:
                pass
            
            try:
                scores["answer_alignment"] = gen_metrics["answer_alignment"].compute(
                    query=query, answer=answer,
                    temporal_focus=temporal_focus,
                    target_anchors=key_anchors
                )
            except:
                pass
        
        # --- Reasoning Metrics ---
        try:
            scores["event_ordering"] = reasoning_metrics["event_ordering"].compute(
                answer=answer, gold_order=None
            )
        except:
            pass
        
        all_results.append(scores)
    
    print(f"  ✓ Evaluated all metrics")
    
    # ================================================================
    # Step 7: Report Results
    # ================================================================
    print("\n[7/7] Computing final results...")
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS - TimeQA with Generated Guidance")
    print("=" * 70)
    
    # Print per-query results
    for result in all_results:
        print(f"\nQuery: {result['query']}...")
        for key, value in result.items():
            if key not in ["query_id", "query"] and value is not None:
                import math
                if isinstance(value, float) and not math.isnan(value):
                    print(f"  {key}: {value:.3f}")
    
    # Aggregate results
    print("\n" + "-" * 50)
    print("AGGREGATED RESULTS")
    print("-" * 50)
    
    import math
    metric_names = [k for k in all_results[0].keys() if k not in ["query_id", "query"]]
    
    for metric in metric_names:
        values = [r.get(metric) for r in all_results if r.get(metric) is not None]
        valid_values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        if valid_values:
            avg = sum(valid_values) / len(valid_values)
            print(f"{metric}: {avg:.3f} (n={len(valid_values)})")
    
    # Compute TempoScore
    print("\n" + "=" * 50)
    print("TEMPOSCORE")
    print("=" * 50)
    
    tempo_scorer = TempoScore.create_balanced()
    
    temposcore_values = []
    for result in all_results:
        ts = tempo_scorer.compute(
            temporal_recall=result.get("temporal_recall@5", 0),
            temporal_precision=result.get("temporal_precision@3", 0),
            temporal_faithfulness=result.get("temporal_faithfulness", 0),
            temporal_coherence=result.get("temporal_coherence", 0),
        )
        if not math.isnan(ts['tempo_weighted']):
            temposcore_values.append(ts['tempo_weighted'])
    
    if temposcore_values:
        avg_temposcore = sum(temposcore_values) / len(temposcore_values)
        print(f"\nAverage TempoScore: {avg_temposcore:.3f}")
    
    print("\n✓ Evaluation complete!")
    
    # Save results
    output = {
        "results": all_results,
        "generated_guidance": {qid: guidance_map[qid] for qid in query_ids},
    }
    output_path = "timeqa_with_generated_guidance.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
