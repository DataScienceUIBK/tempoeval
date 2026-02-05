"""
Complete TEMPO HSM Evaluation Pipeline with Cost Tracking + THREADING
======================================================================

This example demonstrates a full evaluation pipeline with 20-thread concurrency:
1. Load TEMPO dataset (HSM - History of Science and Medicine domain)
2. Load corpus documents
3. Retrieve with BM25
4. Generate answers with Azure GPT-4 (20 CONCURRENT THREADS)
5. Evaluate ALL 16 temporal metrics (20 CONCURRENT THREADS)
6. Compute final TempoScore
7. Track efficiency (latency & cost) for all LLM calls

Requirements:
    pip install datasets pyserini gensim openai pyarrow huggingface_hub python-dotenv

Setup:
    1. Copy examples/.env.example to examples/.env
    2. Fill in your Azure OpenAI credentials
    3. Run this script
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Verify environment is configured
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    print("ERROR: Azure OpenAI credentials not found!")
    print("Please copy examples/.env.example to examples/.env and fill in your credentials.")
    exit(1)

# Number of concurrent threads
MAX_THREADS = 20

from tqdm import tqdm

# TempoEval imports
from tempoeval.llm import AzureOpenAIProvider
from tempoeval.metrics import (
    # Retrieval metrics (Layer 1)
    TemporalRecall,
    TemporalPrecision,
    TemporalNDCG,
    TemporalMRR,
    TemporalCoverage,
    AnchorCoverage,
    TemporalDiversity,
    TemporalRelevance,
    NDCGFullCoverage,
    # Generation metrics (Layer 2)
    TemporalFaithfulness,
    TemporalHallucination,
    TemporalCoherence,
    AnswerTemporalAlignment,
    # Reasoning metrics (Layer 3)
    EventOrdering,
    DurationAccuracy,
    CrossPeriodReasoning,
    # Composite
    TempoScore,
)
from tempoeval.core.result import EvaluationResult, QueryResult
from tempoeval.efficiency.tracker import EfficiencyTracker


# ================================================================
# Helper function for numpy array safety  
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


def to_dict(val, default=None):
    """Convert value to dict, handling None and numpy."""
    if val is None:
        return default if default is not None else {}
    if isinstance(val, dict):
        return val
    if hasattr(val, 'items'):
        return dict(val)
    return default if default is not None else {}


# ================================================================
# BM25 Retrieval Function
# ================================================================
def retrieval_bm25(queries, query_ids, documents, doc_ids, k=10):
    """BM25 retrieval using Gensim (Pyserini optional)."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity
        from gensim.utils import simple_preprocess
        
        # Try to use Pyserini analyzer if available (better), otherwise Gensim
        try:
            from pyserini import analysis
            analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
            check_analyzer = lambda x: analyzer.analyze(x)
            print("  ✓ Using Pyserini analyzer")
        except ImportError:
            check_analyzer = lambda x: simple_preprocess(x)
            print("  ✓ Using Gensim simple_preprocess (Pyserini not found)")

        # Prepare corpus
        print("  Index building...")
        corpus = [check_analyzer(x) for x in documents]
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
        for query_id, query_text in tqdm(zip(query_ids, queries), desc="BM25 Retrieval", total=len(queries)):
            query_tokens = check_analyzer(query_text)
            bm25_query = model[dictionary.doc2bow(query_tokens)]
            similarities = bm25_index[bm25_query].tolist()
            
            scored = list(zip(doc_ids, similarities))
            scored.sort(key=lambda x: x[1], reverse=True)
            all_results[str(query_id)] = [doc_id for doc_id, _ in scored[:k]]
        
        return all_results
        
    except ImportError as e:
        print(f"Warning: gensim not installed ({e}), using simple keyword matching")
        return simple_keyword_retrieval(queries, query_ids, documents, doc_ids, k)


def simple_keyword_retrieval(queries, query_ids, documents, doc_ids, k=10):
    """Fallback simple keyword-based retrieval."""
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


# ================================================================
# Answer Generation Function (Thread-Safe)
# ================================================================
def generate_answer(query: str, contexts: list, llm) -> str:
    """Generate answer using Azure GPT-4."""
    context_text = "\n\n".join([f"[Document {i+1}]:\n{c[:1500]}" for i, c in enumerate(contexts[:5])])
    
    prompt = f"""Based on the following documents, answer the question about history of science and medicine.
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
# Threaded Answer Generation
# ================================================================
def generate_answers_threaded(query_ids, queries, retrieval_results, doc_map, llm, tracker, max_threads=MAX_THREADS):
    """Generate answers with concurrent threading."""
    generated_answers = {}
    lock = Lock()
    empty_context_count = 0
    
    def process_query(qid, query):
        nonlocal empty_context_count
        retrieved_doc_ids = retrieval_results.get(qid, [])[:5]
        contexts = [doc_map.get(did, "") for did in retrieved_doc_ids if did in doc_map]
        
        if not contexts or all(c == "" for c in contexts):
            with lock:
                empty_context_count += 1
            return qid, "No relevant documents found."
        else:
            tracker.start_tracking(f"gen_{qid}", "answer_generation")
            answer = generate_answer(query, contexts, llm)
            prompt_chars = sum(len(c) for c in contexts[:5]) + len(query) + 200
            tracker.stop_tracking(f"gen_{qid}", prompt_chars=prompt_chars, completion_chars=len(answer))
            return qid, answer
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_query, qid, query): qid for qid, query in zip(query_ids, queries)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating ({max_threads} threads)"):
            qid, answer = future.result()
            generated_answers[qid] = answer
    
    return generated_answers, empty_context_count


# ================================================================
# Threaded Evaluation
# ================================================================
def evaluate_query_threaded(qid, query, retrieval_results, doc_map, gold_ids_map, guidance_map, 
                            generated_answers, retrieval_metrics, gen_metrics, reasoning_metrics,
                            temporal_precision, temporal_coverage):
    """Evaluate a single query (thread-safe)."""
    retrieved_ids = retrieval_results.get(qid, [])
    retrieved_docs = [doc_map.get(did, "") for did in retrieved_ids if did in doc_map]
    gold_ids = gold_ids_map.get(qid, [])
    answer = generated_answers.get(qid, "")
    guidance = guidance_map.get(qid, {})
    
    temporal_focus = guidance.get("temporal_reasoning_class_primary") or "specific_time"
    key_anchors = to_list(guidance.get("key_time_anchors"), [])
    is_cross_period = "cross_period" in temporal_focus.lower() or "trend" in temporal_focus.lower()
    
    scores = {"query_id": qid, "query": query[:50]}
    
    # --- Retrieval Metrics (Layer 1) ---
    # GOLD-BASED MODE (traditional IR)
    scores["temporal_recall@10_gold"] = retrieval_metrics["temporal_recall"].compute(
        retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=10
    )
    scores["temporal_ndcg@10_gold"] = retrieval_metrics["temporal_ndcg"].compute(
        retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=10
    )
    scores["temporal_mrr@10_gold"] = retrieval_metrics["temporal_mrr"].compute(
        retrieved_ids=retrieved_ids, gold_ids=gold_ids, k=10
    )
    
    # LLM-AS-JUDGE MODE (no gold IDs needed)
    if retrieved_docs:
        try:
            scores["temporal_recall@10_llm"] = retrieval_metrics["temporal_recall_llm"].compute(
                query=query, retrieved_docs=retrieved_docs[:10], k=10
            )
        except:
            scores["temporal_recall@10_llm"] = 0.0
        
        try:
            scores["temporal_ndcg@10_llm"] = retrieval_metrics["temporal_ndcg_llm"].compute(
                query=query, retrieved_docs=retrieved_docs[:10], k=10
            )
        except:
            scores["temporal_ndcg@10_llm"] = 0.0
        
        try:
            scores["temporal_mrr@10_llm"] = retrieval_metrics["temporal_mrr_llm"].compute(
                query=query, retrieved_docs=retrieved_docs[:10], k=10
            )
        except:
            scores["temporal_mrr@10_llm"] = 0.0
    
    # Other retrieval metrics
    scores["anchor_coverage"] = retrieval_metrics["anchor_coverage"].compute(
        retrieved_anchors=key_anchors,
        required_anchors=key_anchors
    )
    scores["temporal_diversity"] = retrieval_metrics["temporal_diversity"].compute(
        retrieved_docs=retrieved_docs, k=10
    )
    
    # LLM-based retrieval (Precision, Coverage)
    if retrieved_docs:
        try:
            scores["temporal_precision@10"] = temporal_precision.compute(
                query=query, retrieved_docs=retrieved_docs[:10], temporal_focus=temporal_focus, k=10
            )
        except:
            scores["temporal_precision@10"] = 0.0
        
        if is_cross_period and len(key_anchors) >= 2:
            try:
                scores["temporal_coverage@10"] = temporal_coverage.compute(
                    query=query, retrieved_docs=retrieved_docs[:10], temporal_focus=temporal_focus,
                    baseline_anchor=str(key_anchors[0]), comparison_anchor=str(key_anchors[-1]), k=10
                )
            except:
                scores["temporal_coverage@10"] = 0.0
    
    # --- Generation Metrics (Layer 2) ---
    if retrieved_docs and answer and answer != "No relevant documents found.":
        try:
            scores["temporal_faithfulness"] = gen_metrics["temporal_faithfulness"].compute(
                answer=answer, contexts=retrieved_docs[:5]
            )
        except:
            scores["temporal_faithfulness"] = 0.0
        
        try:
            scores["temporal_hallucination"] = gen_metrics["temporal_hallucination"].compute(
                answer=answer, contexts=retrieved_docs[:5]
            )
        except:
            scores["temporal_hallucination"] = 0.0
        
        try:
            scores["temporal_coherence"] = gen_metrics["temporal_coherence"].compute(answer=answer)
        except:
            scores["temporal_coherence"] = 0.0
        
        try:
            scores["answer_alignment"] = gen_metrics["answer_alignment"].compute(
                query=query, answer=answer, temporal_focus=temporal_focus, target_anchors=key_anchors
            )
        except:
            scores["answer_alignment"] = 0.0
    
    # --- Reasoning Metrics (Layer 3) ---
    try:
        scores["event_ordering"] = reasoning_metrics["event_ordering"].compute(answer=answer, gold_order=None)
    except:
        scores["event_ordering"] = 0.0
    
    if is_cross_period:
        try:
            scores["cross_period_reasoning"] = reasoning_metrics["cross_period"].compute(
                query=query, answer=answer,
                baseline_anchor=str(key_anchors[0]) if key_anchors else "",
                comparison_anchor=str(key_anchors[-1]) if key_anchors else "",
            )
        except:
            scores["cross_period_reasoning"] = 0.0
    
    return scores


def evaluate_all_threaded(query_ids, queries, retrieval_results, doc_map, gold_ids_map, guidance_map,
                          generated_answers, retrieval_metrics, gen_metrics, reasoning_metrics,
                          temporal_precision, temporal_coverage, max_threads=MAX_THREADS):
    """Evaluate all queries with concurrent threading."""
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(
                evaluate_query_threaded, qid, query, retrieval_results, doc_map, gold_ids_map,
                guidance_map, generated_answers, retrieval_metrics, gen_metrics, reasoning_metrics,
                temporal_precision, temporal_coverage
            ): qid 
            for qid, query in zip(query_ids, queries)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating ({max_threads} threads)"):
            result = future.result()
            all_results.append(result)
    
    return all_results


# ================================================================
# Main Evaluation Pipeline
# ================================================================
def main():
    print("=" * 70)
    print("TemporalEval - Complete TEMPO HSM Evaluation Pipeline (THREADED)")
    print(f"Using {MAX_THREADS} concurrent threads for generation and evaluation")
    print("=" * 70)
    
    # ================================================================
    # Step 1: Load TEMPO HSM Dataset
    # ================================================================
    print("\n[1/6] Loading TEMPO HSM dataset...")
    
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        
        domain = "hsm"
        
        print(f"    Downloading {domain} steps...")
        steps_path = hf_hub_download(
            repo_id="tempo26/Tempo",
            filename=f"steps/{domain}.parquet",
            repo_type="dataset"
        )
        
        print(f"    Downloading {domain} documents...")
        docs_path = hf_hub_download(
            repo_id="tempo26/Tempo",
            filename=f"documents/{domain}.parquet",
            repo_type="dataset"
        )
        
        steps_table = pq.read_table(steps_path)
        docs_table = pq.read_table(docs_path)
        
        steps_df = steps_table.to_pandas()
        docs_df = docs_table.to_pandas()
        
        print(f"  ✓ Loaded {len(steps_df)} queries from HSM domain")
        print(f"  ✓ Loaded {len(docs_df)} documents")
        
        queries = steps_df["query"].tolist()
        query_ids = [str(x) for x in steps_df["id"].tolist()]
        gold_ids_map = {str(row["id"]): to_list(row["gold_ids"], []) for _, row in steps_df.iterrows()}
        
        guidance_map = {}
        for _, row in steps_df.iterrows():
            qid = str(row["id"])
            raw_guidance = row.get("query_guidance")
            guidance = to_dict(raw_guidance, {})
            if "key_time_anchors" in guidance:
                guidance["key_time_anchors"] = to_list(guidance["key_time_anchors"], [])
            guidance_map[qid] = guidance
        
        documents = docs_df["content"].tolist()
        doc_ids = [str(x) for x in docs_df["id"].tolist()]
        doc_map = {str(row["id"]): row["content"] for _, row in docs_df.iterrows()}
        
    except Exception as e:
        print(f"  ⚠ Error loading TEMPO: {e}")
        print("  Using sample HSM data instead...")
        
        queries = [
            "When did Alexander Fleming discover penicillin?",
            "How has the germ theory of disease evolved since the 1860s?",
            "What were the key milestones in the development of the polio vaccine?",
        ]
        query_ids = ["q1", "q2", "q3"]
        gold_ids_map = {
            "q1": ["doc_1", "doc_2"],
            "q2": ["doc_3", "doc_4", "doc_5"],
            "q3": ["doc_6", "doc_7"],
        }
        
        documents = [
            "In 1928, Alexander Fleming discovered penicillin at St. Mary's Hospital.",
            "Fleming's discovery in 1928 led to mass production of penicillin by 1943.",
            "Louis Pasteur's germ theory, proposed in the 1860s, revolutionized medicine.",
            "Robert Koch further developed germ theory in the 1880s.",
            "By the 1900s, germ theory was universally accepted.",
            "Jonas Salk developed the first successful polio vaccine in 1955.",
            "Albert Sabin's oral polio vaccine, introduced in 1961, became dominant.",
        ]
        doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6", "doc_7"]
        doc_map = dict(zip(doc_ids, documents))
        guidance_map = {
            "q1": {"temporal_reasoning_class_primary": "specific_time", "key_time_anchors": ["1928"]},
            "q2": {"temporal_reasoning_class_primary": "trends_changes_and_cross_period", "key_time_anchors": ["1860s", "1880s", "1900s"]},
            "q3": {"temporal_reasoning_class_primary": "temporal_ordering", "key_time_anchors": ["1955", "1961"]},
        }
    
    # Limit for demo (FAST VERIFICATION)
    max_queries = -1
    queries = queries[:max_queries]
    query_ids = query_ids[:max_queries]
    
    # Limit documents for speed - REMOVED for full run
    # max_docs = 5000
    # if len(documents) > max_docs:
    #     print(f"  ⚠ Limiting to {max_docs} documents for speed")
    #     documents = documents[:max_docs]
    #     doc_ids = doc_ids[:max_docs]
    
    print(f"  Processing {len(queries)} queries...")
    
    # ================================================================
    # Step 2: BM25 Retrieval
    # ================================================================
    print("\n[2/6] Running BM25 retrieval...")
    
    k = 10
    retrieval_results = retrieval_bm25(queries, query_ids, documents, doc_ids, k=k)
    
    print(f"  ✓ Retrieved top-{k} documents for each query")
    
    # ================================================================
    # Step 3: Initialize LLM, Metrics, and Efficiency Tracker
    # ================================================================
    print("\n[3/7] Initializing Azure OpenAI, metrics, and efficiency tracker...")
    
    llm = AzureOpenAIProvider()
    print("  ✓ Azure OpenAI connected")
    
    tracker = EfficiencyTracker(model_name="gpt-4o")
    print("  ✓ Efficiency Tracker initialized")
    
    # Initialize all metrics
    # GOLD-BASED (traditional IR)
    retrieval_metrics = {
        "temporal_recall": TemporalRecall(use_llm=False),
        "temporal_ndcg": TemporalNDCG(use_llm=False),
        "temporal_mrr": TemporalMRR(use_llm=False),
        "anchor_coverage": AnchorCoverage(use_llm=True),
        "temporal_diversity": TemporalDiversity(use_llm=True),
        "temporal_relevance": TemporalRelevance(),
        "ndcg_full_coverage": NDCGFullCoverage(),
        # LLM-AS-JUDGE MODE (no gold IDs needed)
        "temporal_recall_llm": TemporalRecall(use_llm=True),
        "temporal_ndcg_llm": TemporalNDCG(use_llm=True),
        "temporal_mrr_llm": TemporalMRR(use_llm=True),
    }
    
    # Assign LLM to all metrics that need it
    temporal_precision = TemporalPrecision()
    temporal_precision.llm = llm
    temporal_coverage = TemporalCoverage()
    temporal_coverage.llm = llm
    
    # Assign LLM to dual-mode and LLM-based metrics
    retrieval_metrics["anchor_coverage"].llm = llm
    retrieval_metrics["temporal_diversity"].llm = llm
    retrieval_metrics["temporal_relevance"].llm = llm
    retrieval_metrics["ndcg_full_coverage"].llm = llm
    retrieval_metrics["temporal_recall_llm"].llm = llm
    retrieval_metrics["temporal_ndcg_llm"].llm = llm
    retrieval_metrics["temporal_mrr_llm"].llm = llm
    
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
        "duration_accuracy": DurationAccuracy(),
        "cross_period": CrossPeriodReasoning(),
    }
    for m in reasoning_metrics.values():
        m.llm = llm
    
    print("  ✓ All metrics initialized")
    
    # ================================================================
    # Step 4: Generate Answers (THREADED)
    # ================================================================
    print(f"\n[4/7] Generating answers with GPT-4 ({MAX_THREADS} threads)...")
    
    generated_answers, empty_context_count = generate_answers_threaded(
        query_ids, queries, retrieval_results, doc_map, llm, tracker, max_threads=MAX_THREADS
    )
    
    if empty_context_count > 0:
        print(f"\n  ⚠ Warning: {empty_context_count}/{len(queries)} queries had no matching contexts")
    
    print(f"  ✓ Generated {len(generated_answers)} answers")
    
    # ================================================================
    # Step 5: Evaluate All Metrics (THREADED)
    # ================================================================
    print(f"\n[5/7] Evaluating all metrics ({MAX_THREADS} threads)...")
    
    all_results = evaluate_all_threaded(
        query_ids, queries, retrieval_results, doc_map, gold_ids_map, guidance_map,
        generated_answers, retrieval_metrics, gen_metrics, reasoning_metrics,
        temporal_precision, temporal_coverage, max_threads=MAX_THREADS
    )
    
    print(f"  ✓ Evaluated all metrics")
    
    # ================================================================
    # Step 6: Aggregate and Report Results
    # ================================================================
    print("\n[6/7] Computing final results...")
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS - TEMPO HSM Domain")
    print("=" * 70)
    
    for result in all_results[:5]:  # Show first 5
        print(f"\nQuery: {result['query']}...")
        for key, value in result.items():
            if key not in ["query_id", "query"] and value is not None:
                import math
                if isinstance(value, (int, float)) and not math.isnan(value):
                    print(f"  {key}: {value:.3f}")
    
    # Aggregate results
    print("\n" + "-" * 50)
    print("AGGREGATED RESULTS")
    print("-" * 50)
    
    import math
    metric_names = [k for k in all_results[0].keys() if k not in ["query_id", "query"]]
    
    for metric in metric_names:
        valid_values = []
        for r in all_results:
            v = r.get(metric)
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v):
                valid_values.append(v)
        if valid_values:
            avg = sum(valid_values) / len(valid_values)
            print(f"{metric}: {avg:.3f} (n={len(valid_values)})")
    
    # Compute TempoScore
    print("\n" + "=" * 50)
    print("TEMPOSCORE")
    print("=" * 50)
    
    tempo_scorer = TempoScore.create_rag_balanced()
    
    temposcore_values = []
    for result in all_results:
        ts = tempo_scorer.compute(
            temporal_recall=result.get("temporal_recall@10_gold", 0),  # Use gold-based
            temporal_precision=result.get("temporal_precision@10", 0),
            temporal_faithfulness=result.get("temporal_faithfulness", 0),
            temporal_coverage=result.get("temporal_coverage@10", 0),
            cross_period_reasoning=result.get("cross_period_reasoning", 0),
        )
        if not math.isnan(ts):
            temposcore_values.append(ts)
    
    if temposcore_values:
        avg_temposcore = sum(temposcore_values) / len(temposcore_values)
        print(f"\nAverage TempoScore: {avg_temposcore:.3f}")
    
    # ================================================================
    # Step 7: Efficiency & Cost Report
    # ================================================================
    print("\n[7/7] Generating efficiency & cost report...")
    print("\n" + "=" * 70)
    print("EFFICIENCY & COST REPORT")
    print("=" * 70)
    
    eff_summary = tracker.summary()
    print(f"  Total LLM Operations:   {eff_summary['total_operations']}")
    print(f"  Total Latency:          {eff_summary['total_latency_seconds']:.2f} seconds")
    print(f"  Avg Latency per Op:     {eff_summary['avg_latency_ms']:.1f} ms")
    print(f"  Estimated Total Cost:   ${eff_summary['total_cost_usd']:.4f} USD")
    
    if eff_summary.get("metrics_breakdown"):
        print("\n  Per-Metric Breakdown:")
        for metric_name, stats in eff_summary["metrics_breakdown"].items():
            print(f"    {metric_name}: {stats['count']} calls, {stats['avg_latency_ms']:.1f}ms avg, ${stats['total_cost_usd']:.4f}")
    
    print("\n✓ Evaluation complete!")
    
    # Calculate aggregated scores properly - 0 is a valid score!
    aggregated_scores = {}
    for metric in metric_names:
        valid_values = []
        for r in all_results:
            v = r.get(metric)
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v):
                valid_values.append(v)
        if valid_values:
            aggregated_scores[metric] = sum(valid_values) / len(valid_values)
    
    # Save results
    output_path = "tempo_hsm_results_threaded.json"
    final_output = {
        "results": all_results,
        "aggregated": aggregated_scores,
        "tempo_score": avg_temposcore if temposcore_values else None,
        "efficiency": eff_summary,
        "config": {"max_threads": MAX_THREADS}
    }
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
