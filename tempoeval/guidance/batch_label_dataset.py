#!/usr/bin/env python3
"""
Batch Temporal Labeling CLI
============================

Generate temporal annotations for an entire dataset in batch mode.
Supports parallel processing and checkpointing for large datasets.

Usage:
    python batch_label_dataset.py \\
        --dataset-root /path/to/my_ir_dataset \\
        --out output_annotations.jsonl \\
        --concurrency 20

Dataset Structure Expected:
    my_ir_dataset/
    ├── corpus/           # .txt files organized by domain
    │   └── domain/
    │       └── doc_id.txt
    └── queries/          # JSONL files with queries
        └── domain_queries.jsonl

Query JSONL Format:
    {"id": "q1", "query": "...", "gold_ids": ["domain/doc1.txt"], "negative_ids": [...]}
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tempoeval.llm import AzureOpenAIProvider
from tempoeval.guidance import TemporalGuidanceGenerator


def iter_jsonl(path: str):
    """Iterate over JSONL file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def load_processed_ids(*paths: str) -> set:
    """Load already processed query IDs from output files."""
    processed = set()
    for p in paths:
        if not p or not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        for rec in iter_jsonl(p):
            qid = rec.get("id") or rec.get("qid")
            if qid is not None:
                processed.add(str(qid))
    return processed


def append_jsonl(path: str, record: Dict[str, Any]):
    """Append a record to JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def safe_corpus_path(corpus_root: str, doc_id: str) -> str:
    """Get safe path for document, preventing traversal."""
    doc_id = doc_id.lstrip("/\\")
    full = os.path.normpath(os.path.join(corpus_root, doc_id))
    corpus_root_norm = os.path.normpath(corpus_root)
    if not full.startswith(corpus_root_norm + os.sep) and full != corpus_root_norm:
        raise ValueError(f"Unsafe doc_id path traversal: {doc_id}")
    return full


def read_passage_text(corpus_root: str, doc_id: str, max_chars: Optional[int] = None) -> Tuple[str, Optional[str]]:
    """Read passage text from corpus. Returns (text, error)."""
    try:
        path = safe_corpus_path(corpus_root, doc_id)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if max_chars and max_chars > 0:
            text = text[:max_chars]
        return text, None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch temporal labeling for IR datasets"
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Path to dataset (with corpus/ and queries/ subdirs)"
    )
    parser.add_argument(
        "--domain", default=None,
        help="Process only queries/<domain>_queries.jsonl"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSONL path"
    )
    parser.add_argument(
        "--checkpoint-file", default=None,
        help="Separate checkpoint file (default: same as --out)"
    )
    parser.add_argument(
        "--max-gold", type=int, default=None,
        help="Max gold passages per query"
    )
    parser.add_argument(
        "--max-neg", type=int, default=None,
        help="Max negative passages per query"
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Parallel passage annotation requests"
    )
    parser.add_argument(
        "--max-passage-chars", type=int, default=6000,
        help="Truncate passages to this length"
    )
    parser.add_argument(
        "--reprocess-existing", action="store_true",
        help="Force reprocess already-processed queries"
    )
    args = parser.parse_args()

    # Initialize LLM
    print("[1] Initializing Azure OpenAI...")
    try:
        llm = AzureOpenAIProvider()
    except Exception as e:
        sys.exit(f"Failed to initialize LLM: {e}")
    print("  ✓ Connected")

    # Initialize generator
    generator = TemporalGuidanceGenerator(
        llm=llm,
        max_retries=2,
        concurrency=args.concurrency,
        max_passage_chars=args.max_passage_chars,
    )

    # Verify dataset structure
    dataset_root = args.dataset_root
    corpus_root = os.path.join(dataset_root, "corpus")
    queries_root = os.path.join(dataset_root, "queries")

    if not os.path.isdir(corpus_root):
        sys.exit(f"Missing corpus directory: {corpus_root}")
    if not os.path.isdir(queries_root):
        sys.exit(f"Missing queries directory: {queries_root}")

    # Find query files
    if args.domain:
        query_files = [os.path.join(queries_root, f"{args.domain}_queries.jsonl")]
        if not os.path.exists(query_files[0]):
            sys.exit(f"Query file not found: {query_files[0]}")
    else:
        query_files = sorted(
            os.path.join(queries_root, fn)
            for fn in os.listdir(queries_root)
            if fn.endswith("_queries.jsonl")
        )
        if not query_files:
            sys.exit(f"No *_queries.jsonl files in {queries_root}")

    print(f"[2] Found {len(query_files)} query file(s)")

    # Load checkpoint
    ckpt_path = args.checkpoint_file or args.out
    processed = set()
    if not args.reprocess_existing:
        processed = load_processed_ids(args.out, ckpt_path)
        if processed:
            print(f"  ✓ Resuming from checkpoint ({len(processed)} already done)")

    total_written = 0

    for qf in query_files:
        print(f"\n[Processing] {os.path.basename(qf)}")
        
        for qi, q in enumerate(iter_jsonl(qf), 1):
            qid = str(q.get("id") or q.get("qid") or "")
            if not qid:
                continue

            if (not args.reprocess_existing) and qid in processed:
                if qi % 100 == 0:
                    print(f"  ... skipped {qi} (already processed)")
                continue

            query_text = q.get("query", "")
            gold_ids = list(q.get("gold_ids", []) or [])
            neg_ids = list(q.get("negative_ids", []) or [])

            if args.max_gold is not None:
                gold_ids = gold_ids[:args.max_gold]
            if args.max_neg is not None:
                neg_ids = neg_ids[:args.max_neg]

            # Generate query guidance
            guidance = generator.generate_query_guidance(query_text)

            # Load and annotate gold passages
            gold_passages = []
            for doc_id in gold_ids:
                text, err = read_passage_text(corpus_root, doc_id, args.max_passage_chars)
                if not err and text.strip():
                    gold_passages.append((doc_id, text))

            gold_annotations = []
            if gold_passages:
                texts = [p[1] for p in gold_passages]
                ids = [p[0] for p in gold_passages]
                gold_annotations = [
                    a.to_dict()
                    for a in generator.annotate_passages(query_text, texts, ids)
                ]

            # Load and annotate negative passages
            neg_passages = []
            for doc_id in neg_ids:
                text, err = read_passage_text(corpus_root, doc_id, args.max_passage_chars)
                if not err and text.strip():
                    neg_passages.append((doc_id, text))

            neg_annotations = []
            if neg_passages:
                texts = [p[1] for p in neg_passages]
                ids = [p[0] for p in neg_passages]
                neg_annotations = [
                    a.to_dict()
                    for a in generator.annotate_passages(query_text, texts, ids)
                ]

            # Build output record (preserve original fields)
            out_rec = dict(q)
            out_rec["query_guidance"] = guidance.to_dict()
            out_rec["gold_passage_annotations"] = gold_annotations
            out_rec["negative_passage_annotations"] = neg_annotations

            append_jsonl(ckpt_path, out_rec)
            total_written += 1
            processed.add(qid)

            if total_written % 50 == 0:
                print(f"  ✓ Wrote {total_written} records")

    print(f"\n{'=' * 50}")
    print(f"✓ Done! Wrote {total_written} records to {ckpt_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
