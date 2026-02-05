"""TEMPO benchmark dataset loader.

Updated for HuggingFace datasets v4.5+ which deprecated trust_remote_code.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from tempoeval.datasets.base import TemporalDataset, TemporalQuery

logger = logging.getLogger(__name__)

# Default TEMPO domains
TEMPO_DOMAINS = [
    "bitcoin",
    "cardano",
    "economics",
    "genealogy",
    "history",
    "hsm",
    "iota",
    "law",
    "monero",
    "politics",
    "quant",
    "travel",
    "workplace",
]


def load_tempo(
    domain: str = "bitcoin",
    cache_dir: Optional[str] = None,
    include_guidance: bool = True,
    max_samples: Optional[int] = None,
) -> TemporalDataset:
    """
    Load TEMPO benchmark dataset from Hugging Face.
    
    TEMPO is a temporal QA benchmark with 13 domains and rich temporal annotations.
    
    HuggingFace: tempo26/Tempo
    
    Args:
        domain: Domain/split to load (e.g., "bitcoin", "hsm", "economics")
                Available: bitcoin, cardano, economics, genealogy, history,
                          hsm, iota, law, monero, politics, quant, travel, workplace
        cache_dir: Cache directory for HuggingFace
        include_guidance: Whether to include query guidance metadata
        max_samples: Maximum number of samples to load
        
    Returns:
        TemporalDataset with queries
    
    Example:
        >>> from tempoeval.datasets import load_tempo
        >>> dataset = load_tempo(domain="hsm")
        >>> print(len(dataset))  # 150 queries
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install 'datasets': pip install datasets")
    
    if domain not in TEMPO_DOMAINS:
        logger.warning(f"Domain '{domain}' not in known list: {TEMPO_DOMAINS}")
    
    logger.info(f"Loading TEMPO dataset: domain={domain}")
    
    # Load from "steps" using hf_hub_download and pyarrow (more robust than datasets.load_dataset)
    try:
        steps_path = hf_hub_download(
            repo_id="tempo26/Tempo",
            filename=f"steps/{domain}.parquet",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        table = pq.read_table(steps_path)
        ds = table.to_pylist()
        logger.info(f"Loaded {len(ds)} items from steps config")
    except Exception as e:
        logger.warning(f"Failed to load 'steps', trying 'examples': {e}")
        # Fallback to examples config
        examples_path = hf_hub_download(
            repo_id="tempo26/Tempo",
            filename=f"examples/{domain}.parquet",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        table = pq.read_table(examples_path)
        ds = table.to_pylist()
        logger.info(f"Loaded {len(ds)} items from examples config")
    
    # Convert to TemporalQuery objects
    queries: List[TemporalQuery] = []
    
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        
        qid = str(item["id"])
        query_text = item.get("query", "")
        gold_ids = item.get("gold_ids", [])
        gold_answers = item.get("gold_answers", [])
        
        # Extract guidance if available
        guidance = item.get("query_guidance", {}) or {}
        
        # Parse guidance fields
        temporal_focus = guidance.get("temporal_reasoning_class_primary", "specific_time")
        key_anchors = guidance.get("key_time_anchors", []) or []
        
        # Determine if cross-period
        is_cross = _is_cross_period_query(temporal_focus, query_text)
        baseline, comparison = "", ""
        if is_cross and len(key_anchors) >= 2:
            baseline, comparison = str(key_anchors[0]), str(key_anchors[-1])
        
        query = TemporalQuery(
            id=qid,
            query=query_text,
            gold_ids=gold_ids,
            gold_answer=gold_answers[0] if gold_answers else None,
            temporal_focus=temporal_focus,
            key_time_anchors=[str(a) for a in key_anchors],
            baseline_anchor=baseline,
            comparison_anchor=comparison,
            is_cross_period=is_cross,
            metadata={
                "domain": domain,
                "gold_answers": gold_answers,
                "query_guidance": guidance,
                "gold_passage_annotations": item.get("gold_passage_annotations", []),
                "source": "TEMPO",
            },
        )
        queries.append(query)
    
    logger.info(f"Loaded {len(queries)} queries from TEMPO/{domain}")
    
    return TemporalDataset(
        queries=queries,
        name=f"tempo_{domain}",
        metadata={"domain": domain, "source": "tempo26/Tempo", "split": domain},
    )


def _is_cross_period_query(temporal_focus: str, query_text: str) -> bool:
    """Detect if query requires cross-period reasoning."""
    if temporal_focus == "trends_changes_and_cross_period":
        return True
    
    ql = query_text.lower()
    return any(kw in ql for kw in ["since", "over the last", "changed", "trend", "evolution", "compare"])


def load_tempo_documents(
    domain: str = "bitcoin",
    cache_dir: Optional[str] = None,
    max_docs: Optional[int] = None,
) -> Dict[str, str]:
    """
    Load TEMPO corpus documents.
    
    Args:
        domain: Domain to load
        cache_dir: Cache directory
        max_docs: Maximum documents to load
        
    Returns:
        Dict mapping doc_id -> doc_text
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install 'datasets': pip install datasets")
    
    logger.info(f"Loading TEMPO documents: domain={domain}")
    
    try:
        docs_path = hf_hub_download(
            repo_id="tempo26/Tempo",
            filename=f"documents/{domain}.parquet",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        table = pq.read_table(docs_path)
        docs_ds = table.to_pylist()
    except Exception as e:
        logger.error(f"Failed to load documents for domain {domain}: {e}")
        raise e
    
    result = {}
    for i, d in enumerate(docs_ds):
        if max_docs and i >= max_docs:
            break
        result[str(d["id"])] = d.get("content", "")
    
    logger.info(f"Loaded {len(result)} documents")
    return result


def list_tempo_domains() -> List[str]:
    """List available TEMPO domains."""
    return TEMPO_DOMAINS.copy()
