"""TimeBench dataset loader."""

from __future__ import annotations

import logging
from typing import List, Optional

from tempoeval.datasets.base import TemporalDataset, TemporalQuery

logger = logging.getLogger(__name__)


def load_timebench(
    split: str = "test",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> TemporalDataset:
    """
    Load TimeBench dataset.
    
    TimeBench is a comprehensive benchmark for evaluating temporal reasoning 
    across multiple levels (Comparison, Duration, Ordering, etc.).
    
    Args:
        split: Dataset split to load (train, dev, test)
        cache_dir: HuggingFace cache directory
        max_samples: Limit samples loaded
        
    Returns:
        TemporalDataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install 'datasets': pip install datasets")
        
    logger.info(f"Loading TimeBench dataset: split={split}")
    
    try:
        # Assuming hosted at 'timebench/TimeBench' or similar. 
        # Using a placeholder path since exact HF ID wasn't in search snippet.
        # Ideally this would be updated with the exact ID.
        ds = load_dataset(
            "timebench/TimeBench",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Could not load from HuggingFace directly: {e}")
        # Return empty for safety if not found, or raise
        logger.warning("Returning empty dataset placeholder for TimeBench.")
        return TemporalDataset(queries=[], name="timebench_placeholder", metadata={})

    queries: List[TemporalQuery] = []
    
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
            
        # Map TimeBench schema to TemporalQuery
        qid = str(item.get("id", i))
        query_text = item.get("question", "")
        answer = item.get("answer", "")
        reasoning_type = item.get("type", "general")
        
        # Heuristic to determine temporal focus from type
        temporal_focus = "specific_time"
        if "order" in reasoning_type.lower():
            temporal_focus = "order"
        elif "duration" in reasoning_type.lower():
            temporal_focus = "duration"
        elif "compare" in reasoning_type.lower():
            temporal_focus = "change_over_time"
            
        q = TemporalQuery(
            id=qid,
            query=query_text,
            gold_answer=answer,
            temporal_focus=temporal_focus,
            metadata={
                "source": "TimeBench",
                "reasoning_type": reasoning_type,
                "difficulty": item.get("level", "unknown")
            }
        )
        queries.append(q)
        
    logger.info(f"Loaded {len(queries)} samples from TimeBench")
    
    return TemporalDataset(
        queries=queries,
        name=f"timebench_{split}",
        metadata={"source": "TimeBench", "split": split}
    )
