"""TimeQA dataset loader.

Loads from HuggingFace: hugosousa/TimeQA
"""

from __future__ import annotations

import logging
from typing import List, Optional

from tempoeval.datasets.base import TemporalDataset, TemporalQuery

logger = logging.getLogger(__name__)


def load_timeqa(
    split: str = "test",
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> TemporalDataset:
    """
    Load TimeQA dataset from Hugging Face.
    
    TimeQA is a time-sensitive question answering dataset created from
    Wikipedia with questions requiring temporal reasoning.
    
    HuggingFace: hugosousa/TimeQA
    Paper: "A Dataset for Answering Time-Sensitive Questions"
    
    Args:
        split: Dataset split ("train", "validation", "test")
        subset: Optional subset filter ("easy" or "hard")
        cache_dir: Cache directory for HuggingFace
        max_samples: Maximum number of samples to load
        
    Returns:
        TemporalDataset with queries
    
    Example:
        >>> from tempoeval.datasets import load_timeqa
        >>> dataset = load_timeqa(split="test")
        >>> print(len(dataset))
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install 'datasets': pip install datasets")
    
    logger.info(f"Loading TimeQA dataset: split={split}, subset={subset}")
    
    # Load from HuggingFace
    ds = load_dataset(
        "hugosousa/TimeQA",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Get the appropriate split
    if split not in ds:
        available = list(ds.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available}")
    
    data = ds[split]
    
    # Filter by subset if specified
    if subset:
        data = data.filter(lambda x: x.get("type", "") == subset)
    
    queries: List[TemporalQuery] = []
    
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
        
        # Extract fields
        question = item.get("question", "")
        answer = item.get("answer", item.get("answers", ""))
        context = item.get("context", item.get("passage", ""))
        
        # Handle list answers
        if isinstance(answer, list):
            answer_str = answer[0] if answer else ""
        else:
            answer_str = str(answer) if answer else ""
        
        # Infer temporal focus
        temporal_focus = _infer_temporal_focus(question)
        
        query = TemporalQuery(
            id=str(item.get("id", i)),
            query=question,
            gold_answer=answer_str,
            temporal_focus=temporal_focus,
            metadata={
                "context": context,
                "type": item.get("type", ""),
                "source": "TimeQA",
            },
        )
        queries.append(query)
    
    logger.info(f"Loaded {len(queries)} queries from TimeQA/{split}")
    
    return TemporalDataset(
        queries=queries,
        name=f"timeqa_{split}" + (f"_{subset}" if subset else ""),
        metadata={"source": "hugosousa/TimeQA", "split": split, "subset": subset},
    )


def _infer_temporal_focus(question: str) -> str:
    """Infer temporal focus from question text."""
    q_lower = question.lower()
    
    if any(w in q_lower for w in ["when did", "when was", "what year", "what date"]):
        return "specific_time"
    elif any(w in q_lower for w in ["how long", "duration", "lasted"]):
        return "duration"
    elif any(w in q_lower for w in ["before", "after", "first", "last", "prior"]):
        return "temporal_ordering"
    elif any(w in q_lower for w in ["changed", "since", "trend", "over time"]):
        return "change_over_time"
    else:
        return "specific_time"
