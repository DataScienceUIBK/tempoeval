"""ComplexTempQA dataset loader.

Loads from HuggingFace: DataScienceUIBK/ComplexTempQA
"""

from __future__ import annotations

import logging
from typing import List, Optional

from tempoeval.datasets.base import TemporalDataset, TemporalQuery

logger = logging.getLogger(__name__)


def load_complex_tempqa(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    question_type: Optional[str] = None,
) -> TemporalDataset:
    """
    Load ComplexTempQA dataset from Hugging Face.
    
    ComplexTempQA is one of the largest temporal QA datasets with
    over 100 million question-answer pairs. It categorizes questions 
    into Attribute, Comparison, and Counting types.
    
    HuggingFace: DataScienceUIBK/ComplexTempQA
    Paper: "ComplexTempQA: A Large-Scale Dataset for Complex Temporal Question Answering"
    
    Args:
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum samples to load (recommended due to dataset size)
        cache_dir: Cache directory for HuggingFace
        question_type: Filter by type ("attribute", "comparison", "counting")
        
    Returns:
        TemporalDataset with queries
    
    Example:
        >>> from tempoeval.datasets import load_complex_tempqa
        >>> dataset = load_complex_tempqa(split="train", max_samples=1000)
        >>> print(len(dataset))
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install 'datasets': pip install datasets")
    
    logger.info(f"Loading ComplexTempQA: split={split}")
    
    # Load from HuggingFace
    ds = load_dataset(
        "DataScienceUIBK/ComplexTempQA",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    if split not in ds:
        available = list(ds.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available}")
    
    data = ds[split]
    
    # Filter by question type if specified
    if question_type:
        data = data.filter(
            lambda x: x.get("question_type", "").lower() == question_type.lower()
        )
    
    queries: List[TemporalQuery] = []
    
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        q_type = item.get("question_type", "")
        
        # Map question type to temporal focus
        temporal_focus = _map_question_type(q_type, question)
        
        query = TemporalQuery(
            id=str(item.get("id", i)),
            query=question,
            gold_answer=str(answer) if answer else "",
            temporal_focus=temporal_focus,
            metadata={
                "question_type": q_type,
                "source": "ComplexTempQA",
            },
        )
        queries.append(query)
    
    logger.info(f"Loaded {len(queries)} queries from ComplexTempQA/{split}")
    
    return TemporalDataset(
        queries=queries,
        name=f"complex_tempqa_{split}",
        metadata={"source": "DataScienceUIBK/ComplexTempQA", "split": split},
    )


def _map_question_type(q_type: str, question: str) -> str:
    """Map ComplexTempQA question type to temporal focus."""
    q_type_lower = q_type.lower() if q_type else ""
    q_lower = question.lower()
    
    if "counting" in q_type_lower or "how many" in q_lower:
        return "counting"
    elif "comparison" in q_type_lower:
        return "comparison"
    elif "attribute" in q_type_lower:
        return "specific_time"
    elif any(w in q_lower for w in ["when", "what year", "what date"]):
        return "specific_time"
    elif any(w in q_lower for w in ["how long", "duration"]):
        return "duration"
    elif any(w in q_lower for w in ["before", "after", "first", "last"]):
        return "temporal_ordering"
    else:
        return "specific_time"
