"""SituatedQA dataset loader.

Loads from GitHub: https://github.com/mikejqzhang/SituatedQA
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

from tempoeval.datasets.base import TemporalDataset, TemporalQuery

logger = logging.getLogger(__name__)

# GitHub raw URLs for SituatedQA
SITUATEDQA_BASE_URL = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data"

SITUATEDQA_FILES = {
    # qa_data contains questions with answers and temporal context
    "temp.train": f"{SITUATEDQA_BASE_URL}/qa_data/temp.train.jsonl",
    "temp.dev": f"{SITUATEDQA_BASE_URL}/qa_data/temp.dev.jsonl",
    "temp.test": f"{SITUATEDQA_BASE_URL}/qa_data/temp.test.jsonl",
    "geo.train": f"{SITUATEDQA_BASE_URL}/qa_data/geo.train.jsonl",
    "geo.dev": f"{SITUATEDQA_BASE_URL}/qa_data/geo.dev.jsonl",
    "geo.test": f"{SITUATEDQA_BASE_URL}/qa_data/geo.test.jsonl",
}


def load_situatedqa(
    split: str = "test",
    context_type: str = "temporal",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> TemporalDataset:
    """
    Load SituatedQA dataset from GitHub.
    
    SituatedQA is a context-dependent QA dataset with temporal
    and geographical context requirements.
    
    GitHub: https://github.com/mikejqzhang/SituatedQA
    Paper: "Situational-QA: Context-Dependent Question Answering"
    
    Args:
        split: Dataset split ("train", "dev", "test")
        context_type: "temporal" or "geo" (we focus on temporal)
        cache_dir: Directory to cache downloaded files
        max_samples: Maximum number of samples to load
        
    Returns:
        TemporalDataset with queries
    
    Example:
        >>> from tempoeval.datasets import load_situatedqa
        >>> dataset = load_situatedqa(split="test", context_type="temporal")
        >>> print(len(dataset))
    """
    logger.info(f"Loading SituatedQA: split={split}, context_type={context_type}")
    
    # Determine file key
    prefix = "temp" if context_type == "temporal" else "geo"
    file_key = f"{prefix}.{split}"
    
    if file_key not in SITUATEDQA_FILES:
        available = [k for k in SITUATEDQA_FILES.keys() if k.startswith(prefix)]
        raise ValueError(f"Split '{split}' not found for {context_type}. Available: {available}")
    
    url = SITUATEDQA_FILES[file_key]
    
    # Download or use cache
    data = _download_jsonl(url, cache_dir, f"situatedqa_{file_key}.jsonl")
    
    queries: List[TemporalQuery] = []
    
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
        
        question = item.get("question", item.get("edited_question", ""))
        answers = item.get("answer", [])
        date = item.get("date", "")
        
        query = TemporalQuery(
            id=str(item.get("id", i)),
            query=question,
            gold_answer=answers[0] if answers else "",
            temporal_focus="specific_time",
            key_time_anchors=[date] if date else [],
            metadata={
                "edited_question": item.get("edited_question", ""),
                "date": date,
                "date_type": item.get("date_type", ""),
                "all_answers": answers,
                "any_answer": item.get("any_answer", []),
                "source": "SituatedQA",
            },
        )
        queries.append(query)
    
    logger.info(f"Loaded {len(queries)} queries from SituatedQA/{file_key}")
    
    return TemporalDataset(
        queries=queries,
        name=f"situatedqa_{file_key}",
        metadata={"source": "mikejqzhang/SituatedQA", "split": split, "context_type": context_type},
    )


def _download_jsonl(url: str, cache_dir: Optional[str], filename: str) -> List[dict]:
    """Download JSONL file and return parsed data."""
    
    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / filename
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return _read_jsonl(cache_path)
    
    # Download
    logger.info(f"Downloading from {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    
    # Parse
    data = []
    for line in content.strip().split("\n"):
        if line:
            data.append(json.loads(line))
    
    # Cache if directory provided
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        with open(cache_path / filename, "w") as f:
            f.write(content)
        logger.info(f"Cached to {cache_path / filename}")
    
    return data


def _read_jsonl(path: Path) -> List[dict]:
    """Read JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
