"""Base retriever class and utilities."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    query_id: str
    ranked_doc_ids: List[str]
    scores: Dict[str, float]
    
    def top_k(self, k: int) -> List[str]:
        """Get top-k document IDs."""
        return self.ranked_doc_ids[:k]


@dataclass  
class RetrievalConfig:
    """Configuration for retrieval."""
    cache_dir: str = "./cache"
    batch_size: int = 32
    max_doc_length: int = 4096
    max_query_length: int = 512
    top_k: int = 100
    normalize_embeddings: bool = True
    use_cache: bool = True


def get_scores(
    query_ids: List[str],
    doc_ids: List[str], 
    scores: List[List[float]],
    excluded_ids: Optional[Dict[str, List[str]]] = None,
    top_k: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Convert score matrix to ranked results dict.
    
    Args:
        query_ids: List of query IDs
        doc_ids: List of document IDs
        scores: 2D score matrix [num_queries, num_docs]
        excluded_ids: Dict mapping query_id -> list of doc_ids to exclude
        top_k: Number of top results to keep per query
        
    Returns:
        Dict mapping query_id -> {doc_id: score}
    """
    if excluded_ids is None:
        excluded_ids = {str(qid): [] for qid in query_ids}
    
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(doc_ids), f"{len(scores[0])}, {len(doc_ids)}"
    
    results = {}
    for query_id, doc_scores in zip(query_ids, scores):
        cur_scores = {}
        for did, s in zip(doc_ids, doc_scores):
            cur_scores[str(did)] = float(s)
        
        # Remove excluded docs
        for did in set(excluded_ids.get(str(query_id), [])):
            if did != "N/A" and did in cur_scores:
                cur_scores.pop(did)
        
        # Sort and keep top-k
        cur_scores = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results[str(query_id)] = {pair[0]: pair[1] for pair in cur_scores}
    
    return results


class EmbeddingCache:
    """Smart caching for document embeddings with ID mapping."""
    
    def __init__(self, cache_dir: str, model_id: str, task: str = "default"):
        self.cache_dir = Path(cache_dir) / 'doc_emb' / model_id / task
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_path = self.cache_dir / 'embeddings.npy'
        self.mapping_path = self.cache_dir / 'doc_id_mapping.json'
        
        self.cached_embeddings: Dict[str, np.ndarray] = {}
        self.doc_id_to_index: Dict[str, int] = {}
        
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache if available."""
        if self.embeddings_path.exists() and self.mapping_path.exists():
            print(f"Loading existing cache from {self.cache_dir}...")
            cached_emb_array = np.load(self.embeddings_path, allow_pickle=True)
            with open(self.mapping_path, 'r') as f:
                self.doc_id_to_index = json.load(f)
            
            for doc_id, idx in self.doc_id_to_index.items():
                if idx < len(cached_emb_array):
                    self.cached_embeddings[doc_id] = cached_emb_array[idx]
            
            print(f"  ✓ Loaded {len(self.cached_embeddings)} cached embeddings")
    
    def get_missing_docs(
        self, 
        doc_ids: List[str], 
        documents: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Get documents that need to be encoded."""
        docs_to_encode = []
        docs_to_encode_ids = []
        
        for doc_id, doc_text in zip(doc_ids, documents):
            if doc_id not in self.cached_embeddings:
                docs_to_encode.append(doc_text)
                docs_to_encode_ids.append(doc_id)
        
        print(f"Documents in corpus: {len(documents)}")
        print(f"Already cached: {len(self.cached_embeddings)}")
        print(f"Need to encode: {len(docs_to_encode)}")
        
        return docs_to_encode_ids, docs_to_encode
    
    def add_embeddings(self, doc_ids: List[str], embeddings: np.ndarray):
        """Add new embeddings to cache."""
        next_index = len(self.cached_embeddings)
        
        for i, doc_id in enumerate(doc_ids):
            self.cached_embeddings[doc_id] = embeddings[i]
            self.doc_id_to_index[doc_id] = next_index + i
    
    def save(self):
        """Save cache to disk."""
        if not self.cached_embeddings:
            return
        
        print("Saving cache...")
        sorted_ids = sorted(self.doc_id_to_index.keys(), key=lambda x: self.doc_id_to_index[x])
        all_embeddings = np.array([self.cached_embeddings[doc_id] for doc_id in sorted_ids])
        
        np.save(self.embeddings_path, all_embeddings)
        with open(self.mapping_path, 'w') as f:
            json.dump(self.doc_id_to_index, f, indent=2)
        
        print(f"  ✓ Saved {len(self.cached_embeddings)} embeddings to cache")
    
    def get_embeddings_in_order(self, doc_ids: List[str]) -> np.ndarray:
        """Get embeddings in the order of provided doc_ids."""
        return np.array([self.cached_embeddings[doc_id] for doc_id in doc_ids])


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    
    All retrievers implement:
    - encode_documents: Encode corpus documents
    - encode_queries: Encode queries
    - retrieve: End-to-end retrieval
    """
    
    name: str = "base"
    requires_gpu: bool = False
    supports_instructions: bool = False
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
    
    @abstractmethod
    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        """
        Encode documents to embeddings.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            
        Returns:
            Numpy array of embeddings [num_docs, embedding_dim]
        """
        pass
    
    @abstractmethod
    def encode_queries(
        self,
        queries: List[str],
        **kwargs
    ) -> np.ndarray:
        """
        Encode queries to embeddings.
        
        Args:
            queries: List of query texts
            
        Returns:
            Numpy array of embeddings [num_queries, embedding_dim]
        """
        pass
    
    def retrieve(
        self,
        queries: List[str],
        query_ids: List[str],
        documents: List[str],
        doc_ids: List[str],
        excluded_ids: Optional[Dict[str, List[str]]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        End-to-end retrieval.
        
        Args:
            queries: List of query texts
            query_ids: List of query IDs
            documents: List of document texts
            doc_ids: List of document IDs
            excluded_ids: Documents to exclude per query
            top_k: Number of results per query
            
        Returns:
            Dict mapping query_id -> {doc_id: score}
        """
        if top_k is None:
            top_k = self.config.top_k
        
        if excluded_ids is None:
            excluded_ids = {str(qid): [] for qid in query_ids}
        
        # Encode documents (with caching)
        doc_emb = self.encode_documents(documents, doc_ids, **kwargs)
        
        # Encode queries
        query_emb = self.encode_queries(queries, **kwargs)
        
        # Compute scores
        scores = self._compute_similarity(query_emb, doc_emb)
        
        return get_scores(
            query_ids=query_ids,
            doc_ids=doc_ids,
            scores=scores,
            excluded_ids=excluded_ids,
            top_k=top_k
        )
    
    def _compute_similarity(
        self, 
        query_emb: np.ndarray, 
        doc_emb: np.ndarray
    ) -> List[List[float]]:
        """Compute cosine similarity between queries and documents."""
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_emb, doc_emb)
        return scores.tolist()


def add_instruction_concat(texts: List[str], task: str, instruction: str) -> List[str]:
    """Add instruction prefix to texts."""
    return [instruction.format(task=task) + t for t in texts]


def add_instruction_list(texts: List[str], task: str, instruction: str) -> List[List[str]]:
    """Add instruction as separate element."""
    return [[instruction.format(task=task), t] for t in texts]


def last_token_pool(last_hidden_states, attention_mask):
    """Extract last token embeddings (for decoder models)."""
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def cut_text(text: str, tokenizer, threshold: int) -> str:
    """Truncate text to max tokens."""
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text


def cut_text_tiktoken(text: str, tokenizer, threshold: int = 6000) -> str:
    """Truncate text using tiktoken tokenizer."""
    token_ids = tokenizer.encode(text, disallowed_special=())
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text
