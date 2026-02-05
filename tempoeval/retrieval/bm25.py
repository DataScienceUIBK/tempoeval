"""BM25 Retriever using Pyserini and Gensim."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from tempoeval.retrieval.base import BaseRetriever, RetrievalConfig, get_scores


class BM25Retriever(BaseRetriever):
    """
    BM25 lexical retriever using Pyserini/Gensim.
    
    A fast, CPU-based sparse retriever that doesn't require embeddings.
    Good baseline for temporal retrieval tasks.
    
    Example:
        >>> retriever = BM25Retriever()
        >>> results = retriever.retrieve(
        ...     queries=["When did WWI begin?"],
        ...     query_ids=["q1"],
        ...     documents=["WWI started in 1914...", "The war ended in 1918..."],
        ...     doc_ids=["d1", "d2"]
        ... )
    """
    
    name = "bm25"
    requires_gpu = False
    supports_instructions = False
    
    def __init__(
        self, 
        config: Optional[RetrievalConfig] = None,
        k1: float = 0.9,
        b: float = 0.4,
    ):
        super().__init__(config)
        self.k1 = k1
        self.b = b
        self._model = None
        self._dictionary = None
        self._index = None
        self._analyzer = None
    
    def _init_model(self, documents: List[str]):
        """Initialize BM25 model with corpus."""
        from pyserini import analysis
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity
        
        self._analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        
        # Tokenize corpus
        corpus = [self._analyzer.analyze(doc) for doc in tqdm(documents, desc="Tokenizing")]
        
        # Build dictionary and model
        self._dictionary = Dictionary(corpus)
        self._model = LuceneBM25Model(dictionary=self._dictionary, k1=self.k1, b=self.b)
        
        # Build index
        bm25_corpus = self._model[list(map(self._dictionary.doc2bow, corpus))]
        self._index = SparseMatrixSimilarity(
            bm25_corpus,
            num_docs=len(corpus),
            num_terms=len(self._dictionary),
            normalize_queries=False,
            normalize_documents=False
        )
    
    def encode_documents(
        self, 
        documents: List[str], 
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        """BM25 doesn't use embeddings - initialize index instead."""
        if self._model is None:
            self._init_model(documents)
        return np.zeros((len(documents), 1))  # Placeholder
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """BM25 doesn't use embeddings."""
        return np.zeros((len(queries), 1))  # Placeholder
    
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
        """Run BM25 retrieval."""
        if top_k is None:
            top_k = self.config.top_k
        
        if excluded_ids is None:
            excluded_ids = {str(qid): [] for qid in query_ids}
        
        # Initialize model if needed
        if self._model is None:
            self._init_model(documents)
        
        all_scores = {}
        
        for query_id, query in tqdm(zip(query_ids, queries), total=len(queries), desc="BM25 Retrieval"):
            # Tokenize and score
            query_tokens = self._analyzer.analyze(query)
            bm25_query = self._model[self._dictionary.doc2bow(query_tokens)]
            similarities = self._index[bm25_query].tolist()
            
            # Build scores dict
            cur_scores = {}
            for did, s in zip(doc_ids, similarities):
                cur_scores[str(did)] = float(s)
            
            # Remove excluded docs
            for did in set(excluded_ids.get(str(query_id), [])):
                if did != "N/A" and did in cur_scores:
                    cur_scores.pop(did)
            
            # Sort and keep top-k
            sorted_scores = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            all_scores[str(query_id)] = {pair[0]: pair[1] for pair in sorted_scores}
        
        return all_scores
