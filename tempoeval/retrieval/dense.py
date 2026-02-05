"""Dense retrievers using SentenceTransformers (SBERT, BGE, Instructor, Nomic)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from tqdm import trange

from tempoeval.retrieval.base import (
    BaseRetriever, 
    RetrievalConfig, 
    EmbeddingCache,
    add_instruction_concat,
)


class SentenceTransformerRetriever(BaseRetriever):
    """
    Dense retriever using SentenceTransformers.
    
    Supports various models:
    - sentence-transformers/all-mpnet-base-v2 (SBERT)
    - BAAI/bge-large-en-v1.5 (BGE)
    - nomic-ai/nomic-embed-text-v1 (Nomic)
    
    Example:
        >>> retriever = SentenceTransformerRetriever(model_name="BAAI/bge-large-en-v1.5")
        >>> results = retriever.retrieve(queries, query_ids, documents, doc_ids)
    """
    
    name = "sbert"
    requires_gpu = True
    supports_instructions = True
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        config: Optional[RetrievalConfig] = None,
        task: str = "default",
        query_instruction: str = "",
        doc_instruction: str = "",
    ):
        super().__init__(config)
        self.model_name = model_name
        self.task = task
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self._model = None
        self._cache = None
    
    def _load_model(self):
        """Lazy load model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            print("  ✓ Model loaded")
    
    def _get_cache(self) -> EmbeddingCache:
        """Get or create cache."""
        if self._cache is None:
            model_id = self.model_name.replace("/", "_")
            self._cache = EmbeddingCache(
                cache_dir=self.config.cache_dir,
                model_id=model_id,
                task=self.task
            )
        return self._cache
    
    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        """Encode documents with caching."""
        self._load_model()
        cache = self._get_cache()
        
        # Check what needs encoding
        ids_to_encode, docs_to_encode = cache.get_missing_docs(doc_ids, documents)
        
        if docs_to_encode:
            print("Encoding new documents...")
            
            # Add document instruction if provided
            if self.doc_instruction:
                docs_to_encode = add_instruction_concat(
                    docs_to_encode, self.task, self.doc_instruction
                )
            
            new_embeddings = self._model.encode(
                docs_to_encode,
                show_progress_bar=True,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            cache.add_embeddings(ids_to_encode, new_embeddings)
            cache.save()
        
        return cache.get_embeddings_in_order(doc_ids)
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries."""
        self._load_model()
        
        # Add query instruction if provided
        if self.query_instruction:
            queries = add_instruction_concat(queries, self.task, self.query_instruction)
        
        return self._model.encode(
            queries,
            show_progress_bar=True,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings
        )


class BGERetriever(SentenceTransformerRetriever):
    """BGE retriever with default settings."""
    
    name = "bge"
    
    def __init__(self, config: Optional[RetrievalConfig] = None, task: str = "default"):
        super().__init__(
            model_name="BAAI/bge-large-en-v1.5",
            config=config,
            task=task,
            query_instruction="Represent this sentence for searching relevant passages: ",
        )


class NomicRetriever(SentenceTransformerRetriever):
    """Nomic retriever with default settings."""
    
    name = "nomic"
    
    def __init__(self, config: Optional[RetrievalConfig] = None, task: str = "default"):
        super().__init__(
            model_name="nomic-ai/nomic-embed-text-v1",
            config=config,
            task=task,
        )


class InstructorRetriever(BaseRetriever):
    """
    Instructor embedding model retriever.
    
    Supports:
    - hkunlp/instructor-large
    - hkunlp/instructor-xl
    """
    
    name = "instructor"
    requires_gpu = True
    supports_instructions = True
    
    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        config: Optional[RetrievalConfig] = None,
        task: str = "default",
        query_instruction: str = "Represent the question for retrieving supporting documents: ",
        doc_instruction: str = "Represent the document for retrieval: ",
    ):
        super().__init__(config)
        self.model_name = model_name
        self.task = task
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self._model = None
        self._cache = None
    
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            self._model.set_pooling_include_prompt(False)
            print("  ✓ Model loaded")
    
    def _get_cache(self) -> EmbeddingCache:
        if self._cache is None:
            model_id = self.model_name.replace("/", "_")
            self._cache = EmbeddingCache(
                cache_dir=self.config.cache_dir,
                model_id=model_id,
                task=self.task
            )
        return self._cache
    
    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        self._load_model()
        cache = self._get_cache()
        
        ids_to_encode, docs_to_encode = cache.get_missing_docs(doc_ids, documents)
        
        if docs_to_encode:
            print("Encoding new documents...")
            new_embeddings = self._model.encode(
                docs_to_encode,
                show_progress_bar=True,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                prompt=self.doc_instruction.format(task=self.task)
            )
            cache.add_embeddings(ids_to_encode, new_embeddings)
            cache.save()
        
        return cache.get_embeddings_in_order(doc_ids)
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        self._load_model()
        return self._model.encode(
            queries,
            show_progress_bar=True,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            prompt=self.query_instruction.format(task=self.task)
        )
