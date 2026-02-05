"""API-based retrievers (OpenAI, Azure, Cohere, Voyage, Google)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import trange

from tempoeval.retrieval.base import (
    BaseRetriever,
    RetrievalConfig,
    cut_text_tiktoken,
)


class OpenAIRetriever(BaseRetriever):
    """
    OpenAI text-embedding-3-large retriever.
    
    Environment variables:
        OPENAI_API_KEY: OpenAI API key
    """
    
    name = "openai"
    requires_gpu = False
    supports_instructions = False
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        model: str = "text-embedding-3-large",
        task: str = "default",
        max_tokens: int = 6000,
    ):
        super().__init__(config)
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self._client = None
        self._tokenizer = None
        self._cache_dir = None
    
    def _init_client(self):
        if self._client is None:
            from openai import OpenAI
            import tiktoken
            self._client = OpenAI()
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _get_embeddings(self, texts: List[str], batch_size: int = 1024) -> List[List[float]]:
        """Get embeddings from OpenAI API."""
        self._init_client()
        
        # Truncate texts
        texts = [cut_text_tiktoken(t, self._tokenizer, self.max_tokens) for t in texts]
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        """Encode documents with caching."""
        cache_dir = Path(self.config.cache_dir) / 'doc_emb' / self.name / self.task
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / '0.npy'
        
        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file, allow_pickle=True)
        
        print("Encoding documents with OpenAI...")
        embeddings = self._get_embeddings(documents, batch_size=self.config.batch_size)
        doc_emb = np.array(embeddings)
        
        np.save(cache_file, doc_emb)
        print(f"  ✓ Saved {len(doc_emb)} embeddings to cache")
        
        return doc_emb
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries."""
        embeddings = self._get_embeddings(queries, batch_size=self.config.batch_size)
        return np.array(embeddings)


class AzureOpenAIRetriever(BaseRetriever):
    """
    Azure OpenAI embedding retriever.
    
    Environment variables:
        AZURE_OPENAI_ENDPOINT: Azure endpoint
        AZURE_OPENAI_API_KEY: API key
        AZURE_DEPLOYMENT_NAME: Embedding deployment name (e.g., text-embedding-3-large)
        AZURE_API_VERSION: API version
    """
    
    name = "azure_openai"
    requires_gpu = False
    supports_instructions = False
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        task: str = "default",
        max_tokens: int = 6000,
    ):
        super().__init__(config)
        self.task = task
        self.max_tokens = max_tokens
        self._client = None
        self._tokenizer = None
        self._deployment = None
    
    def _init_client(self):
        if self._client is None:
            from openai import AzureOpenAI
            import tiktoken
            
            self._client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION", "2024-05-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self._deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _get_embeddings(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Get embeddings from Azure OpenAI (handles smaller batch limits)."""
        self._init_client()
        
        # Truncate texts
        texts = [cut_text_tiktoken(t, self._tokenizer, self.max_tokens) for t in texts]
        texts = [t if t.strip() else " " for t in texts]  # Handle empty strings
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(3):
                try:
                    response = self._client.embeddings.create(
                        input=batch,
                        model=self._deployment
                    )
                    all_embeddings.extend([item.embedding for item in response.data])
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise
        
        return all_embeddings
    
    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        cache_dir = Path(self.config.cache_dir) / 'doc_emb' / self.name / self.task
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / '0.npy'
        
        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file, allow_pickle=True)
        
        print("Encoding documents with Azure OpenAI...")
        embeddings = []
        for i in trange(0, len(documents), self.config.batch_size, desc="Encoding"):
            batch_emb = self._get_embeddings(
                documents[i:i + self.config.batch_size],
                batch_size=16  # Azure limit
            )
            embeddings.extend(batch_emb)
        
        doc_emb = np.array(embeddings)
        np.save(cache_file, doc_emb)
        print(f"  ✓ Saved {len(doc_emb)} embeddings to cache")
        
        return doc_emb
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        embeddings = self._get_embeddings(queries, batch_size=16)
        return np.array(embeddings)


class CohereRetriever(BaseRetriever):
    """
    Cohere embed-english-v3.0 retriever.
    
    Environment variables:
        COHERE_API_KEY: Cohere API key
    """
    
    name = "cohere"
    requires_gpu = False
    supports_instructions = False
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        model: str = "embed-english-v3.0",
        task: str = "default",
    ):
        super().__init__(config)
        self.model = model
        self.task = task
        self._client = None
    
    def _init_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client()
    
    def _get_embeddings(
        self, 
        texts: List[str], 
        input_type: str = "search_document",
        batch_size: int = 96
    ) -> List[List[float]]:
        self._init_client()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(5):
                try:
                    response = self._client.embed(
                        texts=batch,
                        input_type=input_type,
                        model=self.model
                    )
                    all_embeddings.extend(response.embeddings)
                    break
                except Exception as e:
                    print(f"Cohere error: {e}")
                    time.sleep(60)
        
        return all_embeddings
    
    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        cache_dir = Path(self.config.cache_dir) / 'doc_emb' / self.name / self.task
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / '0.npy'
        
        if cache_file.exists():
            return np.load(cache_file, allow_pickle=True)
        
        print("Encoding documents with Cohere...")
        embeddings = self._get_embeddings(documents, input_type="search_document")
        doc_emb = np.array(embeddings)
        
        np.save(cache_file, doc_emb)
        return doc_emb
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        embeddings = self._get_embeddings(queries, input_type="search_query")
        return np.array(embeddings)


class VoyageRetriever(BaseRetriever):
    """
    Voyage AI voyage-large-2-instruct retriever.
    
    Environment variables:
        VOYAGE_API_KEY: Voyage API key
    """
    
    name = "voyage"
    requires_gpu = False
    supports_instructions = True
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        model: str = "voyage-large-2-instruct",
        task: str = "default",
        max_tokens: int = 16000,
    ):
        super().__init__(config)
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self._client = None
        self._tokenizer = None
    
    def _init_client(self):
        if self._client is None:
            import voyageai
            from transformers import AutoTokenizer
            self._client = voyageai.Client()
            self._tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    
    def _truncate(self, text: str) -> str:
        from tempoeval.retrieval.base import cut_text
        return cut_text(text, self._tokenizer, self.max_tokens)
    
    def encode_documents(
        self,
        documents: List[str],
        doc_ids: List[str],
        **kwargs
    ) -> np.ndarray:
        self._init_client()
        
        cache_dir = Path(self.config.cache_dir) / 'doc_emb' / self.name / self.task
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / '0.npy'
        
        if cache_file.exists():
            return np.load(cache_file, allow_pickle=True)
        
        print("Encoding documents with Voyage...")
        documents = [self._truncate(d) for d in documents]
        
        all_embeddings = []
        for i in trange(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            
            for attempt in range(5):
                try:
                    response = self._client.embed(
                        batch,
                        model=self.model,
                        input_type="document"
                    )
                    all_embeddings.extend(response.embeddings)
                    break
                except Exception as e:
                    print(f"Voyage error: {e}")
                    time.sleep(5)
        
        doc_emb = np.array(all_embeddings)
        np.save(cache_file, doc_emb)
        return doc_emb
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        self._init_client()
        queries = [self._truncate(q) for q in queries]
        
        all_embeddings = []
        for i in range(0, len(queries), self.config.batch_size):
            batch = queries[i:i + self.config.batch_size]
            response = self._client.embed(batch, model=self.model, input_type="query")
            all_embeddings.extend(response.embeddings)
        
        return np.array(all_embeddings)
