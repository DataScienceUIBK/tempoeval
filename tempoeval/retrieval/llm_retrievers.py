"""Large language model based retrievers (Qwen, E5, SF, GritLM, Contriever, RaDeR, Diver)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import trange

from tempoeval.retrieval.base import (
    BaseRetriever,
    RetrievalConfig,
    EmbeddingCache,
    add_instruction_concat,
    last_token_pool,
)


class QwenE5SFRetriever(BaseRetriever):
    """
    Large embedding model retriever for Qwen, E5-Mistral, SF-Mistral.
    
    Supports:
    - alibaba-nlp/gte-qwen2-7b-instruct (qwen2)
    - alibaba-nlp/gte-qwen1.5-7b-instruct (qwen)
    - intfloat/e5-mistral-7b-instruct (e5)
    - salesforce/sfr-embedding-mistral (sf)
    
    Requires GPU.
    """
    
    name = "qwen_e5_sf"
    requires_gpu = True
    supports_instructions = True
    
    MODEL_CONFIGS = {
        "qwen2": {
            "model_name": "alibaba-nlp/gte-qwen2-7b-instruct",
            "max_length": 8192,
            "trust_remote_code": True,
        },
        "qwen": {
            "model_name": "alibaba-nlp/gte-qwen1.5-7b-instruct",
            "max_length": 8192,
            "trust_remote_code": True,
        },
        "e5": {
            "model_name": "intfloat/e5-mistral-7b-instruct",
            "max_length": 4096,
            "trust_remote_code": False,
        },
        "sf": {
            "model_name": "salesforce/sfr-embedding-mistral",
            "max_length": 4096,
            "trust_remote_code": False,
        },
    }
    
    def __init__(
        self,
        model_id: str = "qwen2",
        config: Optional[RetrievalConfig] = None,
        task: str = "default",
        query_instruction: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    ):
        super().__init__(config)
        
        if model_id not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model_id: {model_id}. Choose from {list(self.MODEL_CONFIGS.keys())}")
        
        self.model_id = model_id
        self.model_config = self.MODEL_CONFIGS[model_id]
        self.task = task
        self.query_instruction = query_instruction
        self._model = None
        self._tokenizer = None
        self._cache = None
    
    def _load_model(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModel
            
            print(f"Loading {self.model_config['model_name']}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['model_name'],
                trust_remote_code=self.model_config['trust_remote_code']
            )
            self._model = AutoModel.from_pretrained(
                self.model_config['model_name'],
                device_map="auto",
                trust_remote_code=self.model_config['trust_remote_code'],
                torch_dtype=torch.bfloat16
            ).eval()
            
            if hasattr(self._model.config, 'use_cache'):
                self._model.config.use_cache = False
            
            print("  ✓ Model loaded")
    
    def _get_cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(
                cache_dir=self.config.cache_dir,
                model_id=self.model_id,
                task=self.task
            )
        return self._cache
    
    @torch.no_grad()
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
            new_embeddings = []
            max_length = self.model_config['max_length']
            batch_size = kwargs.get('batch_size', 1)
            
            for start_idx in trange(0, len(docs_to_encode), batch_size, desc="Encoding"):
                batch = docs_to_encode[start_idx:start_idx + batch_size]
                batch_dict = self._tokenizer(
                    batch,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self._model.device)
                
                outputs = self._model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = embeddings.float().cpu().numpy()
                new_embeddings.append(embeddings)
            
            new_embeddings = np.concatenate(new_embeddings, axis=0)
            cache.add_embeddings(ids_to_encode, new_embeddings)
            cache.save()
        
        return cache.get_embeddings_in_order(doc_ids)
    
    @torch.no_grad()
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        self._load_model()
        
        # Add instruction
        queries = add_instruction_concat(queries, self.task, self.query_instruction)
        
        all_embeddings = []
        max_length = self.model_config['max_length']
        batch_size = kwargs.get('batch_size', 1)
        
        for start_idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
            batch = queries[start_idx:start_idx + batch_size]
            batch_dict = self._tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self._model.device)
            
            outputs = self._model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = embeddings.float().cpu().numpy()
            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)


class GritLMRetriever(BaseRetriever):
    """
    GritLM retriever - unified generation and embedding model.
    
    Model: GritLM/GritLM-7B
    """
    
    name = "grit"
    requires_gpu = True
    supports_instructions = True
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        model_path: str = "GritLM/GritLM-7B",
        task: str = "default",
        query_instruction: str = "Represent this query for searching relevant documents: ",
        doc_instruction: str = "",
        query_max_length: int = 256,
        doc_max_length: int = 2048,
    ):
        super().__init__(config)
        self.model_path = model_path
        self.task = task
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self._model = None
        self._cache = None
    
    def _load_model(self):
        if self._model is None:
            from gritlm import GritLM
            
            print(f"Loading GritLM from {self.model_path}...")
            self._model = GritLM(self.model_path, torch_dtype="auto", mode="embedding")
            
            # Disable cache for stability
            def unwrap(m):
                while hasattr(m, "module"):
                    m = m.module
                return m
            
            grit = unwrap(self._model)
            backbone = unwrap(grit.model)
            backbone.config.use_cache = False
            
            gc = getattr(backbone, "generation_config", None)
            if gc is not None:
                gc.use_cache = False
            
            grit.eval()
            torch.set_grad_enabled(False)
            print("  ✓ Model loaded")
    
    def _get_cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(
                cache_dir=self.config.cache_dir,
                model_id="grit",
                task=self.task
            )
        return self._cache
    
    @torch.no_grad()
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
                instruction=self.doc_instruction,
                batch_size=1,
                max_length=self.doc_max_length
            )
            cache.add_embeddings(ids_to_encode, new_embeddings)
            cache.save()
        
        return cache.get_embeddings_in_order(doc_ids)
    
    @torch.no_grad()
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        self._load_model()
        
        instruction = self.query_instruction.format(task=self.task)
        return self._model.encode(
            queries,
            instruction=instruction,
            batch_size=1,
            max_length=self.query_max_length
        )


class ContrieverRetriever(BaseRetriever):
    """
    Contriever/Contriever-MSMARCO retriever.
    
    Facebook's unsupervised dense retriever.
    """
    
    name = "contriever"
    requires_gpu = True
    supports_instructions = False
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        model_name: str = "facebook/contriever-msmarco",
        task: str = "default",
    ):
        super().__init__(config)
        self.model_name = model_name
        self.task = task
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModel
            
            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self._model.eval()
            print("  ✓ Model loaded")
    
    def _mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    @torch.no_grad()
    def _encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        self._load_model()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True, return_tensors='pt'
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            outputs = self._model(**inputs)
            embeddings = self._mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
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
        
        doc_emb = self._encode(documents, batch_size=self.config.batch_size)
        np.save(cache_file, doc_emb)
        return doc_emb
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        return self._encode(queries, batch_size=self.config.batch_size)


class DiverRetriever(BaseRetriever):
    """
    Diver-Retriever (Qwen3-based) retriever.
    
    Model: AQ-MedAI/Diver-Retriever-4B
    """
    
    name = "diver"
    requires_gpu = True
    supports_instructions = True
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        model_path: str = "AQ-MedAI/Diver-Retriever-4B",
        task: str = "default",
        max_length: int = 8192,
    ):
        super().__init__(config)
        self.model_path = model_path
        self.task = task
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"Loading Diver-Retriever from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16
            ).eval()
            print("  ✓ Model loaded")
    
    def _truncate_text(self, text: str) -> str:
        text_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
            text = self._tokenizer.decode(text_ids)
        return text
    
    @torch.no_grad()
    def _embed_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        self._load_model()
        
        if is_query:
            texts = [f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{t}' for t in texts]
        else:
            texts = [f'Represent this text:{t}' for t in texts]
        
        texts = [self._truncate_text(t) for t in texts]
        
        all_embeddings = []
        batch_size = self.config.batch_size
        
        for i in trange(0, len(texts), batch_size, desc="Embedding"):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self._model.device)
            
            outputs = self._model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            all_embeddings.append(embeddings.float().cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
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
        
        doc_emb = self._embed_batch(documents, is_query=False)
        np.save(cache_file, doc_emb)
        return doc_emb
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        return self._embed_batch(queries, is_query=True)
