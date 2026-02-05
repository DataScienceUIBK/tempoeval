"""
TempoEval Retrieval Module
==============================

Comprehensive retrieval module with multiple embedding models and caching support.

Retrievers:
    Sparse:
        - BM25Retriever: Pyserini/Gensim BM25
    
    Dense (SentenceTransformers):
        - SentenceTransformerRetriever: Generic ST model
        - BGERetriever: BAAI/bge-large-en-v1.5
        - NomicRetriever: nomic-ai/nomic-embed-text-v1
        - InstructorRetriever: hkunlp/instructor-large/xl
    
    API-based:
        - OpenAIRetriever: text-embedding-3-large
        - AzureOpenAIRetriever: Azure OpenAI embeddings
        - CohereRetriever: embed-english-v3.0
        - VoyageRetriever: voyage-large-2-instruct
    
    LLM-based (GPU):
        - QwenE5SFRetriever: Qwen2/E5-Mistral/SF-Mistral
        - GritLMRetriever: GritLM-7B
        - ContrieverRetriever: contriever-msmarco
        - DiverRetriever: Diver-Retriever-4B

Example:
    >>> from tempoeval.retrieval import BM25Retriever, BGERetriever
    >>> 
    >>> # BM25 retrieval
    >>> bm25 = BM25Retriever()
    >>> results = bm25.retrieve(queries, query_ids, documents, doc_ids)
    >>> 
    >>> # Dense retrieval with BGE
    >>> bge = BGERetriever()
    >>> results = bge.retrieve(queries, query_ids, documents, doc_ids)
"""

from tempoeval.retrieval.base import (
    BaseRetriever,
    RetrievalConfig,
    RetrievalResult,
    EmbeddingCache,
    get_scores,
)

from tempoeval.retrieval.bm25 import BM25Retriever

from tempoeval.retrieval.dense import (
    SentenceTransformerRetriever,
    BGERetriever,
    NomicRetriever,
    InstructorRetriever,
)

from tempoeval.retrieval.api_retrievers import (
    OpenAIRetriever,
    AzureOpenAIRetriever,
    CohereRetriever,
    VoyageRetriever,
)

from tempoeval.retrieval.llm_retrievers import (
    QwenE5SFRetriever,
    GritLMRetriever,
    ContrieverRetriever,
    DiverRetriever,
)


# Registry of all retrievers
RETRIEVER_REGISTRY = {
    # Sparse
    "bm25": BM25Retriever,
    
    # Dense - SentenceTransformers
    "sbert": SentenceTransformerRetriever,
    "bge": BGERetriever,
    "nomic": NomicRetriever,
    "instructor": InstructorRetriever,
    
    # API-based
    "openai": OpenAIRetriever,
    "azure_openai": AzureOpenAIRetriever,
    "cohere": CohereRetriever,
    "voyage": VoyageRetriever,
    
    # LLM-based
    "qwen2": lambda **kw: QwenE5SFRetriever(model_id="qwen2", **kw),
    "qwen": lambda **kw: QwenE5SFRetriever(model_id="qwen", **kw),
    "e5": lambda **kw: QwenE5SFRetriever(model_id="e5", **kw),
    "sf": lambda **kw: QwenE5SFRetriever(model_id="sf", **kw),
    "grit": GritLMRetriever,
    "contriever": ContrieverRetriever,
    "diver": DiverRetriever,
}


def get_retriever(name: str, **kwargs) -> BaseRetriever:
    """
    Get retriever by name.
    
    Args:
        name: Retriever name (bm25, bge, openai, cohere, qwen2, etc.)
        **kwargs: Additional arguments passed to retriever
        
    Returns:
        Retriever instance
        
    Example:
        >>> retriever = get_retriever("bge", task="temporal_qa")
        >>> results = retriever.retrieve(queries, query_ids, docs, doc_ids)
    """
    if name not in RETRIEVER_REGISTRY:
        available = ", ".join(sorted(RETRIEVER_REGISTRY.keys()))
        raise ValueError(f"Unknown retriever: {name}. Available: {available}")
    
    retriever_cls = RETRIEVER_REGISTRY[name]
    if callable(retriever_cls) and not isinstance(retriever_cls, type):
        return retriever_cls(**kwargs)
    return retriever_cls(**kwargs)


def list_retrievers() -> list:
    """List all available retriever names."""
    return sorted(RETRIEVER_REGISTRY.keys())


__all__ = [
    # Base
    "BaseRetriever",
    "RetrievalConfig",
    "RetrievalResult", 
    "EmbeddingCache",
    "get_scores",
    
    # BM25
    "BM25Retriever",
    
    # Dense
    "SentenceTransformerRetriever",
    "BGERetriever",
    "NomicRetriever",
    "InstructorRetriever",
    
    # API
    "OpenAIRetriever",
    "AzureOpenAIRetriever",
    "CohereRetriever",
    "VoyageRetriever",
    
    # LLM
    "QwenE5SFRetriever",
    "GritLMRetriever",
    "ContrieverRetriever",
    "DiverRetriever",
    
    # Registry
    "RETRIEVER_REGISTRY",
    "get_retriever",
    "list_retrievers",
]
