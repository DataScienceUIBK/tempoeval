"""Temporal Relevance metric - ratio of temporally relevant documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tempoeval.metrics.retrieval.temporal_precision import TEMPORAL_RELEVANCE_PROMPT


@dataclass
class TemporalRelevance:
    """
    Temporal Relevance measures the simple ratio of temporally relevant 
    documents in the retrieved set.
    
    Unlike TemporalPrecision which uses position-weighting, this metric
    treats all positions equally - it's simply:
    
        relevance = (# of temporally relevant docs) / (# of retrieved docs)
    
    This is useful when you want to know what percentage of your retrieved
    documents contain useful temporal information, regardless of their rank.
    
    Example:
        >>> from tempoeval.metrics import TemporalRelevance
        >>> metric = TemporalRelevance()
        >>> metric.llm = llm  # Set LLM provider
        >>> score = metric.compute(
        ...     query="When did WWI begin?",
        ...     retrieved_docs=["WWI started in 1914...", "General war info..."],
        ...     temporal_focus="specific_time",
        ...     k=5
        ... )
        >>> print(score)  # 0.5 (1 out of 2 docs relevant)
    
    Attributes:
        name: Metric name for reporting
        llm: LLM provider for judging temporal relevance
    """
    
    name: str = field(init=False, default="temporal_relevance")
    llm: Optional[object] = field(default=None)
    
    def compute(
        self,
        query: str,
        retrieved_docs: List[str],
        temporal_focus: str = "specific_time",
        k: Optional[int] = None,
        **kwargs
    ) -> float:
        """
        Compute temporal relevance score.
        
        Args:
            query: The search query
            retrieved_docs: List of retrieved document texts
            temporal_focus: Type of temporal query (specific_time, duration, etc.)
            k: Number of top documents to consider (default: all)
            
        Returns:
            Relevance score between 0 and 1
        """
        if self.llm is None:
            raise ValueError("LLM provider required. Set metric.llm = your_llm_provider")
        
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        
        if not retrieved_docs:
            return 0.0
        
        verdicts = []
        for doc in retrieved_docs:
            doc_text = doc[:2000] if len(doc) > 2000 else doc
            try:
                prompt = TEMPORAL_RELEVANCE_PROMPT.format(
                    query=query,
                    temporal_focus=temporal_focus,
                    document=doc_text
                )
                result = self.llm.generate_json(prompt)
                verdict = 1 if int(result.get("verdict", 0)) == 1 else 0
            except Exception:
                verdict = 0
            verdicts.append(verdict)
        
        # Simple ratio - no position weighting
        relevant_count = sum(verdicts)
        return round(relevant_count / len(verdicts), 5)
    
    async def acompute(
        self,
        query: str,
        retrieved_docs: List[str],
        temporal_focus: str = "specific_time",
        k: Optional[int] = None,
        **kwargs
    ) -> float:
        """Async version of compute."""
        if self.llm is None:
            raise ValueError("LLM provider required. Set metric.llm = your_llm_provider")
        
        if k is not None:
            retrieved_docs = retrieved_docs[:k]
        
        if not retrieved_docs:
            return 0.0
        
        verdicts = []
        for doc in retrieved_docs:
            doc_text = doc[:2000] if len(doc) > 2000 else doc
            try:
                prompt = TEMPORAL_RELEVANCE_PROMPT.format(
                    query=query,
                    temporal_focus=temporal_focus,
                    document=doc_text
                )
                if hasattr(self.llm, 'agenerate_json'):
                    result = await self.llm.agenerate_json(prompt)
                else:
                    result = self.llm.generate_json(prompt)
                verdict = 1 if int(result.get("verdict", 0)) == 1 else 0
            except Exception:
                verdict = 0
            verdicts.append(verdict)
        
        relevant_count = sum(verdicts)
        return round(relevant_count / len(verdicts), 5)
