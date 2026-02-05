"""Synthetic failure generators for meta-evaluation."""

import random
import re
from typing import Any, Dict, List
from tempoeval.meta_eval.base import BasePerturbator

class DateScrambler(BasePerturbator):
    """Randomly changes years in the text to incorrect ones."""
    
    @property
    def name(self) -> str:
        return "date_scrambler"
        
    def perturb(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = input_data.copy()
        text = result.get("answer", "")
        
        # Simple regex to find 4-digit years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        
        for year in years:
            # Change year by +/- 5-50 years
            offset = random.choice([-1, 1]) * random.randint(5, 50)
            new_year = str(int(year) + offset)
            text = text.replace(year, new_year)
            
        result["answer"] = text
        return result

class EventShuffler(BasePerturbator):
    """Shuffles the order of sentences/events."""
    
    @property
    def name(self) -> str:
        return "event_shuffler"
        
    def perturb(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = input_data.copy()
        text = result.get("answer", "")
        
        # Split by simple sentence boundary
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if len(sentences) > 1:
            random.shuffle(sentences)
            result["answer"] = ". ".join(sentences) + "."
            
        return result

class HallucinationInjector(BasePerturbator):
    """Injects fake temporal claims."""
    
    @property
    def name(self) -> str:
        return "hallucination_injector"
        
    def perturb(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = input_data.copy()
        text = result.get("answer", "")
        
        fakes = [
            " In 1999, a major update occurred.",
            " This lasted for exactly 100 years.",
            " By 2050, the process completed."
        ]
        
        result["answer"] = text + random.choice(fakes)
        return result
