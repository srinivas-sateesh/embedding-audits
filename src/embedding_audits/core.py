import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json

from .providers import get_provider
from .metrics import recall_at_k, precision_at_k
from .chunking import simple_chunk

@dataclass
class ComparisonResult:
    """Results for embedding model comparison"""
    model_scores: Dict[str,Dict[str,float]]
    best_model: str

    def summary(self) -> str:
        """Summary of Results"""
        lines= ["Embedding Models comparison results for your documents" ]
        for model,scores in self.model_scores.items():
            emoji = "🏆" if model == self.best_model else "📈"
            lines.append(f"{emoji}{model}")
            for metric,score in scores.items()
                lines.append(f"     {metric}:{score:.3f}")
            lines.append("")
        return "\n".join(lines)
        