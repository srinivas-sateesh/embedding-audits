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


def quick_compare(
    documents: List[str],
    queries: List[str],
    relevant_docs: Optional[Dict[int,List[int]]] = None,
    models: Optional[List[str]] = None,
    chunk_size: int = 500
) -> ComparisonResult:

"""
    Compare embedding models on your data
    
    Args:
        documents: List of text documents
        queries: List of query strings  
        relevant_docs: Dict mapping query_idx -> [relevant_doc_indices]
                      If None, will auto-generate simple relevance
        models: List of model names to compare
               If None, uses ["openai", "sentence-transformers"]
        chunk_size: Token size for chunking documents
    
    Returns:
        ComparisonResult with model performance and recommendations
        
    Example:
        >>> results = quick_compare(
        ...     documents=["AI is transforming healthcare", "Machine learning helps doctors"],
        ...     queries=["How does AI help healthcare?"]
        ... )
        >>> print(f"Best model: {results.best_model}")
"""

if models is None:
    models = ["openai","sentence-transformers"]

    chunks =[]
    chunk_to_doc = []
    for doc_idx,doc in enumerate(documents):
        doc_chunks = simple_chunk(doc,chunk_size)
        chunks.extend(doc_chunks)
        chunk_to_doc.extend([doc_idx] * len(doc_chunks))

       # Auto-generate relevance if not provided
    if relevant_docs is None:
        relevant_docs = _auto_generate_relevance(queries, documents)
     # Evaluate each model
    model_scores = {}
    
    for model_name in models:
        try:
            provider = get_provider(model_name)
            scores = _evaluate_model(provider, chunks, queries, relevant_docs, chunk_to_doc)
            model_scores[model_name] = scores
            print(f"✅ Evaluated {model_name}")
        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")
            continue
    
    if not model_scores:
        raise ValueError("No models could be evaluated. Check API keys and dependencies.")
    
    # Find best model (by recall@5)
    best_model = max(model_scores.keys(), 
                    key=lambda m: model_scores[m].get('recall@5', 0))
    
    return ComparisonResult(
        model_scores=model_scores,
        best_model=best_model
    )


def _evaluate_model(provider, chunks, queries, relevant_docs, chunk_to_doc):
    """Evaluate a single embedding model"""
    
    # Embed all chunks
    chunk_embeddings = provider.embed_documents(chunks)
    
    all_recall_5 = []
    all_precision_5 = []
    
    for query_idx, query in enumerate(queries):
        # Embed query
        query_embedding = provider.embed_query(query)
        
        # Calculate similarities
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Get top 5 chunks
        top_indices = np.argsort(similarities)[-5:][::-1]
        top_doc_indices = [chunk_to_doc[i] for i in top_indices]
        
        # Calculate metrics
        relevant_set = set(relevant_docs.get(query_idx, []))
        retrieved_set = set(top_doc_indices)
        
        recall = recall_at_k(retrieved_set, relevant_set, k=5)
        precision = precision_at_k(retrieved_set, relevant_set, k=5)
        
        all_recall_5.append(recall)
        all_precision_5.append(precision)
    
    return {
        'recall@5': np.mean(all_recall_5),
        'precision@5': np.mean(all_precision_5),
        'avg_similarity': np.mean([np.max(np.dot(chunk_embeddings, provider.embed_query(q))) 
                                  for q in queries])
    }


def _auto_generate_relevance(queries, documents):
    """Simple heuristic to generate relevance judgments"""
    # Very simple: document is relevant if it shares words with query
    relevant_docs = {}
    
    for query_idx, query in enumerate(queries):
        query_words = set(query.lower().split())
        relevant = []
        
        for doc_idx, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            if len(query_words & doc_words) > 0:  # Any word overlap
                relevant.append(doc_idx)
        
        relevant_docs[query_idx] = relevant if relevant else [0]  # At least one relevant
    
    return relevant_docs