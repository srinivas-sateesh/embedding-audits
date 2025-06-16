"""
OpenAI Embedding Provider
"""

import os
from typing import List
import numpy as np

try:
    import openai
except ImportError:
    OpenAI = None

from .base import EmbeddingProvider

class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embdedding-3-small
    """


    def __init__(self,model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding provider.

        Args:
            model: Model name to use for embeddings.
                   Default is "text-embedding-3-small".
        """
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Please install it with 'pip install openai'.")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self,texts: List[str]) -> np.ndarray:
        """
        Embed documents using OpenAI API.
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def embed_query(self,query: str) -> np.ndarray:
        """
        Embed a single query using OpenAI API.
        """
        return self.embed_documents([query])[0]
    
    @property
    def model_name(self) -> str:
        """
        Return the name of the OpenAI embedding model.
        """
        return f" OpenAI {self.model}"

        