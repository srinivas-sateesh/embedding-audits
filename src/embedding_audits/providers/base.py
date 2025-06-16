"""
Base class forembedding providers.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingProvider(ABC):
    """
    Base class for embedding providers.
    """

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed a list of documents into vectors.

        Args:
            documents: List of text documents to embed.

        Returns:
            Numpy array of shape (num_documents, embedding_dim).
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string into a vector.

        Args:
            query: Query string to embed.

        Returns:
            Numpy array of shape (embedding_dim,).
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Name of the embedding model.

        Returns:
            String representing the model name.
        """
        pass