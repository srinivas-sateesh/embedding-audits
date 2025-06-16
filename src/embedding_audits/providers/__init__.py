from .openai_provider import OpenAIProvider
from .sentence_transformers_provider import SentenceTransformersProvider

_PROVIDERS = {
    "openai": OpenAIProvider,
    "sentence-transformers": SentenceTransformersProvider,
}

def get_provider(name: str):
    """
    Get the embedding provider class by name.
    
    Args:
        name: Name of the provider (e.g. "openai", "sentence-transformers")
    
    Returns:
        Provider class instance
    """
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available providers: {list(_PROVIDERS.keys())}")
    
    return _PROVIDERS[name]()

def get_available_providers():
    """
    Get a list of available embedding providers.
    
    Returns:
        List of provider names
    """
    return list(_PROVIDERS.keys())