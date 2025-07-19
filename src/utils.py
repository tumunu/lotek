"""
LoTek Fractal Network Utilities Module

Provides utility functions for the LoTek fractal cybersecurity neural network,
including hashing, embedding storage, and analysis helpers.
"""

import torch
import hashlib
from config import EMBEDDING_DIM

def fractal_hash(embedding):
    """Generate a hash for LoTek fractal network embeddings."""
    # A simple hashing function, in practice, you'd want something more sophisticated
    return int(hashlib.md5(str(embedding.tolist()).encode()).hexdigest(), 16)

def store_embedding_hash(model, X):
    """Store embedding hashes for LoTek fractal network analysis."""
    with torch.no_grad():
        embeddings, _ = model.attention(X)
        hashes = [fractal_hash(e) for e in embeddings.view(-1, EMBEDDING_DIM)]
    return hashes