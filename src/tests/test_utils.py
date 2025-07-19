# test_utils.py - Pytest for utils

import pytest
import torch
from src.utils import fractal_hash, store_embedding_hash
from src.model import FractalNetwork
from config import EMBEDDING_DIM, VOCAB_SIZE

def test_fractal_hash():
    embedding = torch.randn(EMBEDDING_DIM)
    hash_value = fractal_hash(embedding)
    assert isinstance(hash_value, int)

def test_store_embedding_hash():
    model = FractalNetwork()
    input = torch.randint(0, VOCAB_SIZE, (1, 10))
    hashes = store_embedding_hash(model, input)
    assert len(hashes) == 10  # Sequence length
    assert all(isinstance(h, int) for h in hashes)