# test_compression.py - Pytest for compression

import pytest
import torch
from src.fractal_modules.compression import FractalCompression

@pytest.fixture
def compression_module():
    return FractalCompression(embedding_dim=32)

def test_encode_decode(compression_module):
    X = torch.randn(10, 32)
    encoded = compression_module.encode(X)
    decoded = compression_module.decode(encoded)
    assert torch.allclose(X, decoded, atol=1e-4), "Encoding and decoding should reconstruct the original"

def test_hierarchical_compression(compression_module):
    X = torch.randn(10, 32)
    compressed = compression_module(X)
    assert compressed.shape == X.shape, "Hierarchical compression should maintain shape after decompression"