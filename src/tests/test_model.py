# test_model.py - Pytest for model

import pytest
import torch
from src.model import FractalNetwork
from src.comparison_model import SimpleTransformer
from config import VOCAB_SIZE

def test_fractal_network_init():
    model = FractalNetwork()
    assert isinstance(model, FractalNetwork)

def test_fractal_network_forward():
    model = FractalNetwork()
    input = torch.randint(0, VOCAB_SIZE, (1, 10))  # Batch size 1, sequence length 10
    output = model(input)
    assert output.shape == (1, VOCAB_SIZE)

def test_simple_transformer_init():
    model = SimpleTransformer()
    assert isinstance(model, SimpleTransformer)

def test_simple_transformer_forward():
    model = SimpleTransformer()
    input = torch.randint(0, VOCAB_SIZE, (1, 10))  # Batch size 1, sequence length 10
    output = model(input)
    assert output.shape == (1, VOCAB_SIZE)