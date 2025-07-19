# test_visualize.py - Pytest for visualize

import pytest
import torch
from src.visualize import visualize_attention, visualize_embeddings
from config import EMBEDDING_DIM

def test_visualize_attention():
    attention_weights = torch.randn(1, 10, 10)  # Batch size 1, sequence length 10x10
    visualize_attention(attention_weights, save_path='test_attention.png')

def test_visualize_embeddings():
    embeddings = torch.randn(10, EMBEDDING_DIM)  # 10 tokens with EMBEDDING_DIM dimension
    visualize_embeddings(embeddings, save_path='test_embeddings.png')