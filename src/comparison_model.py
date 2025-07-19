"""
Comparison Model Module for LoTek Fractal Network

Provides baseline transformer models for comparison against the LoTek fractal
cybersecurity neural network architecture in performance evaluations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VOCAB_SIZE, EMBEDDING_DIM

class SimpleTransformer(nn.Module):
    """Simple transformer baseline for comparison with LoTek fractal network."""
    
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.transformer = nn.TransformerEncoderLayer(d_model=EMBEDDING_DIM, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=6)
        self.fc = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(EMBEDDING_DIM)
        src = self.transformer_encoder(src)
        output = self.fc(src[:, -1, :])
        return F.log_softmax(output, dim=-1)