"""
LoTek Fractal Network Model Module

Contains the core model classes for the LoTek fractal cybersecurity neural network,
including fractal attention mechanisms and the unified network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VOCAB_SIZE, EMBEDDING_DIM, NUM_TOKENS, NUM_CLUSTERS
from src.fractal_modules.unification import FractalUnifyingModel

class FractalAttention(nn.Module):
    """LoTek Fractal Attention mechanism with clustering and distance-based weighting."""
    
    def __init__(self, embedding_dim, num_tokens, num_clusters=NUM_CLUSTERS, gamma=1.0, scale_factor=1.0, tau=0.1):
        super(FractalAttention, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, embedding_dim))
        self.gamma = nn.Parameter(torch.tensor([gamma]), requires_grad=True)
        self.scale_factor = scale_factor
        self.tau = tau

    def forward(self, X):
        batch_size, seq_length = X.shape
        X = self.embedding(X)
        cluster_assignments = torch.argmin(torch.cdist(X, self.cluster_centers.expand(batch_size, -1, -1)), dim=-1)
        
        distances = torch.cdist(X, X)
        attention_weights = torch.exp(-self.gamma * distances / self.scale_factor)
        sparse_mask = (distances > self.tau).float()
        attention_weights = attention_weights * (1 - sparse_mask)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        return torch.matmul(attention_weights, X), attention_weights

class FractalNetwork(nn.Module):
    """LoTek Fractal Network wrapper class for the unified model architecture."""
    
    def __init__(self):
        super(FractalNetwork, self).__init__()
        self.model = FractalUnifyingModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

    def forward(self, X):
        return self.model(X)

    def get_attention_and_embeddings(self, X):
        return self.model.get_attention_and_embeddings(X)