"""
LoTek Fractal Attention Module

Implements optimized LoTek fractal attention with hierarchical clustering, multi-head
attention, and self-similarity patterns for cybersecurity applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .interfaces import FractalAttentionInterface

class FractalAttention(nn.Module, FractalAttentionInterface):
    """
    Optimized LoTek Fractal Attention mechanism with hierarchical clustering and self-similarity.
    
    Based on LoTek fractal network principles from:
    - FractalNet: Ultra-Deep Neural Networks without Residuals (Larsson et al., 2016)
    - Self-similar attention patterns for improved cybersecurity efficiency
    """
    
    def __init__(self, embedding_dim: int, vocab_size: int = 50257, 
                 num_clusters: int = 4, fractal_scaling_factor: float = 1.0,
                 dropout: float = 0.1, use_flash_attention: bool = True):
        super(FractalAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.fractal_scaling_factor = fractal_scaling_factor
        self.use_flash_attention = use_flash_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Fractal cluster centers for hierarchical attention
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, embedding_dim) / math.sqrt(embedding_dim))
        
        # Multi-head attention components for efficiency
        self.num_heads = max(1, embedding_dim // 64)
        self.head_dim = embedding_dim // self.num_heads
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(self.embedding_dim))
    
    def _fractal_cluster_attention(self, embedded_X: torch.Tensor) -> torch.Tensor:
        """Apply fractal clustering for hierarchical attention patterns."""
        batch_size, seq_len, embed_dim = embedded_X.shape
        
        # Compute distances to cluster centers for fractal patterns
        # Shape: (batch_size, seq_len, num_clusters)
        distances = torch.cdist(
            embedded_X, 
            self.cluster_centers.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        # Soft assignment to clusters with temperature scaling
        cluster_weights = F.softmax(-distances * self.fractal_scaling_factor, dim=-1)
        
        # Apply cluster-based modulation to embeddings
        cluster_features = torch.matmul(cluster_weights, self.cluster_centers.unsqueeze(0).expand(batch_size, -1, -1))
        
        return embedded_X + 0.1 * cluster_features  # Residual connection with fractal features
    
    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized scaled dot-product attention with optional flash attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply fractal scaling to attention scores for self-similar patterns
        scores = scores * self.fractal_scaling_factor
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LoTek fractal attention mechanism.
        
        Args:
            X: Input tensor of token indices, shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        batch_size, seq_len = X.shape
        
        # Convert token indices to embeddings
        embedded_X = self.embedding(X)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Apply fractal clustering for hierarchical patterns
        embedded_X = self._fractal_cluster_attention(embedded_X)
        
        # Reshape for multi-head attention
        q = self.q_proj(embedded_X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(embedded_X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(embedded_X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = self._scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        output = self.out_proj(attn_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(embedded_X + output)
        
        return output, attn_weights.mean(dim=1)  # Average attention weights across heads
