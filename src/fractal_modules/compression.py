"""
LoTek Fractal Compression Module

Implements hierarchical LoTek fractal compression with self-similar patterns
for efficient neural network compression suitable for cybersecurity applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from .interfaces import FractalCompressionInterface
from ..error_handling import error_handler, InputValidator, ErrorSeverity

logger = logging.getLogger(__name__)

class FractalCompression(nn.Module, FractalCompressionInterface):
    """
    LoTek fractal compression module implementing hierarchical self-similar compression patterns.
    
    Based on LoTek fractal geometry principles for efficient neural network compression
    suitable for cybersecurity applications.
    """
    
    def __init__(self, embedding_dim: int, compression_ratio: float = 0.5, 
                 levels: int = 3, similarity_threshold: float = 0.5):
        super(FractalCompression, self).__init__()
        
        # Validate inputs
        InputValidator.validate_number(embedding_dim, "embedding_dim", min_value=1, integer_only=True)
        InputValidator.validate_number(compression_ratio, "compression_ratio", min_value=0.1, max_value=1.0)
        InputValidator.validate_number(levels, "levels", min_value=1, max_value=10, integer_only=True)
        
        self.embedding_dim = embedding_dim
        self.compression_ratio = compression_ratio
        self.levels = levels
        self.similarity_threshold = similarity_threshold
        
        # Calculate compressed dimension based on ratio
        self.compressed_dim = max(1, int(embedding_dim * compression_ratio))
        
        # Build compression layers with fractal hierarchy
        self.compress_layers = nn.ModuleList()
        self.decompress_layers = nn.ModuleList()
        
        # Compression path
        current_dim = embedding_dim
        for i in range(levels):
            next_dim = max(1, int(current_dim * compression_ratio))
            self.compress_layers.append(nn.Sequential(
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ))
            current_dim = next_dim
        
        # Decompression path (reverse)
        for i in range(levels):
            if i == levels - 1:
                next_dim = embedding_dim
            else:
                next_dim = int(current_dim / compression_ratio)
            
            self.decompress_layers.append(nn.Sequential(
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU(inplace=True) if i < levels - 1 else nn.Identity()
            ))
            current_dim = next_dim
        
        # Fractal self-similarity detection
        self.similarity_detector = nn.Sequential(
            nn.Linear(self.compressed_dim, self.compressed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.compressed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @error_handler(exceptions=Exception, severity=ErrorSeverity.MEDIUM)
    def compress(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compress input tensor using fractal hierarchical compression.
        
        Args:
            X: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Compressed tensor of shape (batch_size, seq_len, compressed_dim)
        """
        InputValidator.validate_tensor(X, "input_tensor", expected_dims=3)
        
        compressed = X
        
        # Apply hierarchical compression with fractal patterns
        for i, layer in enumerate(self.compress_layers):
            compressed = layer(compressed)
            
            # Apply fractal self-similarity detection at intermediate levels
            if i > 0 and i < len(self.compress_layers) - 1:
                similarity_scores = self.similarity_detector(compressed)
                
                # Use similarity scores to modulate compression
                # Higher similarity = more aggressive compression
                similarity_mask = similarity_scores > self.similarity_threshold
                compressed = compressed * (1.0 + 0.1 * similarity_scores)
        
        logger.debug(f"Compressed tensor from {X.shape} to {compressed.shape}")
        return compressed
    
    @error_handler(exceptions=Exception, severity=ErrorSeverity.MEDIUM)
    def decompress(self, X: torch.Tensor) -> torch.Tensor:
        """
        Decompress tensor back to original dimensions.
        
        Args:
            X: Compressed tensor
            
        Returns:
            Decompressed tensor of original embedding dimension
        """
        InputValidator.validate_tensor(X, "compressed_tensor", expected_dims=3)
        
        decompressed = X
        
        # Apply hierarchical decompression
        for layer in self.decompress_layers:
            decompressed = layer(decompressed)
        
        logger.debug(f"Decompressed tensor from {X.shape} to {decompressed.shape}")
        return decompressed
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compress then decompress (autoencoder behavior).
        
        Args:
            X: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        compressed = self.compress(X)
        reconstructed = self.decompress(compressed)
        return reconstructed
    
    def get_compression_ratio(self) -> float:
        """Get the actual compression ratio achieved."""
        original_params = self.embedding_dim
        compressed_params = self.compressed_dim
        return compressed_params / original_params
    
    def get_compression_stats(self) -> dict:
        """Get detailed compression statistics."""
        return {
            'original_dim': self.embedding_dim,
            'compressed_dim': self.compressed_dim,
            'compression_ratio': self.get_compression_ratio(),
            'levels': self.levels,
            'similarity_threshold': self.similarity_threshold,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }