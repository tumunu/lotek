"""
LoTek Fractal Inference Module

Implements hierarchical LoTek fractal inference patterns for next token prediction
and probability distribution generation in cybersecurity applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from .interfaces import FractalInferenceInterface
from ..error_handling import error_handler, InputValidator, ErrorSeverity

logger = logging.getLogger(__name__)

class FractalInference(nn.Module, FractalInferenceInterface):
    """
    LoTek fractal inference module for next token prediction and probability distribution generation.
    
    Implements hierarchical LoTek fractal patterns for efficient inference suitable for 
    cybersecurity applications.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 use_compression: bool = True, dropout: float = 0.1):
        super(FractalInference, self).__init__()
        
        # Validate inputs
        InputValidator.validate_number(vocab_size, "vocab_size", min_value=1, integer_only=True)
        InputValidator.validate_number(embedding_dim, "embedding_dim", min_value=1, integer_only=True)
        InputValidator.validate_number(dropout, "dropout", min_value=0.0, max_value=1.0)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_compression = use_compression
        
        # Optional compression for efficiency
        if use_compression:
            # Import here to avoid circular imports
            from .compression import FractalCompression
            self.compression = FractalCompression(
                embedding_dim=embedding_dim,
                compression_ratio=0.5
            )
            # Use compressed dimension for final projection
            final_dim = self.compression.compressed_dim
        else:
            self.compression = None
            final_dim = embedding_dim
        
        # Fractal hierarchical projection layers
        self.hierarchical_layers = nn.ModuleList([
            # Level 1: Local patterns
            nn.Sequential(
                nn.Linear(final_dim, final_dim),
                nn.LayerNorm(final_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Level 2: Regional patterns  
            nn.Sequential(
                nn.Linear(final_dim, final_dim // 2),
                nn.LayerNorm(final_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # Level 3: Global patterns
            nn.Sequential(
                nn.Linear(final_dim // 2, final_dim // 4),
                nn.LayerNorm(final_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # Final projection to vocabulary
        self.output_projection = nn.Linear(final_dim // 4, vocab_size)
        
        # Temperature scaling for calibrated probabilities
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @error_handler(exceptions=Exception, severity=ErrorSeverity.HIGH)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward inference pass to generate probability distributions.
        
        Args:
            X: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Probability distributions over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        InputValidator.validate_tensor(X, "input_tensor", expected_dims=3)
        
        # Optional compression for efficiency
        if self.use_compression and self.compression is not None:
            X = self.compression.compress(X)
        
        # Apply hierarchical fractal processing
        processed = X
        for i, layer in enumerate(self.hierarchical_layers):
            processed = layer(processed)
            logger.debug(f"Hierarchical layer {i+1} output shape: {processed.shape}")
        
        # Project to vocabulary space
        logits = self.output_projection(processed)
        
        # Apply temperature scaling for calibrated probabilities
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # Generate probability distributions
        probabilities = F.softmax(scaled_logits, dim=-1)
        
        logger.debug(f"Generated probabilities shape: {probabilities.shape}")
        return probabilities
    
    def get_logits(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits without softmax for loss computation.
        
        Args:
            X: Input tensor
            
        Returns:
            Raw logits
        """
        InputValidator.validate_tensor(X, "input_tensor", expected_dims=3)
        
        if self.use_compression and self.compression is not None:
            X = self.compression.compress(X)
        
        processed = X
        for layer in self.hierarchical_layers:
            processed = layer(processed)
        
        logits = self.output_projection(processed)
        return logits / torch.clamp(self.temperature, min=0.1, max=10.0)
    
    def predict_next_token(self, X: torch.Tensor, top_k: int = 5) -> tuple:
        """
        Predict next token with top-k sampling.
        
        Args:
            X: Input tensor
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (top_k_tokens, top_k_probabilities)
        """
        with torch.no_grad():
            probabilities = self.forward(X)
            
            # Get last token predictions
            last_token_probs = probabilities[:, -1, :]  # (batch_size, vocab_size)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(last_token_probs, top_k, dim=-1)
            
            return top_k_indices, top_k_probs
    
    def sample_token(self, X: torch.Tensor, temperature: float = 1.0, 
                    top_p: float = 0.9) -> torch.Tensor:
        """
        Sample next token using nucleus (top-p) sampling.
        
        Args:
            X: Input tensor
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Sampled token indices
        """
        with torch.no_grad():
            logits = self.get_logits(X)
            last_logits = logits[:, -1, :] / temperature
            
            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # Set logits to -inf for removed tokens
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            last_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(last_logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, num_samples=1)
            
            return sampled_tokens
    
    def get_inference_stats(self) -> dict:
        """Get inference module statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        
        stats = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'use_compression': self.use_compression,
            'total_parameters': total_params,
            'temperature': self.temperature.item(),
            'hierarchical_levels': len(self.hierarchical_layers)
        }
        
        if self.compression is not None:
            stats.update(self.compression.get_compression_stats())
        
        return stats