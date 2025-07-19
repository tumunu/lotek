"""
LoTek Fractal Batching Module

Implements recursive LoTek fractal batching with hierarchical self-similar patterns
for efficient batch processing in cybersecurity neural networks.
"""

import torch
import logging
from typing import Optional, List
from .interfaces import FractalBatchingInterface
from ..error_handling import error_handler, InputValidator, ErrorSeverity

logger = logging.getLogger(__name__)

class FractalBatching(FractalBatchingInterface):
    """
    LoTek fractal batching implementation using hierarchical self-similar patterns.
    
    Implements recursive LoTek fractal subdivision for efficient batch processing
    suitable for cybersecurity applications.
    """
    
    def __init__(self, max_batch_size: int = 32, fractal_factor: int = 4,
                 adaptive_batching: bool = True):
        # Validate inputs
        InputValidator.validate_number(max_batch_size, "max_batch_size", min_value=1, integer_only=True)
        InputValidator.validate_number(fractal_factor, "fractal_factor", min_value=2, max_value=16, integer_only=True)
        
        self.max_batch_size = max_batch_size
        self.fractal_factor = fractal_factor
        self.adaptive_batching = adaptive_batching
        
        # Statistics tracking
        self.total_batches_processed = 0
        self.total_elements_processed = 0
    
    @error_handler(exceptions=Exception, severity=ErrorSeverity.MEDIUM)
    def process_batch(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Process tensor in batches using fractal patterns.
        
        Args:
            X: Input tensor to batch process
            batch_size: Optional batch size override
            
        Returns:
            Processed tensor with fractal batching patterns applied
        """
        InputValidator.validate_tensor(X, "input_tensor")
        
        effective_batch_size = batch_size or self.max_batch_size
        
        # If tensor is already small enough, return as-is
        if X.shape[0] <= effective_batch_size:
            logger.debug(f"Tensor size {X.shape[0]} <= batch_size {effective_batch_size}, no batching needed")
            return X
        
        # Apply fractal recursive batching
        if self.adaptive_batching:
            batched_result = self._adaptive_fractal_batch(X, effective_batch_size)
        else:
            batched_result = self._fixed_fractal_batch(X, effective_batch_size)
        
        # Update statistics
        self.total_batches_processed += 1
        self.total_elements_processed += X.shape[0]
        
        logger.debug(f"Processed batch with shape {X.shape} -> {batched_result.shape}")
        return batched_result
    
    def _adaptive_fractal_batch(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Adaptive fractal batching that adjusts based on tensor characteristics.
        
        Args:
            X: Input tensor
            batch_size: Target batch size
            
        Returns:
            Adaptively batched tensor
        """
        batch_dim, *other_dims = X.shape
        
        # Calculate optimal fractal subdivision
        if batch_dim > batch_size * self.fractal_factor:
            # Use hierarchical fractal subdivision
            num_levels = min(3, int(torch.log2(torch.tensor(batch_dim / batch_size)).item()))
            return self._hierarchical_subdivision(X, num_levels, batch_size)
        else:
            # Simple fractal batching
            return self._simple_fractal_batch(X, batch_size)
    
    def _fixed_fractal_batch(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Fixed fractal batching with consistent subdivision patterns.
        
        Args:
            X: Input tensor
            batch_size: Target batch size
            
        Returns:
            Fractal batched tensor
        """
        return self._simple_fractal_batch(X, batch_size)
    
    def _simple_fractal_batch(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Simple fractal batching using self-similar patterns.
        
        Args:
            X: Input tensor
            batch_size: Target batch size
            
        Returns:
            Batched tensor
        """
        batch_dim = X.shape[0]
        
        # Calculate number of complete batches
        num_batches = (batch_dim + batch_size - 1) // batch_size
        
        # Create fractal pattern by repeating self-similar structures
        batched_chunks = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, batch_dim)
            chunk = X[start_idx:end_idx]
            
            # Apply fractal self-similarity if chunk is too small
            if chunk.shape[0] < batch_size and chunk.shape[0] > 0:
                # Pad with fractal repetition
                repeat_factor = batch_size // chunk.shape[0]
                remainder = batch_size % chunk.shape[0]
                
                if repeat_factor > 0:
                    repeated_chunk = chunk.repeat(repeat_factor, *([1] * (len(chunk.shape) - 1)))
                    if remainder > 0:
                        partial_chunk = chunk[:remainder]
                        chunk = torch.cat([repeated_chunk, partial_chunk], dim=0)
                    else:
                        chunk = repeated_chunk
            
            batched_chunks.append(chunk)
        
        # Combine all chunks
        if len(batched_chunks) == 1:
            return batched_chunks[0]
        else:
            # Ensure all chunks have the same size for concatenation
            max_size = max(chunk.shape[0] for chunk in batched_chunks)
            normalized_chunks = []
            
            for chunk in batched_chunks:
                if chunk.shape[0] < max_size:
                    # Pad with zeros or repeat pattern
                    padding_size = max_size - chunk.shape[0]
                    if chunk.shape[0] > 0:
                        # Use fractal repetition for padding
                        repeat_indices = torch.arange(chunk.shape[0]).repeat(
                            (padding_size + chunk.shape[0] - 1) // chunk.shape[0]
                        )[:padding_size]
                        padding = chunk[repeat_indices]
                        chunk = torch.cat([chunk, padding], dim=0)
                    else:
                        # Zero padding as fallback
                        padding_shape = (padding_size,) + chunk.shape[1:]
                        padding = torch.zeros(padding_shape, dtype=chunk.dtype, device=chunk.device)
                        chunk = torch.cat([chunk, padding], dim=0)
                
                normalized_chunks.append(chunk)
            
            return torch.stack(normalized_chunks, dim=0).view(-1, *X.shape[1:])
    
    def _hierarchical_subdivision(self, X: torch.Tensor, num_levels: int, batch_size: int) -> torch.Tensor:
        """
        Hierarchical fractal subdivision for large tensors.
        
        Args:
            X: Input tensor
            num_levels: Number of hierarchical levels
            batch_size: Target batch size
            
        Returns:
            Hierarchically subdivided tensor
        """
        if num_levels == 0 or X.shape[0] <= batch_size:
            return X
        
        # Split into fractal_factor chunks
        chunk_size = X.shape[0] // self.fractal_factor
        chunks = []
        
        for i in range(self.fractal_factor):
            start_idx = i * chunk_size
            if i == self.fractal_factor - 1:
                # Last chunk gets remainder
                end_idx = X.shape[0]
            else:
                end_idx = (i + 1) * chunk_size
            
            chunk = X[start_idx:end_idx]
            
            # Recursively apply fractal subdivision
            processed_chunk = self._hierarchical_subdivision(chunk, num_levels - 1, batch_size)
            chunks.append(processed_chunk)
        
        # Combine processed chunks
        return torch.cat(chunks, dim=0)
    
    def get_batching_stats(self) -> dict:
        """Get batching statistics."""
        avg_elements_per_batch = (
            self.total_elements_processed / max(1, self.total_batches_processed)
        )
        
        return {
            'max_batch_size': self.max_batch_size,
            'fractal_factor': self.fractal_factor,
            'adaptive_batching': self.adaptive_batching,
            'total_batches_processed': self.total_batches_processed,
            'total_elements_processed': self.total_elements_processed,
            'avg_elements_per_batch': avg_elements_per_batch
        }

def recursive_batching(tokens, k=4):
    """Legacy function - use FractalBatching class instead."""
    logger.warning("recursive_batching is deprecated, use FractalBatching class")
    if len(tokens) <= k:
        return [tokens]
    size = len(tokens) // k
    return [recursive_batching(tokens[i:i+size], k) for i in range(0, len(tokens), size)]

def process_batch(batch, operation):
    """Legacy function - use FractalBatching class instead."""
    logger.warning("process_batch is deprecated, use FractalBatching class")
    return [operation(sub_batch) for sub_batch in batch]