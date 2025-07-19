"""
LoTek Fractal Unification Module

Orchestrates all LoTek fractal components into a unified cybersecurity neural network with
dependency injection, performance monitoring, and modular architecture.
"""

import torch
from torch import nn
from typing import Dict, Any, Optional
import logging

from .interfaces import (
    FractalAttentionInterface, FractalCompressionInterface, FractalMemoryInterface,
    FractalInferenceInterface, FractalBatchingInterface, FractalScalingInterface,
    FractalSearchInterface, FractalModuleFactory
)
from .factory import DefaultFractalFactory

logger = logging.getLogger(__name__)

class FractalUnifyingModel(nn.Module):
    """
    Unified LoTek fractal cybersecurity neural network with dependency injection and modular architecture.
    
    This implementation follows the LoTek fractal network principles with:
    - Hierarchical self-similar patterns
    - Efficient attention mechanisms optimized for cybersecurity tasks
    - Modular components for extensibility
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 config: Optional[Dict[str, Any]] = None,
                 factory: Optional[FractalModuleFactory] = None):
        super(FractalUnifyingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Default configuration
        if config is None:
            config = self._get_default_config()
        self.config = config
        
        # Use provided factory or default
        if factory is None:
            factory = DefaultFractalFactory()
        self.factory = factory
        
        # Initialize fractal modules with dependency injection
        self._initialize_modules()
        
        # Performance monitoring
        self._forward_count = 0
        self._total_inference_time = 0.0
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for LoTek fractal modules."""
        return {
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'num_clusters': 4,
            'fractal_scaling_factor': 1.0,
            'dropout': 0.1,
            'compression_ratio': 0.5,
            'max_size': 1000,
            'path': './memory',
            'max_age_seconds': 3600.0,
            'cleanup_interval': 300.0,
            'max_batch_size': 32,
            'scale_factor': 1.0,
            'k': 5
        }
    
    def _initialize_modules(self) -> None:
        """Initialize all LoTek fractal modules using the factory."""
        try:
            self.attention = self.factory.create_attention(self.config)
            self.compression = self.factory.create_compression(self.config)
            self.inference = self.factory.create_inference(self.config)
            self.memory = self.factory.create_memory(self.config)
            self.batching = self.factory.create_batching(self.config)
            self.scaling = self.factory.create_scaling(self.config)
            self.search = self.factory.create_search(self.config)
            
            logger.info("LoTek fractal modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize fractal modules: {e}")
            raise
    
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the LoTek fractal unifying model.
        
        Args:
            X: Input tensor of token indices, shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Output probabilities for next token prediction
        """
        import time
        start_time = time.time()
        
        try:
            # Process through batching module for fractal batch patterns
            batched_X = self.batching.process_batch(X)
            
            # Apply fractal attention with hierarchical clustering
            processed_X, attention_weights = self.attention.forward(batched_X, mask)
            
            # Apply fractal compression for efficiency
            compressed_X = self.compression.compress(processed_X)
            
            # Generate probability distributions
            probabilities = self.inference.forward(compressed_X)
            
            # Memory management with fractal patterns
            self._update_memory(processed_X, attention_weights)
            
            # Search for similar patterns in memory
            self._search_similar_patterns(processed_X)
            
            # Update performance metrics
            self._forward_count += 1
            self._total_inference_time += time.time() - start_time
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise
    
    def _update_memory(self, processed_X: torch.Tensor, attention_weights: torch.Tensor) -> None:
        """Update fractal memory with processed data and attention patterns."""
        try:
            # Store current batch with timestamp
            batch_key = f"batch_{self._forward_count}"
            self.memory.store(batch_key, {
                'data': processed_X.detach(),
                'attention': attention_weights.detach(),
                'timestamp': torch.tensor(time.time())
            })
            
            # Store most recent as 'last_batch' for quick access
            self.memory.store('last_batch', processed_X.detach())
            
        except Exception as e:
            logger.warning(f"Memory update failed: {e}")
    
    def _search_similar_patterns(self, query: torch.Tensor) -> Optional[list]:
        """Search for similar patterns in memory using fractal search."""
        try:
            # Retrieve past data for similarity search
            past_batches = []
            for i in range(max(0, self._forward_count - 10), self._forward_count):
                batch_data = self.memory.retrieve(f"batch_{i}")
                if batch_data and isinstance(batch_data, dict) and 'data' in batch_data:
                    past_batches.append(batch_data['data'])
            
            if past_batches:
                similar_patterns = self.search.search(past_batches, query)
                logger.debug(f"Found {len(similar_patterns)} similar patterns")
                return similar_patterns
            
        except Exception as e:
            logger.warning(f"Pattern search failed: {e}")
        
        return None
    
    def get_attention_and_embeddings(self, X: torch.Tensor) -> tuple:
        """Get attention weights and embeddings for analysis."""
        with torch.no_grad():
            processed_X, attention_weights = self.attention.forward(X)
            embeddings = processed_X
            return attention_weights, embeddings
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for monitoring."""
        avg_inference_time = (
            self._total_inference_time / max(1, self._forward_count)
        )
        
        memory_stats = self.memory.get_stats() if hasattr(self.memory, 'get_stats') else {}
        
        return {
            'forward_count': self._forward_count,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': self._total_inference_time,
            **memory_stats
        }
    
    def cleanup_memory(self) -> None:
        """Manually trigger memory cleanup."""
        if hasattr(self.memory, 'cleanup'):
            self.memory.cleanup()
        logger.info("Memory cleanup completed")