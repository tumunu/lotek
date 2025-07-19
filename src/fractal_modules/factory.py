"""
LoTek Fractal Module Factory Implementation

Implements the factory pattern for creating LoTek fractal network components
with dependency injection and configuration management.
"""

from typing import Any, Dict
from .interfaces import (
    FractalModuleFactory, FractalAttentionInterface, FractalCompressionInterface,
    FractalMemoryInterface, FractalInferenceInterface, FractalBatchingInterface,
    FractalScalingInterface, FractalSearchInterface
)
from .attention import FractalAttention
from .memory import FractalMemory

class DefaultFractalFactory(FractalModuleFactory):
    """Default implementation of LoTek fractal module factory with dependency injection."""
    
    def create_attention(self, config: Dict[str, Any]) -> FractalAttentionInterface:
        """Create LoTek fractal attention module with configuration."""
        return FractalAttention(
            embedding_dim=config.get('embedding_dim', 768),
            vocab_size=config.get('vocab_size', 50257),
            num_clusters=config.get('num_clusters', 4),
            fractal_scaling_factor=config.get('fractal_scaling_factor', 1.0),
            dropout=config.get('dropout', 0.1)
        )
    
    def create_compression(self, config: Dict[str, Any]) -> FractalCompressionInterface:
        """Create fractal compression module."""
        # Import here to avoid circular dependencies
        from .compression import FractalCompression
        return FractalCompression(
            embedding_dim=config.get('embedding_dim', 768),
            compression_ratio=config.get('compression_ratio', 0.5)
        )
    
    def create_memory(self, config: Dict[str, Any]) -> FractalMemoryInterface:
        """Create fractal memory module."""
        return FractalMemory(
            max_size=config.get('max_size', 1000),
            path=config.get('path', './memory'),
            max_age_seconds=config.get('max_age_seconds', 3600.0),
            cleanup_interval=config.get('cleanup_interval', 300.0)
        )
    
    def create_inference(self, config: Dict[str, Any]) -> FractalInferenceInterface:
        """Create fractal inference module."""
        from .inference import FractalInference
        return FractalInference(
            vocab_size=config.get('vocab_size', 50257),
            embedding_dim=config.get('embedding_dim', 768)
        )
    
    def create_batching(self, config: Dict[str, Any]) -> FractalBatchingInterface:
        """Create fractal batching module."""
        from .batching import FractalBatching
        return FractalBatching(
            max_batch_size=config.get('max_batch_size', 32),
            fractal_factor=config.get('fractal_factor', 4),
            adaptive_batching=config.get('adaptive_batching', True)
        )
    
    def create_scaling(self, config: Dict[str, Any]) -> FractalScalingInterface:
        """Create fractal scaling module."""
        from .scaling import FractalScaling
        return FractalScaling(
            scale_factor=config.get('scale_factor', 1.0),
            min_scale=config.get('min_scale', 0.1),
            max_scale=config.get('max_scale', 10.0),
            adaptive_scaling=config.get('adaptive_scaling', True)
        )
    
    def create_search(self, config: Dict[str, Any]) -> FractalSearchInterface:
        """Create fractal search module."""
        from .search import FractalSearch
        return FractalSearch(
            embedding_dim=config.get('embedding_dim', 768),
            k=config.get('k', 5),
            distance_metric=config.get('distance_metric', 'cosine'),
            use_hierarchical_search=config.get('use_hierarchical_search', True),
            cache_size=config.get('cache_size', 1000)
        )