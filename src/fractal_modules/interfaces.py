"""
LoTek Fractal Module Interfaces for Dependency Injection

Defines abstract interfaces for all LoTek fractal network components to enable
modular architecture and dependency injection patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn

class FractalAttentionInterface(ABC):
    """Interface for LoTek fractal attention mechanisms."""
    
    @abstractmethod
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through LoTek fractal attention.
        
        Args:
            X: Input tensor
            
        Returns:
            Tuple of (processed_tensor, attention_weights)
        """
        pass

class FractalCompressionInterface(ABC):
    """Interface for LoTek fractal compression mechanisms."""
    
    @abstractmethod
    def compress(self, X: torch.Tensor) -> torch.Tensor:
        """Compress input using fractal patterns."""
        pass
    
    @abstractmethod
    def decompress(self, X: torch.Tensor) -> torch.Tensor:
        """Decompress fractal-compressed input."""
        pass

class FractalMemoryInterface(ABC):
    """Interface for fractal memory management."""
    
    @abstractmethod
    def store(self, key: str, data: Any) -> None:
        """Store data with given key."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key."""
        pass
    
    @abstractmethod
    def cleanup(self, max_age_seconds: float = 3600.0) -> None:
        """Clean up old entries."""
        pass

class FractalInferenceInterface(ABC):
    """Interface for fractal inference mechanisms."""
    
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform inference on input tensor."""
        pass

class FractalBatchingInterface(ABC):
    """Interface for fractal batching strategies."""
    
    @abstractmethod
    def process_batch(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Process tensor in batches using fractal patterns."""
        pass

class FractalScalingInterface(ABC):
    """Interface for fractal scaling mechanisms."""
    
    @abstractmethod
    def scale(self, X: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
        """Scale tensor using fractal scaling strategies."""
        pass

class FractalSearchInterface(ABC):
    """Interface for fractal search mechanisms."""
    
    @abstractmethod
    def search(self, data: List[torch.Tensor], query: torch.Tensor, k: int = 5) -> List[Tuple[torch.Tensor, float]]:
        """Search for similar patterns in data."""
        pass

class FractalModuleFactory(ABC):
    """Factory interface for creating fractal modules."""
    
    @abstractmethod
    def create_attention(self, config: Dict[str, Any]) -> FractalAttentionInterface:
        """Create fractal attention module."""
        pass
    
    @abstractmethod
    def create_compression(self, config: Dict[str, Any]) -> FractalCompressionInterface:
        """Create fractal compression module."""
        pass
    
    @abstractmethod
    def create_memory(self, config: Dict[str, Any]) -> FractalMemoryInterface:
        """Create fractal memory module."""
        pass
    
    @abstractmethod
    def create_inference(self, config: Dict[str, Any]) -> FractalInferenceInterface:
        """Create fractal inference module."""
        pass
    
    @abstractmethod
    def create_batching(self, config: Dict[str, Any]) -> FractalBatchingInterface:
        """Create fractal batching module."""
        pass
    
    @abstractmethod
    def create_scaling(self, config: Dict[str, Any]) -> FractalScalingInterface:
        """Create fractal scaling module."""
        pass
    
    @abstractmethod
    def create_search(self, config: Dict[str, Any]) -> FractalSearchInterface:
        """Create fractal search module."""
        pass