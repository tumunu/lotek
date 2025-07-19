"""
LoTek Fractal Scaling Module

Implements adaptive LoTek fractal scaling using golden ratio harmonics and 
self-similar scaling patterns for cybersecurity applications.
"""

import torch
import logging
import math
from typing import Optional, Union
from .interfaces import FractalScalingInterface
from ..error_handling import error_handler, InputValidator, ErrorSeverity

logger = logging.getLogger(__name__)

class FractalScaling(FractalScalingInterface):
    """
    LoTek fractal scaling implementation using self-similar scaling patterns.
    
    Implements adaptive scaling strategies based on LoTek fractal geometry principles
    for efficient processing in cybersecurity applications.
    """
    
    def __init__(self, scale_factor: float = 1.0, min_scale: float = 0.1, 
                 max_scale: float = 10.0, adaptive_scaling: bool = True):
        # Validate inputs
        InputValidator.validate_number(scale_factor, "scale_factor", min_value=0.1, max_value=10.0)
        InputValidator.validate_number(min_scale, "min_scale", min_value=0.01, max_value=1.0)
        InputValidator.validate_number(max_scale, "max_scale", min_value=1.0, max_value=100.0)
        
        self.base_scale_factor = scale_factor
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.adaptive_scaling = adaptive_scaling
        
        # Fractal scaling parameters
        self.golden_ratio = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
        self.fractal_dimension = 1.5  # Default fractal dimension
        
        # Statistics tracking
        self.total_scalings = 0
        self.scale_history = []
    
    @error_handler(exceptions=Exception, severity=ErrorSeverity.MEDIUM)
    def scale(self, X: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
        """
        Scale tensor using fractal scaling strategies.
        
        Args:
            X: Input tensor to scale
            scale_factor: Scaling factor (1.0 = no scaling)
            
        Returns:
            Scaled tensor
        """
        InputValidator.validate_tensor(X, "input_tensor")
        InputValidator.validate_number(scale_factor, "scale_factor", min_value=0.1, max_value=10.0)
        
        effective_scale = scale_factor if scale_factor != 1.0 else self.base_scale_factor
        
        # Clamp scale factor to valid range
        effective_scale = max(self.min_scale, min(self.max_scale, effective_scale))
        
        if self.adaptive_scaling:
            scaled_tensor = self._adaptive_fractal_scale(X, effective_scale)
        else:
            scaled_tensor = self._uniform_fractal_scale(X, effective_scale)
        
        # Update statistics
        self.total_scalings += 1
        self.scale_history.append(effective_scale)
        if len(self.scale_history) > 100:  # Keep only recent history
            self.scale_history = self.scale_history[-100:]
        
        logger.debug(f"Scaled tensor with factor {effective_scale}: {X.shape} -> {scaled_tensor.shape}")
        return scaled_tensor
    
    def _adaptive_fractal_scale(self, X: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Adaptive fractal scaling based on tensor characteristics.
        
        Args:
            X: Input tensor
            scale_factor: Base scaling factor
            
        Returns:
            Adaptively scaled tensor
        """
        # Analyze tensor characteristics for adaptive scaling
        tensor_variance = torch.var(X).item()
        tensor_mean = torch.mean(X).item()
        tensor_size = X.numel()
        
        # Calculate adaptive scale factor using fractal patterns
        if tensor_variance > 1.0:
            # High variance - use more conservative scaling
            adaptive_factor = scale_factor / self.golden_ratio
        elif tensor_variance < 0.1:
            # Low variance - can scale more aggressively
            adaptive_factor = scale_factor * self.golden_ratio
        else:
            # Moderate variance - use base scaling
            adaptive_factor = scale_factor
        
        # Apply size-based fractal adjustment
        size_factor = math.log(tensor_size + 1) / math.log(10000)  # Normalize to [0, 1] range
        fractal_adjustment = math.pow(size_factor, 1 / self.fractal_dimension)
        
        final_scale = adaptive_factor * (1 + 0.1 * fractal_adjustment)
        final_scale = max(self.min_scale, min(self.max_scale, final_scale))
        
        return self._apply_hierarchical_scaling(X, final_scale)
    
    def _uniform_fractal_scale(self, X: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Uniform fractal scaling with consistent patterns.
        
        Args:
            X: Input tensor
            scale_factor: Scaling factor
            
        Returns:
            Uniformly scaled tensor
        """
        return self._apply_hierarchical_scaling(X, scale_factor)
    
    def _apply_hierarchical_scaling(self, X: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Apply hierarchical fractal scaling patterns.
        
        Args:
            X: Input tensor
            scale_factor: Scaling factor
            
        Returns:
            Hierarchically scaled tensor
        """
        if abs(scale_factor - 1.0) < 1e-6:
            return X  # No scaling needed
        
        # Apply different scaling strategies based on scale factor
        if scale_factor > 1.0:
            # Upscaling using fractal interpolation
            return self._fractal_upscale(X, scale_factor)
        else:
            # Downscaling using fractal decimation
            return self._fractal_downscale(X, scale_factor)
    
    def _fractal_upscale(self, X: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Upscale tensor using fractal interpolation patterns.
        
        Args:
            X: Input tensor
            scale_factor: Upscaling factor (> 1.0)
            
        Returns:
            Upscaled tensor
        """
        if len(X.shape) == 1:
            # 1D tensor - simple interpolation
            return torch.nn.functional.interpolate(
                X.unsqueeze(0).unsqueeze(0), 
                scale_factor=scale_factor, 
                mode='linear'
            ).squeeze()
        elif len(X.shape) == 2:
            # 2D tensor - apply scaling to both dimensions
            new_size = (int(X.shape[0] * scale_factor), int(X.shape[1] * scale_factor))
            return torch.nn.functional.interpolate(
                X.unsqueeze(0).unsqueeze(0), 
                size=new_size, 
                mode='bilinear'
            ).squeeze()
        else:
            # Multi-dimensional tensor - apply fractal self-similarity
            return self._fractal_repeat_pattern(X, scale_factor)
    
    def _fractal_downscale(self, X: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Downscale tensor using fractal decimation patterns.
        
        Args:
            X: Input tensor
            scale_factor: Downscaling factor (< 1.0)
            
        Returns:
            Downscaled tensor
        """
        if len(X.shape) <= 2:
            # Simple downsampling for 1D/2D tensors
            downsample_factor = int(1 / scale_factor)
            if len(X.shape) == 1:
                return X[::downsample_factor]
            else:
                return X[::downsample_factor, ::downsample_factor]
        else:
            # Multi-dimensional tensor - use fractal decimation
            return self._fractal_decimate_pattern(X, scale_factor)
    
    def _fractal_repeat_pattern(self, X: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Repeat tensor pattern using fractal self-similarity.
        
        Args:
            X: Input tensor
            scale_factor: Scaling factor
            
        Returns:
            Pattern-repeated tensor
        """
        repeat_factor = max(1, int(scale_factor))
        
        # Calculate repeat dimensions
        repeat_dims = [repeat_factor] + [1] * (len(X.shape) - 1)
        
        # Apply fractal repetition
        repeated = X.repeat(*repeat_dims)
        
        # Apply golden ratio-based modulation for fractal characteristics
        modulation = torch.linspace(1.0, 1 / self.golden_ratio, repeat_factor)
        
        for i in range(repeat_factor):
            start_idx = i * X.shape[0]
            end_idx = (i + 1) * X.shape[0]
            repeated[start_idx:end_idx] *= modulation[i]
        
        return repeated
    
    def _fractal_decimate_pattern(self, X: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """
        Decimate tensor pattern using fractal selection.
        
        Args:
            X: Input tensor
            scale_factor: Scaling factor (< 1.0)
            
        Returns:
            Decimated tensor
        """
        decimate_factor = int(1 / scale_factor)
        
        # Use golden ratio-based decimation for fractal characteristics
        indices = []
        step = self.golden_ratio
        current = 0.0
        
        while current < X.shape[0]:
            indices.append(int(current) % X.shape[0])
            current += step
            if len(indices) >= X.shape[0] // decimate_factor:
                break
        
        # Ensure we have unique indices
        indices = list(set(indices))
        indices.sort()
        
        # Apply decimation
        return X[indices]
    
    def get_optimal_scale_factor(self, target_size: int, current_size: int) -> float:
        """
        Calculate optimal scale factor for target size.
        
        Args:
            target_size: Desired tensor size
            current_size: Current tensor size
            
        Returns:
            Optimal scale factor
        """
        if current_size == 0:
            return 1.0
        
        raw_scale = target_size / current_size
        
        # Apply fractal adjustment based on golden ratio
        if raw_scale > 1.0:
            # Upscaling - use golden ratio harmonics
            scale_levels = [1.0, self.golden_ratio, self.golden_ratio**2]
            optimal_scale = min(scale_levels, key=lambda x: abs(x - raw_scale))
        else:
            # Downscaling - use inverse golden ratio harmonics
            scale_levels = [1.0, 1/self.golden_ratio, 1/(self.golden_ratio**2)]
            optimal_scale = min(scale_levels, key=lambda x: abs(x - raw_scale))
        
        return max(self.min_scale, min(self.max_scale, optimal_scale))
    
    def get_scaling_stats(self) -> dict:
        """Get scaling statistics."""
        avg_scale = sum(self.scale_history) / max(1, len(self.scale_history))
        
        return {
            'base_scale_factor': self.base_scale_factor,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
            'adaptive_scaling': self.adaptive_scaling,
            'total_scalings': self.total_scalings,
            'avg_scale_factor': avg_scale,
            'golden_ratio': self.golden_ratio,
            'fractal_dimension': self.fractal_dimension
        }

def parallel_process(data, func):
    """Legacy function - use FractalScaling class instead."""
    logger.warning("parallel_process is deprecated, use FractalScaling class")
    import multiprocessing
    with multiprocessing.Pool() as pool:
        return pool.map(func, data)

def adaptive_scaling(data, threshold=1000):
    """Legacy function - use FractalScaling class instead."""
    logger.warning("adaptive_scaling is deprecated, use FractalScaling class")
    if len(data) > threshold:
        return [data[i:i+threshold] for i in range(0, len(data), threshold)]
    return [data]