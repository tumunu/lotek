"""
LoTek Fractal Search Module

Implements hierarchical similarity search using LoTek fractal patterns for
efficient cybersecurity pattern recognition and threat detection.
"""

import torch
import torch.nn.functional as F
import logging
import math
from typing import List, Tuple, Optional, Callable
from .interfaces import FractalSearchInterface
from ..error_handling import error_handler, InputValidator, ErrorSeverity

logger = logging.getLogger(__name__)

class FractalSearch(FractalSearchInterface):
    """
    LoTek fractal search implementation using hierarchical similarity patterns.
    
    Implements efficient similarity search using LoTek fractal geometry principles
    for cybersecurity pattern recognition.
    """
    
    def __init__(self, embedding_dim: int, k: int = 5, distance_metric: str = "cosine",
                 use_hierarchical_search: bool = True, cache_size: int = 1000):
        # Validate inputs
        InputValidator.validate_number(embedding_dim, "embedding_dim", min_value=1, integer_only=True)
        InputValidator.validate_number(k, "k", min_value=1, max_value=100, integer_only=True)
        InputValidator.validate_number(cache_size, "cache_size", min_value=10, integer_only=True)
        
        self.embedding_dim = embedding_dim
        self.default_k = k
        self.distance_metric = distance_metric
        self.use_hierarchical_search = use_hierarchical_search
        self.cache_size = cache_size
        
        # Distance metric functions
        self.distance_functions = {
            "euclidean": self._euclidean_distance,
            "manhattan": self._manhattan_distance,
            "cosine": self._cosine_distance,
            "fractal": self._fractal_distance
        }
        
        if distance_metric not in self.distance_functions:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Fractal search parameters
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.fractal_dimension = 1.2
        
        # Search cache for performance
        self.search_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.total_searches = 0
        self.total_comparisons = 0
    
    @error_handler(exceptions=Exception, severity=ErrorSeverity.MEDIUM)
    def search(self, data: List[torch.Tensor], query: torch.Tensor, 
               k: int = 5) -> List[Tuple[torch.Tensor, float]]:
        """
        Search for similar patterns in data using fractal similarity.
        
        Args:
            data: List of data tensors to search through
            query: Query tensor to find similarities for
            k: Number of top results to return
            
        Returns:
            List of tuples (similar_tensor, similarity_score)
        """
        InputValidator.validate_list(data, "data", min_length=1)
        InputValidator.validate_tensor(query, "query_tensor")
        InputValidator.validate_number(k, "k", min_value=1, max_value=len(data), integer_only=True)
        
        # Use provided k or default
        effective_k = min(k, len(data))
        
        # Check cache first
        cache_key = self._generate_cache_key(query, len(data), effective_k)
        if cache_key in self.search_cache:
            self.cache_hits += 1
            logger.debug("Cache hit for search query")
            return self.search_cache[cache_key]
        
        self.cache_misses += 1
        
        # Perform search
        if self.use_hierarchical_search and len(data) > 50:
            results = self._hierarchical_fractal_search(data, query, effective_k)
        else:
            results = self._linear_fractal_search(data, query, effective_k)
        
        # Update cache
        self._update_cache(cache_key, results)
        
        # Update statistics
        self.total_searches += 1
        self.total_comparisons += len(data)
        
        logger.debug(f"Search completed: found {len(results)} results from {len(data)} candidates")
        return results
    
    def _hierarchical_fractal_search(self, data: List[torch.Tensor], 
                                   query: torch.Tensor, k: int) -> List[Tuple[torch.Tensor, float]]:
        """
        Hierarchical fractal search for large datasets.
        
        Args:
            data: Data tensors to search
            query: Query tensor
            k: Number of results to return
            
        Returns:
            List of (tensor, score) tuples
        """
        # Level 1: Coarse fractal filtering
        coarse_candidates = self._coarse_fractal_filter(data, query, k * 4)
        
        # Level 2: Fine fractal matching
        fine_candidates = self._fine_fractal_matching(coarse_candidates, query, k * 2)
        
        # Level 3: Precise similarity scoring
        final_results = self._precise_similarity_scoring(fine_candidates, query, k)
        
        return final_results
    
    def _linear_fractal_search(self, data: List[torch.Tensor], 
                             query: torch.Tensor, k: int) -> List[Tuple[torch.Tensor, float]]:
        """
        Linear fractal search for smaller datasets.
        
        Args:
            data: Data tensors to search
            query: Query tensor
            k: Number of results to return
            
        Returns:
            List of (tensor, score) tuples
        """
        distance_fn = self.distance_functions[self.distance_metric]
        
        # Calculate distances to all data points
        similarities = []
        for i, data_tensor in enumerate(data):
            try:
                distance = distance_fn(query, data_tensor)
                similarity = self._distance_to_similarity(distance)
                similarities.append((data_tensor, similarity, i))
            except Exception as e:
                logger.warning(f"Failed to compute similarity for data[{i}]: {e}")
                continue
        
        # Sort by similarity (higher is better)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return [(tensor, score) for tensor, score, _ in similarities[:k]]
    
    def _coarse_fractal_filter(self, data: List[torch.Tensor], 
                             query: torch.Tensor, num_candidates: int) -> List[torch.Tensor]:
        """
        Coarse filtering using fractal dimensional reduction.
        
        Args:
            data: Data tensors
            query: Query tensor
            num_candidates: Number of candidates to keep
            
        Returns:
            Filtered list of candidate tensors
        """
        # Reduce dimensionality using fractal patterns
        reduced_query = self._fractal_dimension_reduction(query)
        
        distances = []
        for i, data_tensor in enumerate(data):
            try:
                reduced_data = self._fractal_dimension_reduction(data_tensor)
                distance = self._euclidean_distance(reduced_query, reduced_data)
                distances.append((distance, i))
            except Exception:
                continue
        
        # Sort by distance and keep top candidates
        distances.sort(key=lambda x: x[0])
        candidate_indices = [idx for _, idx in distances[:num_candidates]]
        
        return [data[i] for i in candidate_indices]
    
    def _fine_fractal_matching(self, candidates: List[torch.Tensor], 
                             query: torch.Tensor, num_results: int) -> List[torch.Tensor]:
        """
        Fine-grained fractal pattern matching.
        
        Args:
            candidates: Candidate tensors
            query: Query tensor
            num_results: Number of results to keep
            
        Returns:
            Refined list of candidate tensors
        """
        fractal_scores = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Calculate fractal similarity
                fractal_score = self._calculate_fractal_similarity(query, candidate)
                fractal_scores.append((fractal_score, i))
            except Exception:
                continue
        
        # Sort by fractal score
        fractal_scores.sort(key=lambda x: x[0], reverse=True)
        result_indices = [idx for _, idx in fractal_scores[:num_results]]
        
        return [candidates[i] for i in result_indices]
    
    def _precise_similarity_scoring(self, candidates: List[torch.Tensor], 
                                  query: torch.Tensor, k: int) -> List[Tuple[torch.Tensor, float]]:
        """
        Precise similarity scoring for final ranking.
        
        Args:
            candidates: Final candidate tensors
            query: Query tensor
            k: Number of final results
            
        Returns:
            List of (tensor, score) tuples
        """
        distance_fn = self.distance_functions[self.distance_metric]
        
        final_scores = []
        for candidate in candidates:
            try:
                distance = distance_fn(query, candidate)
                similarity = self._distance_to_similarity(distance)
                final_scores.append((candidate, similarity))
            except Exception:
                continue
        
        # Sort by similarity
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores[:k]
    
    def _fractal_dimension_reduction(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce tensor dimensionality using fractal patterns.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Dimension-reduced tensor
        """
        if len(tensor.shape) == 1:
            return tensor
        
        # Flatten and select fractal pattern
        flattened = tensor.view(-1)
        
        # Use golden ratio-based sampling
        num_elements = flattened.shape[0]
        step_size = self.golden_ratio
        
        indices = []
        current = 0.0
        while len(indices) < min(64, num_elements) and current < num_elements:
            indices.append(int(current) % num_elements)
            current += step_size
        
        # Remove duplicates and sort
        indices = list(set(indices))
        indices.sort()
        
        return flattened[indices]
    
    def _calculate_fractal_similarity(self, tensor1: torch.Tensor, 
                                    tensor2: torch.Tensor) -> float:
        """
        Calculate fractal similarity between two tensors.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            Fractal similarity score
        """
        # Ensure tensors have the same shape
        min_size = min(tensor1.numel(), tensor2.numel())
        t1_flat = tensor1.view(-1)[:min_size]
        t2_flat = tensor2.view(-1)[:min_size]
        
        # Calculate self-similarity patterns
        t1_patterns = self._extract_fractal_patterns(t1_flat)
        t2_patterns = self._extract_fractal_patterns(t2_flat)
        
        # Compare pattern similarity
        pattern_similarity = F.cosine_similarity(t1_patterns, t2_patterns, dim=0)
        
        return pattern_similarity.item()
    
    def _extract_fractal_patterns(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract fractal patterns from tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Fractal pattern features
        """
        if tensor.numel() < 4:
            return tensor
        
        # Extract patterns at different scales
        scales = [1, 2, 4]
        patterns = []
        
        for scale in scales:
            if tensor.numel() >= scale:
                # Subsample at current scale
                subsampled = tensor[::scale]
                if subsampled.numel() > 0:
                    patterns.append(torch.mean(subsampled))
                    patterns.append(torch.std(subsampled))
        
        return torch.tensor(patterns)
    
    def _euclidean_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Calculate Euclidean distance."""
        return torch.sqrt(torch.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Calculate Manhattan distance."""
        return torch.sum(torch.abs(x1 - x2))
    
    def _cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Calculate cosine distance."""
        return 1.0 - F.cosine_similarity(x1.view(-1), x2.view(-1), dim=0)
    
    def _fractal_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Calculate fractal-based distance."""
        # Combine Euclidean distance with fractal pattern similarity
        euclidean = self._euclidean_distance(x1, x2)
        fractal_sim = self._calculate_fractal_similarity(x1, x2)
        
        # Weight euclidean distance by inverse fractal similarity
        fractal_weight = 1.0 / (1.0 + fractal_sim)
        return euclidean * fractal_weight
    
    def _distance_to_similarity(self, distance: torch.Tensor) -> float:
        """Convert distance to similarity score."""
        return 1.0 / (1.0 + distance.item())
    
    def _generate_cache_key(self, query: torch.Tensor, data_size: int, k: int) -> str:
        """Generate cache key for search query."""
        query_hash = hash(tuple(query.view(-1).tolist()[:10]))  # Use first 10 elements
        return f"{query_hash}_{data_size}_{k}_{self.distance_metric}"
    
    def _update_cache(self, key: str, results: List[Tuple[torch.Tensor, float]]) -> None:
        """Update search cache with new results."""
        if len(self.search_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[key] = results
    
    def get_search_stats(self) -> dict:
        """Get search statistics."""
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        avg_comparisons = self.total_comparisons / max(1, self.total_searches)
        
        return {
            'embedding_dim': self.embedding_dim,
            'default_k': self.default_k,
            'distance_metric': self.distance_metric,
            'use_hierarchical_search': self.use_hierarchical_search,
            'total_searches': self.total_searches,
            'cache_hit_rate': cache_hit_rate,
            'avg_comparisons_per_search': avg_comparisons,
            'cache_size': len(self.search_cache),
            'golden_ratio': self.golden_ratio,
            'fractal_dimension': self.fractal_dimension
        }

def euclidean_distance(x1, x2):
    """Legacy function - use FractalSearch class instead."""
    logger.warning("euclidean_distance is deprecated, use FractalSearch class")
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))

def manhattan_distance(x1, x2):
    """Legacy function - use FractalSearch class instead."""
    logger.warning("manhattan_distance is deprecated, use FractalSearch class")
    return torch.sum(torch.abs(x1 - x2), dim=-1)

def knn_search(data, query, k=5, metric=euclidean_distance):
    """Legacy function - use FractalSearch class instead."""
    logger.warning("knn_search is deprecated, use FractalSearch class")
    distances = [metric(query, d) for d in data]
    indices = torch.argsort(torch.tensor(distances))[:k]
    return [data[i] for i in indices]