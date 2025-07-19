"""
LoTek Fractal Memory Module

Implements advanced memory management with thread safety, automatic cleanup,
and hierarchical storage patterns for LoTek fractal cybersecurity neural networks.
"""

import torch
import os
import time
import threading
import logging
from typing import Any, Dict, Optional, Set
from pathlib import Path
from .interfaces import FractalMemoryInterface

logger = logging.getLogger(__name__)

class FractalMemory(FractalMemoryInterface):
    """
    Advanced LoTek fractal memory management with automatic cleanup and thread safety.
    
    Features:
    - Automatic cleanup of old entries
    - Thread-safe operations
    - Memory usage tracking
    - Hierarchical storage patterns inspired by LoTek fractal geometry
    """
    
    def __init__(self, max_size: int = 1000, path: str = './memory', 
                 max_age_seconds: float = 3600.0, cleanup_interval: float = 300.0):
        self.max_size = max_size
        self.path = Path(path)
        self.max_age_seconds = max_age_seconds
        self.cleanup_interval = cleanup_interval
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache for frequently accessed items
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp, access_count)
        self._cache_max_size = min(100, max_size // 10)
        
        # Metadata tracking
        self._access_times: Dict[str, float] = {}
        self._file_sizes: Dict[str, int] = {}
        self._total_size = 0
        
        # Initialize storage directory
        self._initialize_storage()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directory and load existing metadata."""
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            
            # Scan existing files and build metadata
            for file_path in self.path.glob("*.pt"):
                try:
                    stat = file_path.stat()
                    key = file_path.stem
                    self._access_times[key] = stat.st_mtime
                    self._file_sizes[key] = stat.st_size
                    self._total_size += stat.st_size
                except (OSError, IOError) as e:
                    logger.warning(f"Failed to read metadata for {file_path}: {e}")
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup thread."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self.cleanup(self.max_age_seconds)
            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")
    
    def _evict_cache(self) -> None:
        """Evict least recently used items from cache."""
        if len(self._cache) <= self._cache_max_size:
            return
        
        # Sort by access count and timestamp
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: (x[1][2], x[1][1])  # (access_count, timestamp)
        )
        
        # Remove least used items
        items_to_remove = len(sorted_items) - self._cache_max_size
        for key, _ in sorted_items[:items_to_remove]:
            del self._cache[key]
    
    def _update_cache(self, key: str, value: Any) -> None:
        """Update in-memory cache with LoTek fractal patterns."""
        current_time = time.time()
        
        if key in self._cache:
            # Update existing entry
            _, _, access_count = self._cache[key]
            self._cache[key] = (value, current_time, access_count + 1)
        else:
            # Add new entry
            self._cache[key] = (value, current_time, 1)
            self._evict_cache()
    
    def store(self, key: str, data: Any) -> None:
        """Store data with automatic size management and caching."""
        if not isinstance(key, str) or not key:
            raise ValueError("Key must be a non-empty string")
        
        with self._lock:
            try:
                file_path = self.path / f"{key}.pt"
                
                # Remove old file if exists
                if file_path.exists():
                    old_size = self._file_sizes.get(key, 0)
                    self._total_size -= old_size
                    file_path.unlink()
                
                # Check if we need to make space
                self._ensure_space()
                
                # Save data
                torch.save(data, file_path)
                
                # Update metadata
                file_size = file_path.stat().st_size
                current_time = time.time()
                
                self._access_times[key] = current_time
                self._file_sizes[key] = file_size
                self._total_size += file_size
                
                # Update cache
                self._update_cache(key, data)
                
                logger.debug(f"Stored {key} ({file_size} bytes)")
                
            except Exception as e:
                logger.error(f"Failed to store {key}: {e}")
                raise
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data with caching and access tracking."""
        if not isinstance(key, str) or not key:
            return None
        
        with self._lock:
            # Check cache first
            if key in self._cache:
                value, _, access_count = self._cache[key]
                self._update_cache(key, value)  # Update access count
                return value
            
            # Load from disk
            file_path = self.path / f"{key}.pt"
            if not file_path.exists():
                return None
            
            try:
                data = torch.load(file_path, weights_only=True)
                
                # Update access time
                self._access_times[key] = time.time()
                
                # Add to cache
                self._update_cache(key, data)
                
                logger.debug(f"Retrieved {key} from disk")
                return data
                
            except Exception as e:
                logger.error(f"Failed to retrieve {key}: {e}")
                return None
    
    def _ensure_space(self) -> None:
        """Ensure there's space for new data by removing old entries."""
        current_files = len(self._access_times)
        
        if current_files < self.max_size:
            return
        
        # Remove oldest files
        sorted_by_age = sorted(self._access_times.items(), key=lambda x: x[1])
        files_to_remove = current_files - self.max_size + 1
        
        for key, _ in sorted_by_age[:files_to_remove]:
            self._remove_file(key)
    
    def _remove_file(self, key: str) -> None:
        """Remove a file and update metadata."""
        try:
            file_path = self.path / f"{key}.pt"
            if file_path.exists():
                file_path.unlink()
            
            # Update metadata
            if key in self._file_sizes:
                self._total_size -= self._file_sizes[key]
                del self._file_sizes[key]
            
            if key in self._access_times:
                del self._access_times[key]
            
            # Remove from cache
            if key in self._cache:
                del self._cache[key]
            
            logger.debug(f"Removed {key}")
            
        except Exception as e:
            logger.error(f"Failed to remove {key}: {e}")
    
    def cleanup(self, max_age_seconds: float = 3600.0) -> None:
        """Clean up old entries beyond specified age."""
        with self._lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, access_time in self._access_times.items():
                if current_time - access_time > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_file(key)
            
            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} old entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            return {
                'total_files': len(self._access_times),
                'total_size_bytes': self._total_size,
                'cache_size': len(self._cache),
                'avg_file_size': self._total_size / max(1, len(self._access_times)),
                'oldest_file_age': time.time() - min(self._access_times.values()) if self._access_times else 0
            }