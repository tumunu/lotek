"""
Advanced Configuration Management System

Provides environment-specific configuration management with validation, caching,
and secure handling for the LoTek fractal cybersecurity neural network.
"""

import os
import json
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    RESEARCH = "research"

@dataclass
class LoTekConfig:
    """LoTek fractal cybersecurity neural network configuration."""
    
    # Model parameters
    vocab_size: int = 50257
    embedding_dim: int = 768
    num_clusters: int = 4
    fractal_scaling_factor: float = 1.0
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Memory management
    memory_max_size: int = 1000
    memory_path: str = "./memory"
    memory_max_age_seconds: float = 3600.0
    memory_cleanup_interval: float = 300.0
    
    # Compression
    compression_ratio: float = 0.5
    
    # Batching
    max_batch_size: int = 32
    
    # Scaling
    scale_factor: float = 1.0
    
    # Search
    search_k: int = 5
    
    # Security
    rate_limit_requests: float = 1.0
    request_timeout: int = 10
    max_concurrent_hosts: int = 10
    validate_ssl: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # MLOps
    wandb_project: str = "fractal-neural-network"
    wandb_entity: Optional[str] = None
    model_checkpoint_dir: str = "./checkpoints"
    
    # Performance
    use_cuda: bool = True
    num_workers: int = 4
    pin_memory: bool = True

class ConfigManager:
    """
    Advanced configuration management with environment-specific configs,
    validation, and secure handling.
    """
    
    def __init__(self, config_dir: str = "configs", environment: Optional[Environment] = None):
        self.config_dir = Path(config_dir)
        self.environment = environment or self._detect_environment()
        self.config_cache: Dict[str, Any] = {}
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize default configs
        self._create_default_configs()
    
    def _detect_environment(self) -> Environment:
        """Auto-detect environment from environment variables."""
        env_var = os.getenv('LOTEK_ENV', 'development').lower()
        try:
            return Environment(env_var)
        except ValueError:
            logger.warning(f"Unknown environment '{env_var}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _create_default_configs(self) -> None:
        """Create default configuration files if they don't exist."""
        configs = {
            Environment.DEVELOPMENT: LoTekConfig(
                log_level="DEBUG",
                epochs=5,
                batch_size=4,
                memory_max_size=100,
                use_cuda=False
            ),
            Environment.TESTING: LoTekConfig(
                log_level="WARNING",
                epochs=1,
                batch_size=2,
                memory_max_size=10,
                use_cuda=False,
                wandb_project="fractal-test"
            ),
            Environment.PRODUCTION: LoTekConfig(
                log_level="ERROR",
                epochs=50,
                batch_size=32,
                memory_max_size=10000,
                use_cuda=True,
                validate_ssl=True
            ),
            Environment.RESEARCH: LoTekConfig(
                log_level="INFO",
                epochs=100,
                batch_size=16,
                memory_max_size=5000,
                use_cuda=True,
                wandb_project="fractal-research"
            )
        }
        
        for env, config in configs.items():
            config_file = self.config_dir / f"{env.value}.yaml"
            if not config_file.exists():
                self._save_config(config, config_file)
                logger.info(f"Created default config: {config_file}")
    
    def _save_config(self, config: LoTekConfig, file_path: Path) -> None:
        """Save configuration to file."""
        try:
            with open(file_path, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=True)
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            raise
    
    def load_config(self, environment: Optional[Environment] = None) -> LoTekConfig:
        """Load configuration for specified environment."""
        env = environment or self.environment
        cache_key = env.value
        
        # Check cache first
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        config_file = self.config_dir / f"{env.value}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Config file {config_file} not found, using defaults")
            config = LoTekConfig()
        else:
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Merge with defaults and validate
                config = self._merge_with_defaults(config_data)
                
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")
                config = LoTekConfig()
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # Validate configuration
        self._validate_config(config)
        
        # Cache the config
        self.config_cache[cache_key] = config
        
        logger.info(f"Loaded configuration for {env.value} environment")
        return config
    
    def _merge_with_defaults(self, config_data: Dict[str, Any]) -> LoTekConfig:
        """Merge loaded config with defaults."""
        defaults = asdict(LoTekConfig())
        defaults.update(config_data)
        return LoTekConfig(**defaults)
    
    def _apply_env_overrides(self, config: LoTekConfig) -> LoTekConfig:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'LOTEK_VOCAB_SIZE': ('vocab_size', int),
            'LOTEK_EMBEDDING_DIM': ('embedding_dim', int),
            'LOTEK_BATCH_SIZE': ('batch_size', int),
            'LOTEK_EPOCHS': ('epochs', int),
            'LOTEK_LEARNING_RATE': ('learning_rate', float),
            'LOTEK_LOG_LEVEL': ('log_level', str),
            'LOTEK_WANDB_PROJECT': ('wandb_project', str),
            'LOTEK_USE_CUDA': ('use_cuda', lambda x: x.lower() == 'true'),
            'LOTEK_MEMORY_MAX_SIZE': ('memory_max_size', int),
            'LOTEK_MEMORY_PATH': ('memory_path', str),
        }
        
        config_dict = asdict(config)
        
        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    config_dict[config_key] = converter(env_value)
                    logger.debug(f"Applied env override: {config_key} = {config_dict[config_key]}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid env var {env_var}={env_value}: {e}")
        
        return LoTekConfig(**config_dict)
    
    def _validate_config(self, config: LoTekConfig) -> None:
        """Validate configuration parameters."""
        validations = [
            (config.vocab_size > 0, "vocab_size must be positive"),
            (config.embedding_dim > 0, "embedding_dim must be positive"),
            (config.batch_size > 0, "batch_size must be positive"),
            (config.epochs > 0, "epochs must be positive"),
            (0.0 <= config.dropout <= 1.0, "dropout must be between 0 and 1"),
            (config.learning_rate > 0, "learning_rate must be positive"),
            (config.memory_max_size > 0, "memory_max_size must be positive"),
            (config.compression_ratio > 0, "compression_ratio must be positive"),
            (config.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
             "log_level must be valid"),
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"Configuration validation failed: {message}")
        
        logger.debug("Configuration validation passed")
    
    def get_config_dict(self, environment: Optional[Environment] = None) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        config = self.load_config(environment)
        return asdict(config)
    
    def update_config(self, updates: Dict[str, Any], 
                     environment: Optional[Environment] = None) -> None:
        """Update configuration with new values."""
        env = environment or self.environment
        config = self.load_config(env)
        config_dict = asdict(config)
        
        # Apply updates
        config_dict.update(updates)
        
        # Create new config and validate
        updated_config = LoTekConfig(**config_dict)
        self._validate_config(updated_config)
        
        # Save to file
        config_file = self.config_dir / f"{env.value}.yaml"
        self._save_config(updated_config, config_file)
        
        # Update cache
        self.config_cache[env.value] = updated_config
        
        logger.info(f"Updated configuration for {env.value}")
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security-specific configuration."""
        config = self.load_config()
        return {
            'rate_limit_requests': config.rate_limit_requests,
            'request_timeout': config.request_timeout,
            'max_concurrent_hosts': config.max_concurrent_hosts,
            'validate_ssl': config.validate_ssl
        }
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self.config_cache.clear()
        logger.debug("Configuration cache cleared")