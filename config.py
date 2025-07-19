"""
Legacy Configuration Module

Provides backward compatibility for existing code while transitioning to the new
configuration management system. New code should use src.config_management directly.
"""

import os
import warnings
from src.config_management import ConfigManager, Environment

_config_manager = ConfigManager()
_config = _config_manager.load_config()

VOCAB_SIZE = _config.vocab_size
EMBEDDING_DIM = _config.embedding_dim
BATCH_SIZE = _config.batch_size
EPOCHS = _config.epochs
NUM_TOKENS = _config.vocab_size
NUM_CLUSTERS = _config.num_clusters
WANDB_PROJECT = _config.wandb_project

def get_config():
    """Get current configuration object."""
    return _config_manager.load_config()

def get_config_manager():
    """Get configuration manager instance."""
    return _config_manager

def get_config_for_env(environment: Environment):
    """Get configuration for specific environment."""
    return _config_manager.load_config(environment)

warnings.warn(
    "Direct import from config.py is deprecated. Use src.config_management.ConfigManager instead.",
    DeprecationWarning,
    stacklevel=2
)