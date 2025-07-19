"""
Comprehensive Error Handling and Input Validation

Provides structured error handling, input validation, and security controls
for the LoTek fractal cybersecurity neural network system.
"""

import functools
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import torch

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels for proper escalation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationError:
    """Structured validation error information."""
    field: str
    value: Any
    message: str
    severity: ErrorSeverity
    
class FractalError(Exception):
    """Base exception for all LoTek fractal network errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = __import__('time').time()

class SecurityError(FractalError):
    """Security-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, context)

class ValidationError(FractalError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: str = "", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, context)
        self.field = field

class ConfigurationError(FractalError):
    """Configuration-related errors."""
    pass

class ModelError(FractalError):
    """Model training/inference errors."""
    pass

class MemoryError(FractalError):
    """Memory management errors."""
    pass

class InputValidator:
    """Comprehensive input validation for LoTek fractal network components."""
    
    @staticmethod
    def validate_tensor(tensor: Any, name: str = "tensor", 
                       expected_dims: Optional[int] = None,
                       expected_shape: Optional[tuple] = None,
                       min_value: Optional[float] = None,
                       max_value: Optional[float] = None) -> torch.Tensor:
        """Validate PyTorch tensor inputs."""
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}", name)
        
        if expected_dims is not None and len(tensor.shape) != expected_dims:
            raise ValidationError(
                f"{name} must have {expected_dims} dimensions, got {len(tensor.shape)}", 
                name
            )
        
        if expected_shape is not None:
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise ValidationError(
                        f"{name} dimension {i} must be {expected}, got {actual}",
                        name
                    )
        
        if min_value is not None and tensor.min() < min_value:
            raise ValidationError(f"{name} values must be >= {min_value}", name)
        
        if max_value is not None and tensor.max() > max_value:
            raise ValidationError(f"{name} values must be <= {max_value}", name)
        
        return tensor
    
    @staticmethod
    def validate_string(value: Any, name: str = "string", 
                       min_length: int = 1, max_length: int = 1000,
                       allowed_chars: Optional[str] = None) -> str:
        """Validate string inputs."""
        if not isinstance(value, str):
            raise ValidationError(f"{name} must be a string, got {type(value)}", name)
        
        if len(value) < min_length:
            raise ValidationError(f"{name} must be at least {min_length} characters", name)
        
        if len(value) > max_length:
            raise ValidationError(f"{name} must be at most {max_length} characters", name)
        
        if allowed_chars is not None:
            invalid_chars = set(value) - set(allowed_chars)
            if invalid_chars:
                raise ValidationError(
                    f"{name} contains invalid characters: {invalid_chars}",
                    name
                )
        
        return value
    
    @staticmethod
    def validate_number(value: Any, name: str = "number",
                       min_value: Optional[float] = None,
                       max_value: Optional[float] = None,
                       integer_only: bool = False) -> Union[int, float]:
        """Validate numeric inputs."""
        if integer_only and not isinstance(value, int):
            raise ValidationError(f"{name} must be an integer, got {type(value)}", name)
        
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number, got {type(value)}", name)
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} must be >= {min_value}", name)
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}", name)
        
        return value
    
    @staticmethod
    def validate_list(value: Any, name: str = "list",
                     min_length: int = 0, max_length: Optional[int] = None,
                     item_validator: Optional[Callable] = None) -> list:
        """Validate list inputs."""
        if not isinstance(value, list):
            raise ValidationError(f"{name} must be a list, got {type(value)}", name)
        
        if len(value) < min_length:
            raise ValidationError(f"{name} must have at least {min_length} items", name)
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(f"{name} must have at most {max_length} items", name)
        
        if item_validator is not None:
            for i, item in enumerate(value):
                try:
                    item_validator(item)
                except ValidationError as e:
                    raise ValidationError(f"{name}[{i}]: {e.message}", f"{name}[{i}]")
        
        return value

def error_handler(exceptions: Union[Type[Exception], tuple] = Exception,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 log_errors: bool = True,
                 reraise: bool = True,
                 default_return: Any = None):
    """Decorator for comprehensive error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                error_context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate for security
                    'kwargs': str(kwargs)[:200],
                    'error_type': type(e).__name__
                }
                
                if log_errors:
                    if severity == ErrorSeverity.CRITICAL:
                        logger.critical(f"Critical error in {func.__name__}: {e}", 
                                      extra={'context': error_context})
                    elif severity == ErrorSeverity.HIGH:
                        logger.error(f"High severity error in {func.__name__}: {e}",
                                   extra={'context': error_context})
                    elif severity == ErrorSeverity.MEDIUM:
                        logger.warning(f"Medium severity error in {func.__name__}: {e}",
                                     extra={'context': error_context})
                    else:
                        logger.info(f"Low severity error in {func.__name__}: {e}",
                                  extra={'context': error_context})
                
                if reraise:
                    if isinstance(e, FractalError):
                        raise
                    else:
                        # Wrap in FractalError for consistent handling
                        raise FractalError(str(e), severity, error_context) from e
                else:
                    return default_return
        
        return wrapper
    return decorator

def validate_model_inputs(func: Callable) -> Callable:
    """Decorator to validate model forward pass inputs."""
    
    @functools.wraps(func)
    def wrapper(self, X: torch.Tensor, *args, **kwargs):
        try:
            # Validate input tensor
            InputValidator.validate_tensor(
                X, "input_tensor", 
                expected_dims=2,
                min_value=0  # Assuming token indices are non-negative
            )
            
            # Check for reasonable sequence length
            batch_size, seq_len = X.shape
            if seq_len > 10000:  # Configurable limit
                logger.warning(f"Very long sequence detected: {seq_len} tokens")
            
            if batch_size > 1000:  # Configurable limit
                logger.warning(f"Very large batch detected: {batch_size} samples")
            
            return func(self, X, *args, **kwargs)
            
        except Exception as e:
            raise ModelError(f"Model input validation failed: {e}")
    
    return wrapper

def secure_operation(func: Callable) -> Callable:
    """Decorator for operations that require security validation."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Log security-sensitive operations
            logger.info(f"Security operation: {func.__name__}", 
                       extra={'security_event': True})
            
            result = func(*args, **kwargs)
            
            logger.debug(f"Security operation completed: {func.__name__}")
            return result
            
        except Exception as e:
            logger.error(f"Security operation failed: {func.__name__}: {e}",
                        extra={'security_event': True, 'security_failure': True})
            raise SecurityError(f"Security operation failed: {e}")
    
    return wrapper

class ErrorContext:
    """Context manager for structured error handling."""
    
    def __init__(self, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.operation = operation
        self.severity = severity
        self.start_time = None
    
    def __enter__(self):
        self.start_time = __import__('time').time()
        logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = __import__('time').time() - self.start_time
        
        if exc_type is None:
            logger.debug(f"Operation completed: {self.operation} ({duration:.3f}s)")
        else:
            logger.error(f"Operation failed: {self.operation} ({duration:.3f}s): {exc_val}")
            
            if issubclass(exc_type, FractalError):
                return False  # Re-raise FractalError as-is
            else:
                # Convert to FractalError
                context = {
                    'operation': self.operation,
                    'duration': duration,
                    'traceback': traceback.format_exc()
                }
                raise FractalError(str(exc_val), self.severity, context) from exc_val
        
        return False

# Global error configuration
class ErrorConfig:
    """Global error handling configuration."""
    
    LOG_SENSITIVE_DATA = False
    MAX_TRACEBACK_LINES = 10
    ERROR_REPORTING_ENABLED = True
    SECURITY_LOGGING_ENABLED = True
    
    @classmethod
    def configure(cls, **kwargs):
        """Configure global error handling settings."""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
            else:
                logger.warning(f"Unknown error config option: {key}")