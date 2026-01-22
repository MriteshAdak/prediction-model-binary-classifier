"""
Scaler factory for feature normalization.

This module provides a factory for creating different types of feature scalers.
Scalers normalize feature values to improve model performance.
"""

from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class NoScaler:
    """
    Pass-through scaler that performs no transformation.
    
    Useful for cases where scaling is disabled via configuration.
    """
    
    def fit(self, X, y=None):
        """
        No-op fit method.
        
        Args:
            X: Feature matrix (ignored)
            y: Target vector (ignored)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        Return data unchanged.
        
        Args:
            X: Feature matrix
            
        Returns:
            X unchanged
        """
        return X
    
    def fit_transform(self, X, y=None):
        """
        Return data unchanged.
        
        Args:
            X: Feature matrix
            y: Target vector (ignored)
            
        Returns:
            X unchanged
        """
        return X


class ScalerFactory:
    """
    Factory for creating feature scalers from configuration."""
    
    _scaler_registry = {
        "none": NoScaler,
        "standard": StandardScaler,  # Z-score normalization: (x - mean) / std
        "minmax": MinMaxScaler,      # Scale to [0, 1] range
        "robust": RobustScaler,      # Uses median and IQR (robust to outliers)
    }
    
    _descriptions = {
        "none": "No scaling (pass-through)",
        "standard": "StandardScaler: Z-score normalization (mean=0, std=1)",
        "minmax": "MinMaxScaler: Scale features to [0, 1] range",
        "robust": "RobustScaler: Scale using median and IQR (robust to outliers)",
    }
    
    @classmethod
    def create_scaler(cls, scaler_type: str = "standard"):
        """
        Create a scaler instance from type.
        
        Args:
            scaler_type: Type of scaler ("none", "standard", "minmax", "robust")
            
        Returns:
            Scaler instance with fit() and transform() methods
            
        Raises:
            ValueError: if scaler_type not found in registry
        """
        scaler_type = scaler_type.lower()
        
        if scaler_type not in cls._scaler_registry:
            available = ", ".join(cls._scaler_registry.keys())
            raise ValueError(
                f"Unknown scaler type: '{scaler_type}'. "
                f"Available scalers: {available}"
            )
        
        scaler_class = cls._scaler_registry[scaler_type]
        return scaler_class()
    
    @classmethod
    def get_description(cls, scaler_type: str) -> str:
        """
        Get description of a scaler type.
        
        Args:
            scaler_type: Type of scaler
            
        Returns:
            Human-readable description
        """
        return cls._descriptions.get(scaler_type, "Unknown scaler")
    
    @classmethod
    def register_scaler(cls, name: str, scaler_class, description: str = ""):
        """
        Register a custom scaler.
        
        Args:
            name: Identifier for the scaler
            scaler_class: Class with fit() and transform() methods
            description: Human-readable description
        """
        cls._scaler_registry[name] = scaler_class
        cls._descriptions[name] = description or name
    
    @classmethod
    def list_available_scalers(cls) -> list:
        """
        Get list of registered scaler types.
        
        Returns:
            List of scaler names
        """
        return list(cls._scaler_registry.keys())