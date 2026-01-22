"""
Sampling strategies for handling class imbalance.

This module implements the Strategy Pattern for different resampling approaches.
Each strategy encapsulates a specific algorithm for balancing training data.
"""

from typing import Tuple
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from .interfaces import ISamplingStrategy


class NoSamplingStrategy(ISamplingStrategy):
    """Strategy that performs no resampling (pass-through)."""
    
    def resample(self, X, y) -> Tuple:
        """Return data unchanged.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Original (X, y) unchanged
        """
        return X, y


class RandomOversamplingStrategy(ISamplingStrategy):
    """Strategy using random oversampling of minority class."""
    
    def __init__(self, sampling_strategy: str = "minority", random_state: int = None):
        """
        Initialize random oversampler.
        
        Args:
            sampling_strategy: 'minority', 'all', or float ratio
            random_state: Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
    
    def resample(self, X, y) -> Tuple:
        """
        Apply random oversampling.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        sampler = RandomOverSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        return sampler.fit_resample(X, y)


class SMOTEStrategy(ISamplingStrategy):
    """Strategy using SMOTE (Synthetic Minority Over-sampling Technique)."""
    
    def __init__(self, sampling_strategy: str = "minority", random_state: int = None):
        """
        Initialize SMOTE sampler.
        
        Args:
            sampling_strategy: 'minority', 'all', or float ratio
            random_state: Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
    
    def resample(self, X, y) -> Tuple:
        """
        Apply SMOTE oversampling.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        sampler = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        return sampler.fit_resample(X, y)


class ADASYNStrategy(ISamplingStrategy):
    """
    Strategy using ADASYN (Adaptive Synthetic Sampling).
    
    ADASYN is an extension of SMOTE that generates more synthetic samples
    for minority class examples that are harder to learn.
    """
    
    def __init__(self, sampling_strategy: str = "minority", random_state: int = None):
        """
        Initialize ADASYN sampler.
        
        Args:
            sampling_strategy: 'minority', 'all', or float ratio
            random_state: Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
    
    def resample(self, X, y) -> Tuple:
        """
        Apply ADASYN oversampling.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        sampler = ADASYN(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        return sampler.fit_resample(X, y)


class SamplingStrategyFactory:
    """Factory for creating sampling strategies from configuration."""
    
    _strategy_registry = {
        "none": NoSamplingStrategy,
        "random": RandomOversamplingStrategy,
        "smote": SMOTEStrategy,
        "adasyn": ADASYNStrategy,
    }
    
    @classmethod
    def create_strategy(cls, method: str, sampling_strategy: str = "minority",
                       random_state: int = None) -> ISamplingStrategy:
        """
        Create a sampling strategy instance.
        
        Args:
            method: Sampling method name ("none", "random", "smote", "adasyn")
            sampling_strategy: How to balance ('minority', 'all', or float)
            random_state: Random seed for reproducibility
            
        Returns:
            ISamplingStrategy instance
            
        Raises:
            ValueError: if method not found in registry
        """
        method = method.lower()
        
        if method not in cls._strategy_registry:
            available = ", ".join(cls._strategy_registry.keys())
            raise ValueError(
                f"Unknown sampling method: '{method}'. "
                f"Available methods: {available}"
            )
        
        strategy_class = cls._strategy_registry[method]
        
        # NoSamplingStrategy doesn't take parameters
        if method == "none":
            return strategy_class()
        
        return strategy_class(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a custom sampling strategy.
        
        Args:
            name: Identifier for the strategy
            strategy_class: Class implementing ISamplingStrategy
        """
        cls._strategy_registry[name] = strategy_class
    
    @classmethod
    def list_available_strategies(cls) -> list:
        """
        Get list of registered sampling strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls._strategy_registry.keys())