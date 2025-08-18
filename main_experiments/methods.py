"""
Method Result Loader Module

This module handles loading pre-processed results from different glare removal methods.
The actual glare removal processing is done externally - this framework only evaluates 
the results by comparing them to ground truth data.
"""

from typing import Dict, Any
from data_loader import load_events
import numpy as np


class MethodResult:
    """
    Represents the result from a glare removal method.
    Simply holds the method name and file path to the processed results.
    """
    
    def __init__(self, name: str, result_file_path: str, **metadata):
        """
        Initialize a method result.
        
        Args:
            name (str): Name/identifier for this method
            result_file_path (str): Path to the processed result file
            **metadata: Additional metadata about the method/processing
        """
        self.name = name
        self.result_file_path = result_file_path
        self.metadata = metadata
        self._events = None

    def get_events(self) -> np.ndarray:
        """
        Load and return the processed events from this method.
        
        Returns:
            np.ndarray: Processed events as structured array
        """
        if self._events is None:
            self._events = load_events(self.result_file_path)
        return self._events

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration and metadata."""
        return {
            'name': self.name,
            'result_file_path': self.result_file_path,
            **self.metadata
        }