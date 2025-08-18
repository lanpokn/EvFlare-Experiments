"""
Event Glare Removal - Main Experiments Framework

This package provides a modular framework for evaluating glare removal methods
on event camera data. The framework is designed to be format-agnostic and
easily extensible.

Key Components:
- data_loader: Event data loading from various formats
- methods: Glare removal algorithm interfaces and implementations  
- metrics: Evaluation metrics for comparing event streams
- evaluator: Main experiment orchestration and result collection

Example Usage:
    from main_experiments import MainExperimentEvaluator, MyAwesomeModel, BaselineFilter
    
    methods = [MyAwesomeModel(), BaselineFilter()]
    evaluator = MainExperimentEvaluator(methods)
    evaluator.run_single_evaluation('obs.aedat4', 'gt.aedat4', 'sample1')
"""

__version__ = "1.0.0"
__author__ = "Event Flick Flare Research Team"

# Import main classes for easy access
from .data_loader import load_events, DataSource
from .methods import MethodResult
from .metrics import calculate_all_metrics, chamfer_distance_loss, gaussian_distance_loss
from .evaluator import MainExperimentEvaluator

__all__ = [
    'load_events',
    'DataSource', 
    'MethodResult',
    'calculate_all_metrics',
    'chamfer_distance_loss', 
    'gaussian_distance_loss',
    'MainExperimentEvaluator'
]