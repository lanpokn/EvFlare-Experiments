"""
Event Data Metrics Module

This module provides various distance metrics for comparing event streams,
including Chamfer Distance and Gaussian Distance as provided by the user.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Dict, Any


def normalize_evs(evs: np.ndarray) -> np.ndarray:
    """
    Normalize events data to standard ranges for consistent metric computation.

    Args:
        evs (np.ndarray): Events data, should be structured array with fields ('t', 'x', 'y', 'p')

    Returns:
        np.ndarray: Data after normalization
    """    
    evs_normalized = evs.copy()
    epsilon = 1e-8  # Small epsilon value to avoid division by zero
    
    evs_normalized['x'] = ((evs['x'] - evs['x'].min()) * 100 / (evs['x'].max() - evs['x'].min() + epsilon)).astype(float)
    evs_normalized['y'] = ((evs['y'] - evs['y'].min()) * 100 / (evs['y'].max() - evs['y'].min() + epsilon)).astype(float)
    evs_normalized['p'] = ((evs['p'] - evs['p'].min()) * 100 / (evs['p'].max() - evs['p'].min() + epsilon)).astype(float)
    evs_normalized['t'] = ((evs['t'] - evs['t'].min()) * 1000 / (evs['t'].max() - evs['t'].min() + epsilon)).astype(float)
    
    return evs_normalized


def chamfer_distance_loss(evs1: np.ndarray, evs2: np.ndarray) -> float:
    """
    Use chamfer distance to calculate the loss between two events data.

    Args:
        evs1 (np.ndarray): Events data1, structured array with fields ('t', 'x', 'y', 'p')
        evs2 (np.ndarray): Events data2, structured array with fields ('t', 'x', 'y', 'p')

    Returns:
        float: Chamfer distance between the two event streams
    """
    
    evs1_norm = normalize_evs(evs1)
    evs2_norm = normalize_evs(evs2)
    
    evs1_float = np.zeros((evs1_norm.shape[0], 4), dtype=np.float64)
    evs2_float = np.zeros((evs2_norm.shape[0], 4), dtype=np.float64)
    
    evs1_float[:, 0] = evs1_norm['x']
    evs1_float[:, 1] = evs1_norm['y']
    evs1_float[:, 2] = evs1_norm['p']
    evs1_float[:, 3] = evs1_norm['t']
    
    evs2_float[:, 0] = evs2_norm['x']
    evs2_float[:, 1] = evs2_norm['y']
    evs2_float[:, 2] = evs2_norm['p']
    evs2_float[:, 3] = evs2_norm['t']
    
    # Create KDTree using evs2_float as points
    tree1 = KDTree(evs1_float)
    tree2 = KDTree(evs2_float)

    # Query the tree with evs1_float as points
    dists1, _ = tree2.query(evs1_float)
    dists2, _ = tree1.query(evs2_float)

    # Return the mean of distances
    return (np.mean(dists1) + np.mean(dists2))


def gaussian_distance_loss(evs1: np.ndarray, evs2: np.ndarray, sigma: float = 0.4) -> float:
    """
    Use Gaussian-weighted distance to calculate the loss between two events data.

    Args:
        evs1 (np.ndarray): Events data1, structured array with fields ('t', 'x', 'y', 'p')
        evs2 (np.ndarray): Events data2, structured array with fields ('t', 'x', 'y', 'p')
        sigma (float): Gaussian kernel parameter

    Returns:
        float: Gaussian distance between the two event streams
    """
    
    evs1_norm = normalize_evs(evs1)
    evs2_norm = normalize_evs(evs2)
    
    evs1_float = np.zeros((evs1_norm.shape[0], 4), dtype=np.float64)
    evs2_float = np.zeros((evs2_norm.shape[0], 4), dtype=np.float64)
    
    evs1_float[:, 0] = evs1_norm['x']
    evs1_float[:, 1] = evs1_norm['y']
    evs1_float[:, 2] = evs1_norm['p']
    evs1_float[:, 3] = evs1_norm['t']
    
    evs2_float[:, 0] = evs2_norm['x']
    evs2_float[:, 1] = evs2_norm['y']
    evs2_float[:, 2] = evs2_norm['p']
    evs2_float[:, 3] = evs2_norm['t']
    
    # Create KDTree using evs2_float as points
    tree1 = KDTree(evs1_float)
    tree2 = KDTree(evs2_float)

    # Query the tree with evs1_float as points
    dists1, _ = tree2.query(evs1_float)
    dists1 = 1 - np.exp(-dists1 * dists1 / sigma)

    dists2, _ = tree1.query(evs2_float)
    dists2 = 1 - np.exp(-dists2 * dists2 / sigma)

    # Return the mean of distances
    return (np.mean(dists1) + np.mean(dists2))


def calculate_event_count_ratio(evs1: np.ndarray, evs2: np.ndarray) -> float:
    """
    Calculate the ratio of event counts between two streams.
    
    Args:
        evs1 (np.ndarray): First event stream
        evs2 (np.ndarray): Second event stream
        
    Returns:
        float: Ratio of event counts (evs1/evs2)
    """
    return len(evs1) / max(len(evs2), 1)


def calculate_temporal_coverage_overlap(evs1: np.ndarray, evs2: np.ndarray) -> float:
    """
    Calculate temporal coverage overlap between two event streams.
    
    Args:
        evs1 (np.ndarray): First event stream
        evs2 (np.ndarray): Second event stream
        
    Returns:
        float: Temporal overlap ratio [0, 1]
    """
    if len(evs1) == 0 or len(evs2) == 0:
        return 0.0
    
    t1_min, t1_max = evs1['t'].min(), evs1['t'].max()
    t2_min, t2_max = evs2['t'].min(), evs2['t'].max()
    
    overlap_start = max(t1_min, t2_min)
    overlap_end = min(t1_max, t2_max)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_duration = overlap_end - overlap_start
    total_duration = max(t1_max, t2_max) - min(t1_min, t2_min)
    
    return overlap_duration / total_duration if total_duration > 0 else 0.0


def calculate_all_metrics(events_est: np.ndarray, events_gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate all available metrics between estimated and ground truth events.
    
    Args:
        events_est (np.ndarray): Estimated event stream
        events_gt (np.ndarray): Ground truth event stream
        
    Returns:
        Dict[str, float]: Dictionary of all computed metrics
    """
    metrics = {}
    
    try:
        metrics['chamfer_distance'] = chamfer_distance_loss(events_est, events_gt)
    except Exception as e:
        print(f"Warning: Chamfer distance calculation failed: {e}")
        metrics['chamfer_distance'] = float('inf')
    
    try:
        metrics['gaussian_distance'] = gaussian_distance_loss(events_est, events_gt)
    except Exception as e:
        print(f"Warning: Gaussian distance calculation failed: {e}")
        metrics['gaussian_distance'] = float('inf')
    
    try:
        metrics['event_count_ratio'] = calculate_event_count_ratio(events_est, events_gt)
    except Exception as e:
        print(f"Warning: Event count ratio calculation failed: {e}")
        metrics['event_count_ratio'] = 0.0
    
    try:
        metrics['temporal_overlap'] = calculate_temporal_coverage_overlap(events_est, events_gt)
    except Exception as e:
        print(f"Warning: Temporal overlap calculation failed: {e}")
        metrics['temporal_overlap'] = 0.0
    
    # Basic counts
    metrics['est_event_count'] = len(events_est)
    metrics['gt_event_count'] = len(events_gt)
    
    return metrics