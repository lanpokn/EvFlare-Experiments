"""
Event Data Metrics Module

This module provides various distance metrics for comparing event streams,
including Chamfer Distance and Gaussian Distance as provided by the user.

Features:
- Metric registry system for extensible metric management
- Configuration-driven metric selection
- Backward compatibility with existing code
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Dict, Any, List, Callable, Optional
import warnings

# Import torch and sklearn for voxel metrics (conditional import for robustness)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, voxel metrics will be disabled")

try:
    from sklearn.metrics import f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available, F1-based voxel metrics will be disabled")


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
    evs_normalized['t'] = ((evs['t'] - evs['t'].min()) * 100 / (evs['t'].max() - evs['t'].min() + epsilon)).astype(float)
    
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
    
    # Create KDTree using ground truth (evs2) as reference points
    tree_gt = KDTree(evs2_float)

    # Only query estimated events (evs1) against ground truth tree
    # This avoids penalizing removal of flare events without adding background
    dists_est_to_gt, _ = tree_gt.query(evs1_float)

    # Return mean distance from estimated events to nearest ground truth events
    return np.mean(dists_est_to_gt)


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
    
    # Create KDTree using ground truth (evs2) as reference points
    tree_gt = KDTree(evs2_float)

    # Only query estimated events (evs1) against ground truth tree
    # This avoids penalizing removal of flare events without adding background
    dists_est_to_gt, _ = tree_gt.query(evs1_float)
    dists_gaussian = 1 - np.exp(-dists_est_to_gt * dists_est_to_gt / sigma)

    # Return mean Gaussian-weighted distance from estimated events to ground truth
    return np.mean(dists_gaussian)


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


def calculate_pger(evs_pred: np.ndarray, evs_gt: np.ndarray) -> float:
    """
    Calculate PGER (Predicted/Ground Truth Event Number Ratio).
    
    This is a simple but important "sanity check". A good decoder should generate 
    roughly the same number of events as the ground truth.
    
    Args:
        evs_pred (np.ndarray): Predicted/estimated event stream
        evs_gt (np.ndarray): Ground truth event stream
        
    Returns:
        float: PGER value. Closer to 1.0 is better.
               >1.0: over-sampling/noise generation
               <1.0: under-sampling/information loss
               
    Note:
        - PGER = 1.0 indicates perfect event count matching
        - This metric is ratio-based, not distance-based
    """
    num_pred = len(evs_pred)
    num_gt = len(evs_gt)
    
    # Handle edge case where ground truth is empty
    if num_gt == 0:
        return np.inf if num_pred > 0 else 0.0
    
    return num_pred / num_gt


# Metric Registry System
_METRIC_REGISTRY = {}

def register_metric(name: str, func: Callable[[np.ndarray, np.ndarray], float], 
                   description: str = "", category: str = "distance"):
    """
    Register a metric function in the global registry.
    
    Args:
        name (str): Unique metric name
        func (Callable): Function that takes (events_est, events_gt) and returns float
        description (str): Human-readable description
        category (str): Metric category (distance, count, temporal, etc.)
    """
    _METRIC_REGISTRY[name] = {
        'func': func,
        'description': description, 
        'category': category
    }

def get_available_metrics() -> Dict[str, Dict[str, str]]:
    """Get all registered metrics with their metadata."""
    return {name: {'description': info['description'], 'category': info['category']} 
            for name, info in _METRIC_REGISTRY.items()}

def get_metric_names_by_category(category: str = None) -> List[str]:
    """Get metric names, optionally filtered by category."""
    if category is None:
        return list(_METRIC_REGISTRY.keys())
    return [name for name, info in _METRIC_REGISTRY.items() 
            if info['category'] == category]

def calculate_metrics(events_est: np.ndarray, events_gt: np.ndarray, 
                     metric_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate specified metrics between estimated and ground truth events.
    
    Args:
        events_est (np.ndarray): Estimated event stream
        events_gt (np.ndarray): Ground truth event stream
        metric_names (List[str], optional): Metrics to calculate. If None, calculates all.
        
    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    if metric_names is None:
        metric_names = list(_METRIC_REGISTRY.keys())
    
    metrics = {}
    
    for name in metric_names:
        if name not in _METRIC_REGISTRY:
            print(f"Warning: Unknown metric '{name}', skipping")
            continue
            
        try:
            metric_func = _METRIC_REGISTRY[name]['func']
            metrics[name] = metric_func(events_est, events_gt)
        except Exception as e:
            print(f"Warning: {name} calculation failed: {e}")
            metrics[name] = float('inf') if 'distance' in name.lower() else 0.0
    
    # Always include basic counts
    metrics['est_event_count'] = len(events_est)
    metrics['gt_event_count'] = len(events_gt)
    
    return metrics

def calculate_all_metrics(events_est: np.ndarray, events_gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate all available metrics between estimated and ground truth events.
    
    Args:
        events_est (np.ndarray): Estimated event stream
        events_gt (np.ndarray): Ground truth event stream
        
    Returns:
        Dict[str, float]: Dictionary of all computed metrics
        
    Note:
        This function maintains backward compatibility. New code should use calculate_metrics().
    """
    return calculate_metrics(events_est, events_gt)


# ===== VOXEL-BASED METRICS =====

def _prepare_voxel_pair(events_est: np.ndarray, events_gt: np.ndarray, 
                       num_bins: int = 8, sensor_size: tuple = (480, 640)) -> tuple:
    """
    Convert two event streams to voxel pairs for comparison.
    
    Args:
        events_est: Estimated events
        events_gt: Ground truth events
        num_bins: Number of temporal bins
        sensor_size: (H, W) sensor dimensions
        
    Returns:
        tuple: (voxel_chunks_est, voxel_chunks_gt) - lists of voxel tensors
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for voxel metrics")
    
    from voxel_utils import events_to_voxel_chunks
    
    # Convert events to voxel chunks
    voxel_chunks_est = events_to_voxel_chunks(events_est, num_bins, sensor_size)
    voxel_chunks_gt = events_to_voxel_chunks(events_gt, num_bins, sensor_size)
    
    return voxel_chunks_est, voxel_chunks_gt


def _separate_polarity_voxels(voxel: torch.Tensor) -> torch.Tensor:
    """
    Separate positive and negative polarities in a voxel.
    
    Args:
        voxel: Input voxel of shape (T, H, W) with mixed polarities
        
    Returns:
        torch.Tensor: Voxel of shape (2, T, H, W) with [positive, negative] channels
    """
    positive_voxel = torch.clamp(voxel, min=0)  # Keep only positive values
    negative_voxel = torch.clamp(voxel, max=0).abs()  # Keep only negative values, make positive
    
    return torch.stack([positive_voxel, negative_voxel], dim=0)  # Shape: (2, T, H, W)


def calculate_pmse(events_est: np.ndarray, events_gt: np.ndarray, pool_size: int = 2) -> float:
    """
    Calculate Pooling Mean Squared Error (PMSE) between event streams.
    
    Args:
        events_est: Estimated event stream
        events_gt: Ground truth event stream
        pool_size: Pooling kernel size and stride
        
    Returns:
        float: PMSE value (lower is better)
        
    Note:
        Implements the user's PMSE specification:
        - Convert events to voxels
        - Apply 2D average pooling to reduce spatial resolution
        - Calculate MSE on pooled voxels
        - Average across all chunks
    """
    if not TORCH_AVAILABLE:
        return float('inf')
    
    try:
        # Convert to voxel chunks
        voxel_chunks_est, voxel_chunks_gt = _prepare_voxel_pair(events_est, events_gt)
        
        if len(voxel_chunks_est) == 0 or len(voxel_chunks_gt) == 0:
            return float('inf')
        
        chunk_pmse_scores = []
        
        for voxel_est, voxel_gt in zip(voxel_chunks_est, voxel_chunks_gt):
            # Separate polarities: (T, H, W) -> (2, T, H, W)  
            voxel_est_pol = _separate_polarity_voxels(voxel_est)
            voxel_gt_pol = _separate_polarity_voxels(voxel_gt)
            
            # Reshape to (B=1, C=2*T, H, W) for 2D pooling as per user's specification
            B, T, H, W = 1, voxel_est_pol.shape[1], voxel_est_pol.shape[2], voxel_est_pol.shape[3]
            C = 2 * T  # 2 polarities * T temporal bins
            
            voxel_est_reshaped = voxel_est_pol.view(1, C, H, W)
            voxel_gt_reshaped = voxel_gt_pol.view(1, C, H, W)
            
            # Apply 2D average pooling 
            pooled_est = F.avg_pool2d(voxel_est_reshaped, kernel_size=pool_size, stride=pool_size)
            pooled_gt = F.avg_pool2d(voxel_gt_reshaped, kernel_size=pool_size, stride=pool_size)
            
            # Calculate MSE for this chunk
            mse_loss = F.mse_loss(pooled_est, pooled_gt)
            chunk_pmse_scores.append(mse_loss.item())
        
        # Average across all chunks
        return float(np.mean(chunk_pmse_scores))
        
    except Exception as e:
        warnings.warn(f"PMSE calculation failed: {e}")
        return float('inf')


def calculate_f1_metrics(events_est: np.ndarray, events_gt: np.ndarray, 
                        threshold: float = 0.001) -> Dict[str, float]:
    """
    Calculate RF1, TF1, TPF1 scores between event streams.
    
    Args:
        events_est: Estimated event stream
        events_gt: Ground truth event stream 
        threshold: Binary threshold for voxel values
        
    Returns:
        dict: Contains 'RF1', 'TF1', 'TPF1' scores
        
    Note:
        Implements the user's F1 specification:
        - RF1: Raw F1, no dimension collapse
        - TF1: Temporal F1, collapse time dimension
        - TPF1: Temporal & Polarity F1, collapse time and polarity
    """
    if not (TORCH_AVAILABLE and SKLEARN_AVAILABLE):
        return {'RF1': 0.0, 'TF1': 0.0, 'TPF1': 0.0}
    
    try:
        # Convert to voxel chunks
        voxel_chunks_est, voxel_chunks_gt = _prepare_voxel_pair(events_est, events_gt)
        
        if len(voxel_chunks_est) == 0 or len(voxel_chunks_gt) == 0:
            return {'RF1': 0.0, 'TF1': 0.0, 'TPF1': 0.0}
        
        chunk_f1_scores = {'RF1': [], 'TF1': [], 'TPF1': []}
        
        for voxel_est, voxel_gt in zip(voxel_chunks_est, voxel_chunks_gt):
            # Separate polarities: (T, H, W) -> (2, T, H, W)
            voxel_est_pol = _separate_polarity_voxels(voxel_est)  
            voxel_gt_pol = _separate_polarity_voxels(voxel_gt)
            
            # Add batch dimension: (2, T, H, W) -> (1, 2, T, H, W)
            voxel_est_batch = voxel_est_pol.unsqueeze(0)
            voxel_gt_batch = voxel_gt_pol.unsqueeze(0)
            
            # Convert to numpy for sklearn
            voxel_est_np = voxel_est_batch.cpu().numpy()
            voxel_gt_np = voxel_gt_batch.cpu().numpy()
            
            # --- RF1 (Raw F1) - No collapse, most strict ---
            binary_pred_r = (voxel_est_np > threshold).flatten()
            binary_gt_r = (voxel_gt_np > threshold).flatten()
            rf1 = f1_score(binary_gt_r, binary_pred_r, zero_division=0)
            chunk_f1_scores['RF1'].append(rf1)
            
            # --- TF1 (Temporal F1) - Collapse time dimension (axis=2) ---
            collapsed_pred_t = voxel_est_np.sum(axis=2)  # Sum over T
            collapsed_gt_t = voxel_gt_np.sum(axis=2)
            binary_pred_t = (collapsed_pred_t > threshold).flatten()
            binary_gt_t = (collapsed_gt_t > threshold).flatten()
            tf1 = f1_score(binary_gt_t, binary_pred_t, zero_division=0)
            chunk_f1_scores['TF1'].append(tf1)
            
            # --- TPF1 (Temporal & Polarity F1) - Collapse time and polarity (axis=1,2) ---
            collapsed_pred_tp = voxel_est_np.sum(axis=(1, 2))  # Sum over polarity and time
            collapsed_gt_tp = voxel_gt_np.sum(axis=(1, 2))
            binary_pred_tp = (collapsed_pred_tp > threshold).flatten()
            binary_gt_tp = (collapsed_gt_tp > threshold).flatten()
            tpf1 = f1_score(binary_gt_tp, binary_pred_tp, zero_division=0)
            chunk_f1_scores['TPF1'].append(tpf1)
        
        # Average F1 scores across all chunks
        return {
            'RF1': float(np.mean(chunk_f1_scores['RF1'])),
            'TF1': float(np.mean(chunk_f1_scores['TF1'])), 
            'TPF1': float(np.mean(chunk_f1_scores['TPF1']))
        }
        
    except Exception as e:
        warnings.warn(f"F1 metrics calculation failed: {e}")
        return {'RF1': 0.0, 'TF1': 0.0, 'TPF1': 0.0}


# Individual F1 metric functions for registration
def calculate_rf1(events_est: np.ndarray, events_gt: np.ndarray) -> float:
    """Raw F1 score - most strict, no dimension collapse."""
    return calculate_f1_metrics(events_est, events_gt)['RF1']

def calculate_tf1(events_est: np.ndarray, events_gt: np.ndarray) -> float:
    """Temporal F1 score - collapse time dimension."""
    return calculate_f1_metrics(events_est, events_gt)['TF1']

def calculate_tpf1(events_est: np.ndarray, events_gt: np.ndarray) -> float:
    """Temporal & Polarity F1 score - collapse time and polarity dimensions.""" 
    return calculate_f1_metrics(events_est, events_gt)['TPF1']

def calculate_pmse_2(events_est: np.ndarray, events_gt: np.ndarray) -> float:
    """PMSE with pool size 2."""
    return calculate_pmse(events_est, events_gt, pool_size=2)

def calculate_pmse_4(events_est: np.ndarray, events_gt: np.ndarray) -> float:
    """PMSE with pool size 4."""
    return calculate_pmse(events_est, events_gt, pool_size=4)


# Register all existing metrics automatically on module import
register_metric(
    'chamfer_distance', 
    chamfer_distance_loss,
    'Chamfer distance between event streams using KDTree nearest neighbors',
    'distance'
)

register_metric(
    'gaussian_distance', 
    gaussian_distance_loss,
    'Gaussian-weighted distance with configurable sigma parameter',
    'distance'  
)

register_metric(
    'event_count_ratio',
    calculate_event_count_ratio, 
    'Ratio of event counts between estimated and ground truth streams',
    'count'
)

register_metric(
    'temporal_overlap',
    calculate_temporal_coverage_overlap,
    'Temporal coverage overlap ratio between event streams', 
    'temporal'
)

register_metric(
    'pger',
    calculate_pger,
    'PGER (Predicted/Ground Truth Event Number Ratio) - sanity check for event count matching',
    'count'
)

# Register voxel-based metrics (conditional on dependencies)
if TORCH_AVAILABLE and SKLEARN_AVAILABLE:
    register_metric(
        'pmse_2',
        calculate_pmse_2,
        'Pooling Mean Squared Error with pool size 2 (voxel-based)',
        'voxel'
    )
    
    register_metric(
        'pmse_4', 
        calculate_pmse_4,
        'Pooling Mean Squared Error with pool size 4 (voxel-based)',
        'voxel'
    )
    
    register_metric(
        'rf1',
        calculate_rf1,
        'Raw F1 score - no dimension collapse (voxel-based)',
        'voxel'
    )
    
    register_metric(
        'tf1',
        calculate_tf1,
        'Temporal F1 score - collapse time dimension (voxel-based)', 
        'voxel'
    )
    
    register_metric(
        'tpf1',
        calculate_tpf1,
        'Temporal & Polarity F1 score - collapse time and polarity (voxel-based)',
        'voxel'
    )