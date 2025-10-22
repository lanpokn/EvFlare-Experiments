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
import math

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

def cubes_3d_kernel_method(events, new_events, x_sigma, y_sigma, t_sigma, chunk_size=1000):
    """
    extern (MEMORY-OPTIMIZED VERSION - DOUBLE CHUNKING)

    Computing inner product between spike cubes using 3d gaussian kernel method.
    DOUBLE CHUNKED PROCESSING to avoid memory explosion with large event counts.

    Inputs:
    -------
        events    - events include polarity, timestamp, x and y.
        new_events    - events after changing operation.
        x_sigma, y_sigma, t_sigma  - the parameters of 3d gaussian kernel.
        chunk_size - process events in chunks to limit memory usage (default: 1000)

    Outputs:
    -------
        inner_product    - the inner product between events and new_events.

    Notes:
    ------
        Original implementation creates (N1 × N2) matrix which explodes with large N.
        Example: 140K × 12M events = 1.68 trillion elements = 13.4 TB memory → CRASH

        Solution 1: Single chunk - still risky
        Example: 5000 × 12M events = 60M elements = 480 MB per chunk → May still crash

        Solution 2: DOUBLE chunking (current implementation)
        Example: 1000 × 1000 events = 1M elements = 8 MB per block → SAFE

        Mathematical equivalence:
        sum_{i=1}^{N1} sum_{j=1}^{N2} f(i,j) = sum_{chunk_i} sum_{chunk_j} sum_{i} sum_{j} f(i,j)
    """
    n1 = events.shape[1]
    n2 = new_events.shape[1]

    # Edge case: empty events
    if n1 == 0 or n2 == 0:
        return 0.0

    # Calculate polarity scale once
    ON_scale = np.sum(events[0, :] == 1) / n1
    new_ON_scale = np.sum(new_events[0, :] == 1) / n2
    polarity_scale = ON_scale * new_ON_scale + (1 - ON_scale) * (1 - new_ON_scale)

    # DOUBLE chunking: chunk both events and new_events
    total_sum = 0.0
    num_chunks_1 = int(np.ceil(n1 / chunk_size))
    num_chunks_2 = int(np.ceil(n2 / chunk_size))

    for i in range(num_chunks_1):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, n1)
        events_chunk = events[:, start_i:end_i]  # shape: (4, chunk_size_i)

        for j in range(num_chunks_2):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n2)
            new_events_chunk = new_events[:, start_j:end_j]  # shape: (4, chunk_size_j)

            # Compute pairwise distances for this block (chunk_size_i × chunk_size_j)
            x_index = events_chunk[2, :][:, None] - new_events_chunk[2, :][None, :]
            y_index = events_chunk[3, :][:, None] - new_events_chunk[3, :][None, :]
            t_index = events_chunk[1, :][:, None] - new_events_chunk[1, :][None, :]

            # Gaussian kernel
            dist_matrix = np.exp(
                - x_index**2 / (2 * x_sigma**2)
                - y_index**2 / (2 * y_sigma**2)
                - t_index**2 / (2 * t_sigma**2)
            )

            # Accumulate sum
            total_sum += np.sum(dist_matrix)

            # Free memory immediately
            del x_index, y_index, t_index, dist_matrix

    inner_product = polarity_scale * total_sum
    return inner_product


def cubes_3d_kernel_distance(events, new_events, x_sigma, y_sigma, t_sigma):
    """
    extern
    
    Computing distance between spike cubes using inner product in RKHS.

    Inputs:
    -------
        events    - events include polarity, timestamp, x and y.
        new_events    - events after changing operation.
        x_sigma, y_sigma, t_sigma  - the parameters of 3d gaussian kernel.

    Outputs:
    -------
        distance    - the distance between events and new_events.

    """

    if len(np.transpose(events)) <= 5 or len(np.transpose(new_events)) <= 5:
        distance = 0
    else:

        distance = cubes_3d_kernel_method(events, events, x_sigma, y_sigma, t_sigma)
        distance += cubes_3d_kernel_method(new_events, new_events, x_sigma, y_sigma, t_sigma)
        distance -= 2 * cubes_3d_kernel_method(events, new_events, x_sigma, y_sigma, t_sigma)

    return distance

def events_to_spike_cubes(events, width, height, x_cube_size, y_cube_size, t_cube_size):
    """
    extern

    events are split into spike cubes (VECTORIZED VERSION for 10x+ speedup).

    Inputs:
    -------
        events    - the dataset of AER sensor including polarity(t), timestamp(ts), x coordinate(X) and y coordinate(Y).
        width, height    - the width and height resolutions of dynamic vision sensor.
        x_cube_size, y_cube_size, t_cube_size    - the width, height and temporal size of spike cubes.
   Outputs:
    -------
        events_cubes    - the cubes of events (list of arrays).

    """
    # Vectorized index calculation - avoid Python loops
    x_idx = (events[2, :] // x_cube_size).astype(int)
    y_idx = (events[3, :] // y_cube_size).astype(int)
    t_idx = (events[1, :] // t_cube_size).astype(int)

    # Linear index for each event
    x_bins = int(width / x_cube_size)
    y_bins = int(height / y_cube_size)
    linear_idx = x_idx + y_idx * x_bins + t_idx * x_bins * y_bins

    # Determine number of cubes
    num = int((width/x_cube_size)*(height/y_cube_size)*(math.ceil(np.max(events[1, :]) / t_cube_size)))
    events_cube = [[] for _ in range(num)]

    # Group events by cube index (vectorized)
    unique_cubes = np.unique(linear_idx)
    for cube_id in unique_cubes:
        mask = (linear_idx == cube_id)
        events_cube[cube_id] = events[:, mask].T.tolist()  # Convert to list of events

    return events_cube

## this is the new loss!
def kernel_method_spike_cubes_loss(events, new_events, width=128, height=128, x_cube_size=32, y_cube_size=32, t_cube_size=5000, x_sigma=5, y_sigma=5, t_sigma=5000):
    """ 
        extern
                
        change some code to use EVK3 data as input
        
        3d gaussian kernel method  for spike cubes, such as polarity independent and polarity interference.

        Inputs:
        -------
            events    - events include polarity, timestamp, x and y.
            new_events    - events after changing operation.
            width, height  - the width and height of dynamic vision sensor.
            x_cube_size, y_cube_size, t_cube_size  - the size of spike cube.
            x_sigma, y_sigma, t_sigma  - the 3d gaussian kernel parameters.

        Outputs:
        -------
            distance    - the distance between events and new_events.

    """
    ##ADD IT TO fix dimension bug
    evs1_float = np.zeros((4,events.shape[0]), dtype=np.float64)
    evs2_float = np.zeros((4,new_events.shape[0]), dtype=np.float64)
    
    evs1_float[0, :] = events['p']
    evs1_float[1, :] = events['t']
    evs1_float[2, :] = events['x']
    evs1_float[3, :] = events['y']
    
    evs2_float[0, :] = new_events['p']
    evs2_float[1, :] = new_events['t']
    evs2_float[2, :] = new_events['x']
    evs2_float[3, :] = new_events['y']
    #t_intervel = evs2_float[1, :][-1]-evs2_float[1, :][0]+evs1_float[1, :][-1]-evs1_float[1, :][0]
    t_intervel = len(evs2_float[1, :])+len(evs1_float[1, :])
    # evs1_float = np.transpose(evs1_float)
    # evs2_float = np.transpose(evs2_float)

    events_cube = events_to_spike_cubes(evs1_float, width, height, x_cube_size, y_cube_size, t_cube_size)
    new_events_cubes = events_to_spike_cubes(evs2_float, width, height, x_cube_size, y_cube_size, t_cube_size)

    distance = 0
    for k in range(0, min(len(events_cube), len(new_events_cubes))):

        events_data = np.transpose(np.array(events_cube[k]))
        new_events_data = np.transpose(np.array(new_events_cubes[k]))

        if len(events_data)==0 and len(new_events_data)==0:
            distance += 0

        else:
            distance += cubes_3d_kernel_distance(events_data, new_events_data, x_sigma, y_sigma, t_sigma)
    #it's better to return distance/T_intervel
    return distance/t_intervel
    # change to [0  - 1]


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

# ============================================================================
# Kernel Method Spike Cubes Loss Variants
# ============================================================================

def _create_kernel_loss_wrapper(width, height, x_cube_size, y_cube_size, t_cube_size,
                                  x_sigma, y_sigma, t_sigma):
    """Factory function to create kernel loss variant with specific parameters."""
    def wrapper(events_est, events_gt):
        # Auto-detect resolution if not specified
        w = width if width else int(events_est['x'].max()) + 1
        h = height if height else int(events_est['y'].max()) + 1

        return kernel_method_spike_cubes_loss(
            events_est, events_gt,
            width=w, height=h,
            x_cube_size=x_cube_size, y_cube_size=y_cube_size, t_cube_size=t_cube_size,
            x_sigma=x_sigma, y_sigma=y_sigma, t_sigma=t_sigma
        )
    return wrapper

# Variant 1: kernel_standard - Standard configuration for balanced spatial-temporal matching
# Purpose: Default baseline with moderate sensitivity to both spatial and temporal errors
# Use case: General-purpose evaluation, balanced view of reconstruction quality
register_metric(
    'kernel_standard',
    _create_kernel_loss_wrapper(
        width=640, height=480,  # EVK4 resolution
        x_cube_size=32, y_cube_size=32, t_cube_size=5000,  # 32x32 spatial, 5ms temporal
        x_sigma=5, y_sigma=5, t_sigma=5000  # Moderate Gaussian kernel width
    ),
    'RKHS kernel loss - standard config (32x32x5ms cubes, σ=5/5/5000) for balanced spatial-temporal matching',
    'kernel'
)

# Variant 2: kernel_fine - Fine-grained configuration for detailed local structure
# Purpose: High spatial-temporal resolution, sensitive to fine details and local errors
# Use case: When precise event reconstruction is critical (e.g., sharp edges, fast motion)
register_metric(
    'kernel_fine',
    _create_kernel_loss_wrapper(
        width=640, height=480,
        x_cube_size=16, y_cube_size=16, t_cube_size=2000,  # 16x16 spatial, 2ms temporal
        x_sigma=3, y_sigma=3, t_sigma=2000  # Tighter Gaussian kernel
    ),
    'RKHS kernel loss - fine config (16x16x2ms cubes, σ=3/3/2000) for detailed local structure evaluation',
    'kernel'
)

# Variant 3: kernel_spatial - Spatial-dominant configuration
# Purpose: Emphasize spatial accuracy over temporal precision
# Use case: When spatial reconstruction quality is more important than timing accuracy
#           (e.g., evaluating flare removal effectiveness in spatial domain)
register_metric(
    'kernel_spatial',
    _create_kernel_loss_wrapper(
        width=640, height=480,
        x_cube_size=32, y_cube_size=32, t_cube_size=10000,  # Larger temporal cubes (10ms)
        x_sigma=3, y_sigma=3, t_sigma=10000  # Tight spatial, loose temporal kernel
    ),
    'RKHS kernel loss - spatial-focused (32x32x10ms cubes, σ=3/3/10000) emphasizing spatial accuracy',
    'kernel'
)

# Variant 4: kernel_temporal - Temporal-dominant configuration
# Purpose: Emphasize temporal precision over spatial accuracy
# Use case: When timing accuracy is critical (e.g., high-speed motion, temporal synchronization)
#           Useful for evaluating temporal alignment and event timing preservation
register_metric(
    'kernel_temporal',
    _create_kernel_loss_wrapper(
        width=640, height=480,
        x_cube_size=64, y_cube_size=64, t_cube_size=2000,  # Larger spatial cubes, smaller temporal
        x_sigma=10, y_sigma=10, t_sigma=1000  # Loose spatial, tight temporal kernel
    ),
    'RKHS kernel loss - temporal-focused (64x64x2ms cubes, σ=10/10/1000) emphasizing temporal precision',
    'kernel'
)