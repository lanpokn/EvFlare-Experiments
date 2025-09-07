"""
Voxel Conversion Utilities for Event Data

This module provides efficient event-to-voxel conversion following the user's specifications:
- Fixed 20ms time windows for consistency
- Vectorized implementation for performance  
- Memory-efficient chunk processing for 100ms files
- Simple polarity accumulation (+1/-1)

Key Design:
- Split 100ms files into 5×20ms chunks to avoid memory issues
- Each chunk generates one voxel tensor (T, H, W)
- Voxel-based metrics computed per chunk then averaged
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
import warnings


def events_to_voxel(events_np: np.ndarray, 
                   num_bins: int = 8, 
                   sensor_size: Tuple[int, int] = (480, 640),
                   fixed_duration_us: float = 20000.0) -> torch.Tensor:
    """
    Convert events to voxel representation with fixed time window.
    
    Args:
        events_np (np.ndarray): Events structured array with fields ('t', 'x', 'y', 'p')
                               Time should be in seconds (will be converted to microseconds)
        num_bins (int): Number of temporal bins (default: 8)
        sensor_size (Tuple[int, int]): (height, width) of sensor (default: DVS camera)
        fixed_duration_us (float): Fixed time window in microseconds (default: 20000 = 20ms)
        
    Returns:
        torch.Tensor: Voxel grid of shape (num_bins, height, width)
        
    Note:
        This implements the user's exact specifications:
        - Fixed 20ms time windows for training consistency
        - Simple polarity accumulation
        - Vectorized PyTorch implementation
    """
    if len(events_np) == 0:
        return torch.zeros((num_bins, sensor_size[0], sensor_size[1]), dtype=torch.float32)
    
    # Initialize voxel grid
    voxel = torch.zeros((num_bins, sensor_size[0], sensor_size[1]), dtype=torch.float32)
    
    # Extract event components - follow user's exact specification
    # User expects input format: events_np as (N, 4) array [timestamp, x, y, polarity]
    if events_np.dtype.names is not None:
        # Structured array format (our current format)
        ts = events_np['t']  # Already in seconds, convert to microseconds
        ts_us = ts * 1e6
        xs = events_np['x'].astype(int)
        ys = events_np['y'].astype(int)  
        ps = events_np['p'].astype(float)
    else:
        # Plain array format (N, 4) as specified by user
        ts = events_np[:, 0]  # timestamp
        xs = events_np[:, 1].astype(int)  # x coordinate  
        ys = events_np[:, 2].astype(int)  # y coordinate
        ps = events_np[:, 3].astype(float)  # polarity
        
        # Convert time to microseconds if needed (assume seconds if < 1000, else already microseconds)
        if ts.max() < 1000:
            ts_us = ts * 1e6
        else:
            ts_us = ts
    
    # Fixed time binning (key design decision)
    t_min = ts_us.min()
    dt = fixed_duration_us / num_bins  # Fixed bin duration: 2.5ms per bin
    bin_indices = np.clip(((ts_us - t_min) / dt).astype(int), 0, num_bins - 1)
    
    # Spatial boundary filtering
    valid_mask = (xs >= 0) & (xs < sensor_size[1]) & \
                 (ys >= 0) & (ys < sensor_size[0])
    
    # Vectorized accumulation (pure PyTorch to avoid memory leaks)
    if valid_mask.any():
        valid_bins = bin_indices[valid_mask]
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask] 
        valid_ps = ps[valid_mask]
        
        # Convert to PyTorch tensors
        bins_tensor = torch.from_numpy(valid_bins).long()
        xs_tensor = torch.from_numpy(valid_xs).long()
        ys_tensor = torch.from_numpy(valid_ys).long()
        ps_tensor = torch.from_numpy(valid_ps).float()
        
        # Linear index accumulation for efficiency
        linear_indices = bins_tensor * (sensor_size[0] * sensor_size[1]) + \
                        ys_tensor * sensor_size[1] + xs_tensor
        
        voxel_1d = voxel.view(-1)
        voxel_1d.index_add_(0, linear_indices, ps_tensor)
    
    return voxel


def split_events_by_time(events_np: np.ndarray, 
                        chunk_duration_us: float = 20000.0) -> List[np.ndarray]:
    """
    Split events into time chunks for voxel processing.
    
    Args:
        events_np (np.ndarray): Events structured array 
        chunk_duration_us (float): Duration of each chunk in microseconds
        
    Returns:
        List[np.ndarray]: List of event chunks, each covering chunk_duration_us
    """
    if len(events_np) == 0:
        return []
    
    chunks = []
    ts_us = events_np['t'] * 1e6  # Convert to microseconds
    t_start = ts_us.min()
    t_end = ts_us.max()
    total_duration = t_end - t_start
    
    # Calculate number of chunks needed
    num_chunks = int(np.ceil(total_duration / chunk_duration_us))
    
    for i in range(num_chunks):
        chunk_start = t_start + i * chunk_duration_us
        chunk_end = chunk_start + chunk_duration_us
        
        # Find events in this time window  
        mask = (ts_us >= chunk_start) & (ts_us < chunk_end)
        
        if mask.any():
            chunk_events = events_np[mask]
            chunks.append(chunk_events)
    
    return chunks


def events_to_voxel_chunks(events_np: np.ndarray,
                          num_bins: int = 8,
                          sensor_size: Tuple[int, int] = (480, 640),
                          chunk_duration_us: float = 20000.0) -> List[torch.Tensor]:
    """
    Convert full event stream to list of voxel chunks.
    
    This is the main function for processing 100ms files:
    - Splits events into 20ms chunks  
    - Converts each chunk to voxel
    - Returns list of voxel tensors for metric computation
    
    Args:
        events_np (np.ndarray): Full events array (e.g., 100ms)
        num_bins (int): Temporal bins per voxel
        sensor_size (Tuple[int, int]): Sensor dimensions
        chunk_duration_us (float): Chunk duration in microseconds
        
    Returns:
        List[torch.Tensor]: List of voxel tensors, each of shape (num_bins, H, W)
    """
    # Split events into time chunks
    event_chunks = split_events_by_time(events_np, chunk_duration_us)
    
    # Convert each chunk to voxel
    voxel_chunks = []
    for chunk in event_chunks:
        voxel = events_to_voxel(chunk, num_bins, sensor_size, chunk_duration_us)
        voxel_chunks.append(voxel)
    
    return voxel_chunks


def get_voxel_info(voxel: torch.Tensor) -> dict:
    """
    Get diagnostic information about a voxel tensor.
    
    Args:
        voxel (torch.Tensor): Voxel tensor of shape (T, H, W)
        
    Returns:
        dict: Diagnostic information including shape, value ranges, etc.
    """
    info = {
        'shape': tuple(voxel.shape),
        'dtype': str(voxel.dtype),
        'total_events': float(torch.abs(voxel).sum()),
        'positive_events': float((voxel > 0).sum()),
        'negative_events': float((voxel < 0).sum()),
        'zero_bins': float((voxel == 0).sum()),
        'min_value': float(voxel.min()),
        'max_value': float(voxel.max()),
        'mean_abs_value': float(torch.abs(voxel).mean()),
    }
    
    # Per-temporal-bin statistics
    info['temporal_bin_stats'] = []
    for t in range(voxel.shape[0]):
        bin_slice = voxel[t]
        bin_info = {
            'bin_index': t,
            'total_events': float(torch.abs(bin_slice).sum()),
            'active_pixels': float((bin_slice != 0).sum()),
            'spatial_coverage': float((bin_slice != 0).sum()) / (bin_slice.shape[0] * bin_slice.shape[1])
        }
        info['temporal_bin_stats'].append(bin_info)
    
    return info


def validate_voxel_conversion(events_np: np.ndarray, 
                             voxel: torch.Tensor,
                             tolerance: float = 1e-6) -> bool:
    """
    Validate that voxel conversion preserved event counts.
    
    Args:
        events_np (np.ndarray): Original events
        voxel (torch.Tensor): Generated voxel
        tolerance (float): Numerical tolerance for comparison
        
    Returns:
        bool: True if validation passes
    """
    # Count original events by polarity
    if len(events_np) == 0:
        return torch.abs(voxel).sum() < tolerance
    
    original_positive = np.sum(events_np['p'] > 0)
    original_negative = np.sum(events_np['p'] < 0)
    original_total = len(events_np)
    
    # Count voxel events
    voxel_positive = float((voxel > 0).sum())
    voxel_negative = float((voxel < 0).sum())
    voxel_total = float(torch.abs(voxel).sum())
    
    # Check conservation (allowing for spatial filtering)
    positive_ok = abs(voxel_positive - original_positive) <= original_positive * 0.01  # 1% tolerance
    negative_ok = abs(voxel_negative - abs(original_negative)) <= abs(original_negative) * 0.01
    
    if not (positive_ok and negative_ok):
        warnings.warn(f"Event count mismatch: "
                     f"Original (+{original_positive}, -{abs(original_negative)}) vs "
                     f"Voxel (+{voxel_positive}, -{voxel_negative})")
        return False
    
    return True