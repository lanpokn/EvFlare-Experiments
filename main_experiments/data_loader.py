"""
Event Data Loading Module

This module provides abstract and concrete implementations for loading event data
from various file formats. All data is standardized to NumPy structured arrays
with fields ('t', 'x', 'y', 'p').
"""

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path


class DataSource(ABC):
    """
    Abstract base class for loading event data from a file.
    The loaded data is stored as a NumPy structured array.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._events = None

    @abstractmethod
    def _load_data(self) -> np.ndarray:
        """
        Private method to be implemented by subclasses for specific file formats.
        Should return a NumPy structured array with fields ('t', 'x', 'y', 'p').
        
        Returns:
            np.ndarray: Structured array with dtype=[('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
        """
        pass

    def get_events(self) -> np.ndarray:
        """
        Public method to access the event data. Loads data on first call.
        
        Returns:
            np.ndarray: Event data as structured array
        """
        if self._events is None:
            self._events = self._load_data()
        return self._events


class Aedat4DataSource(DataSource):
    """Loads event data from AEDAT4 files (DVS camera format)."""
    
    def _load_data(self) -> np.ndarray:
        """
        Loads data from an .aedat4 file using aedat library.
        Based on the reference code provided by user.
        """
        print(f"Loading from AEDAT4: {self.file_path}")
        
        try:
            import aedat
        except ImportError:
            raise ImportError("aedat library not found. Please install with 'pip install aedat'")
        
        xs, ys, ts, ps = [], [], [], []
        
        # Read events from AEDAT4 file
        decoder = aedat.Decoder(self.file_path)
        for packet in decoder:
            if 'events' in packet:
                ev = packet['events']
                xs.append(ev['x'])
                ys.append(ev['y'])
                ts.append(ev['t'])
                ps.append(ev['p'])
        
        if not ts:
            print(f"Warning: No events found in {self.file_path}")
            # Return empty structured array with correct dtype
            dtype = [('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
            return np.array([], dtype=dtype)
        
        # Flatten and convert arrays
        ts_flat = np.concatenate(ts).astype(np.float64)
        xs_flat = np.concatenate(xs).astype(np.uint16)  
        ys_flat = np.concatenate(ys).astype(np.uint16)
        ps_flat = np.concatenate(ps).astype(np.int8)
        
        # Convert timestamps from microseconds to seconds
        ts_flat = ts_flat / 1e6
        
        # Align timestamps to start from 0
        if len(ts_flat) > 0:
            min_ts = ts_flat.min()
            ts_flat = ts_flat - min_ts
        
        # Create structured array matching our standard format
        dtype = [('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
        events_array = np.zeros(len(ts_flat), dtype=dtype)
        events_array['t'] = ts_flat
        events_array['x'] = xs_flat
        events_array['y'] = ys_flat
        events_array['p'] = ps_flat
        
        print(f"Loaded {len(events_array):,} events from AEDAT4 file")
        print(f"Time range: {events_array['t'].min():.3f}s to {events_array['t'].max():.3f}s")
        
        return events_array


class H5DataSource(DataSource):
    """Loads event data from HDF5 files."""
    
    def _load_data(self) -> np.ndarray:
        """
        Loads data from an HDF5 file with format events/t, events/x, events/y, events/p.
        Based on the user's H5 format specification.
        """
        print(f"Loading from H5: {self.file_path}")
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py library not found. Please install with 'pip install h5py'")
        
        try:
            with h5py.File(self.file_path, 'r') as f:
                # Read the four data arrays from events group
                t = np.array(f['events']['t'])  # Timestamp (microseconds)
                x = np.array(f['events']['x'])  # X coordinates  
                y = np.array(f['events']['y'])  # Y coordinates
                p = np.array(f['events']['p'])  # Polarity
                
                # Convert timestamps from microseconds to seconds
                t = t.astype(np.float64) / 1e6
                
                # Align timestamps to start from 0
                if len(t) > 0:
                    min_ts = t.min()
                    t = t - min_ts
                
                # Handle polarity: 1→+1, non-1→-1 (as per user specification)
                p_processed = np.where(p == 1, 1, -1).astype(np.int8)
                
                # Create structured array matching our standard format
                dtype = [('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
                events_array = np.zeros(len(t), dtype=dtype)
                events_array['t'] = t
                events_array['x'] = x.astype(np.uint16)
                events_array['y'] = y.astype(np.uint16) 
                events_array['p'] = p_processed
                
                print(f"Loaded {len(events_array):,} events from H5 file")
                print(f"Time range: {events_array['t'].min():.3f}s to {events_array['t'].max():.3f}s")
                print(f"Spatial range: X[{events_array['x'].min()}, {events_array['x'].max()}], Y[{events_array['y'].min()}, {events_array['y'].max()}]")
                print(f"Polarity distribution: +1={np.sum(events_array['p'] == 1):,}, -1={np.sum(events_array['p'] == -1):,}")
                
                return events_array
                
        except Exception as e:
            raise RuntimeError(f"Failed to load H5 file {self.file_path}: {e}")


class NpyDataSource(DataSource):
    """Loads event data from NumPy .npy files (for testing/debugging)."""
    
    def _load_data(self) -> np.ndarray:
        """Loads data from a .npy file."""
        print(f"Loading from NPY: {self.file_path}")
        events = np.load(self.file_path)
        
        # Ensure correct dtype
        if events.dtype.names != ('t', 'x', 'y', 'p'):
            # Try to convert if possible
            if len(events.shape) == 2 and events.shape[1] >= 4:
                dtype = [('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
                events = np.array([tuple(row[:4]) for row in events], dtype=dtype)
        
        return events


def load_events(file_path: str) -> np.ndarray:
    """
    Factory function that takes a file path, determines the format,
    and returns the event data as a NumPy structured array.
    
    Args:
        file_path (str): Path to the event data file
        
    Returns:
        np.ndarray: Event data as structured array with fields ('t', 'x', 'y', 'p')
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.aedat4':
        loader = Aedat4DataSource(file_path)
    elif suffix == '.h5':
        loader = H5DataSource(file_path)
    elif suffix == '.npy':
        loader = NpyDataSource(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    return loader.get_events()