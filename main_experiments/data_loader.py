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
        Loads data from an .aedat4 file.
        
        TODO: Implement using dv-processing library or similar.
        For now, this is a placeholder that needs actual implementation.
        """
        print(f"Loading from AEDAT4: {self.file_path}")
        
        # TODO: Actual implementation needed
        # Example structure:
        # import dv_processing as dv
        # reader = dv.io.MonoCameraReader(self.file_path)
        # events = []
        # for packet in reader:
        #     events.extend(packet.elements)
        # 
        # # Convert to structured array
        # dtype = [('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
        # events_array = np.array([(e.timestamp(), e.x(), e.y(), e.polarity()) 
        #                         for e in events], dtype=dtype)
        # return events_array
        
        raise NotImplementedError("AEDAT4 loading not yet implemented")


class H5DataSource(DataSource):
    """Loads event data from HDF5 files."""
    
    def _load_data(self) -> np.ndarray:
        """
        Loads data from an HDF5 file.
        
        TODO: Implement using h5py library.
        For now, this is a placeholder that needs actual implementation.
        """
        print(f"Loading from H5: {self.file_path}")
        
        # TODO: Actual implementation needed
        # Example structure:
        # import h5py
        # with h5py.File(self.file_path, 'r') as f:
        #     # Assume standard event dataset structure
        #     t = f['events/t'][:]
        #     x = f['events/x'][:]
        #     y = f['events/y'][:]
        #     p = f['events/p'][:]
        # 
        # dtype = [('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
        # events_array = np.array(list(zip(t, x, y, p)), dtype=dtype)
        # return events_array
        
        raise NotImplementedError("H5 loading not yet implemented")


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