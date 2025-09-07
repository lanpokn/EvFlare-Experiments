#!/usr/bin/env python3
"""
Test Script for Voxel Conversion Functionality

Tests the voxel conversion implementation against user specifications:
- Fixed 20ms time windows
- Proper polarity accumulation
- Chunk-based processing for memory efficiency
"""

import sys
import os
import numpy as np
import torch

# Add current directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voxel_utils import (
    events_to_voxel, 
    split_events_by_time, 
    events_to_voxel_chunks,
    get_voxel_info,
    validate_voxel_conversion
)
from data_loader import load_events


def create_synthetic_events(duration_s=0.1, event_rate=10000, sensor_size=(480, 640)):
    """Create synthetic events for testing."""
    n_events = int(duration_s * event_rate)
    
    # Random timestamps within duration
    timestamps = np.sort(np.random.uniform(0, duration_s, n_events))
    
    # Random spatial coordinates
    xs = np.random.randint(0, sensor_size[1], n_events)
    ys = np.random.randint(0, sensor_size[0], n_events)
    
    # Random polarities
    polarities = np.random.choice([-1, 1], n_events)
    
    # Create structured array
    events = np.zeros(n_events, dtype=[('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')])
    events['t'] = timestamps
    events['x'] = xs
    events['y'] = ys  
    events['p'] = polarities
    
    return events


def test_basic_voxel_conversion():
    """Test basic voxel conversion functionality."""
    print("=" * 60)
    print("TEST 1: Basic Voxel Conversion")
    print("=" * 60)
    
    # Create simple test events
    events = create_synthetic_events(duration_s=0.02, event_rate=5000)  # 20ms, 100 events
    print(f"Created {len(events)} synthetic events over 20ms")
    
    # Convert to voxel
    voxel = events_to_voxel(events, num_bins=8, sensor_size=(480, 640))
    print(f"Voxel shape: {voxel.shape}")
    print(f"Voxel dtype: {voxel.dtype}")
    
    # Get detailed info
    info = get_voxel_info(voxel)
    print(f"Total events in voxel: {info['total_events']}")
    print(f"Positive events: {info['positive_events']}")
    print(f"Negative events: {info['negative_events']}")
    print(f"Active pixels: {info['total_events'] - info['zero_bins']}")
    
    # Validate conversion
    is_valid = validate_voxel_conversion(events, voxel)
    print(f"Validation result: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    return is_valid


def test_time_chunking():
    """Test time-based chunking for 100ms files."""
    print("\n" + "=" * 60)
    print("TEST 2: Time Chunking (100ms → 5×20ms)")
    print("=" * 60)
    
    # Create 100ms synthetic events
    events = create_synthetic_events(duration_s=0.1, event_rate=50000)  # 100ms, 5000 events
    print(f"Created {len(events)} synthetic events over 100ms")
    
    # Split into chunks
    chunks = split_events_by_time(events, chunk_duration_us=20000)  # 20ms chunks
    print(f"Split into {len(chunks)} chunks")
    
    total_events_in_chunks = sum(len(chunk) for chunk in chunks)
    print(f"Total events in chunks: {total_events_in_chunks} (original: {len(events)})")
    
    # Expected: 5 chunks for 100ms with 20ms windows
    expected_chunks = 5
    chunk_test_pass = len(chunks) == expected_chunks
    print(f"Expected chunks: {expected_chunks}, Got: {len(chunks)} {'✓' if chunk_test_pass else '✗'}")
    
    # Check chunk durations
    print("\nChunk statistics:")
    for i, chunk in enumerate(chunks):
        if len(chunk) > 0:
            duration = (chunk['t'].max() - chunk['t'].min()) * 1e6  # microseconds
            print(f"  Chunk {i+1}: {len(chunk)} events, duration: {duration:.1f} μs")
    
    return chunk_test_pass


def test_voxel_chunks_pipeline():
    """Test full pipeline: events → chunks → voxels."""
    print("\n" + "=" * 60) 
    print("TEST 3: Full Voxel Chunks Pipeline")
    print("=" * 60)
    
    # Create 100ms synthetic events
    events = create_synthetic_events(duration_s=0.1, event_rate=100000)  # 100ms, 10k events
    print(f"Input: {len(events)} events over 100ms")
    
    # Convert to voxel chunks
    voxel_chunks = events_to_voxel_chunks(events, num_bins=8, sensor_size=(480, 640))
    print(f"Generated {len(voxel_chunks)} voxel chunks")
    
    # Analyze each chunk
    all_valid = True
    total_voxel_events = 0
    
    for i, voxel in enumerate(voxel_chunks):
        info = get_voxel_info(voxel)
        print(f"  Chunk {i+1}: {info['total_events']:.0f} events, "
              f"shape={info['shape']}, active_pixels={info['positive_events'] + info['negative_events']:.0f}")
        total_voxel_events += info['total_events']
        
        # Validate each chunk if we have the corresponding events
        # (Note: We can't validate individual chunks easily since we'd need to re-split)
    
    print(f"Total events across all voxels: {total_voxel_events}")
    print(f"Original events: {len(events)}")
    
    # Check if we preserved roughly the right number of events
    preservation_ratio = total_voxel_events / len(events)
    preservation_ok = 0.95 <= preservation_ratio <= 1.05  # 5% tolerance
    print(f"Event preservation ratio: {preservation_ratio:.3f} {'✓' if preservation_ok else '✗'}")
    
    return preservation_ok


def test_real_data_voxelization():
    """Test voxelization with real H5 data if available."""
    print("\n" + "=" * 60)
    print("TEST 4: Real Data Voxelization")  
    print("=" * 60)
    
    # Try to load real data
    test_files = [
        "Datasets/simu/background_with_light_events_test/composed_00504_bg_light.h5",
        "Datasets/simu/background_with_flare_events_test/composed_00504_bg_flare.h5"
    ]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            continue
            
        print(f"\nTesting with: {test_file}")
        
        try:
            # Load events
            events = load_events(test_file)
            print(f"Loaded {len(events)} real events")
            
            # Get time span
            time_span = events['t'].max() - events['t'].min()
            print(f"Time span: {time_span:.3f}s ({time_span*1000:.1f}ms)")
            
            # Convert to voxel chunks
            voxel_chunks = events_to_voxel_chunks(events, num_bins=8)
            print(f"Generated {len(voxel_chunks)} voxel chunks")
            
            # Analyze first few chunks
            for i, voxel in enumerate(voxel_chunks[:3]):  # Just first 3
                info = get_voxel_info(voxel)
                print(f"  Real Chunk {i+1}: {info['total_events']:.0f} events, "
                      f"coverage={info['temporal_bin_stats'][0]['spatial_coverage']:.4f}")
            
            print(f"✓ Real data voxelization successful for {test_file}")
            return True
            
        except Exception as e:
            print(f"✗ Error with {test_file}: {e}")
            
    return False


def main():
    """Run all voxel conversion tests."""
    print("VOXEL CONVERSION TEST SUITE")
    print("Testing implementation against user specifications")
    
    results = []
    
    # Run all tests
    results.append(("Basic Voxel Conversion", test_basic_voxel_conversion()))
    results.append(("Time Chunking", test_time_chunking()))
    results.append(("Voxel Chunks Pipeline", test_voxel_chunks_pipeline()))
    results.append(("Real Data Voxelization", test_real_data_voxelization()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Voxel conversion ready for metric development.")
    else:
        print("⚠️  Some tests failed. Please review implementation.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)