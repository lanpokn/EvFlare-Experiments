#!/usr/bin/env python3
"""
Test script for AEDAT4 loading and basic metrics computation.
This script tests the AEDAT4 loading functionality with your provided data files.
"""

import sys
from pathlib import Path
import numpy as np

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_events
from metrics import calculate_all_metrics


def test_aedat4_loading():
    """Test AEDAT4 file loading with the provided file paths."""
    
    # Convert Windows paths to Unix format
    clean_file = "E:/2025/event_flick_flare/datasets/DAVIS/test/full_noFlare-2025_06_25_14_50_52.aedat4"
    flare_file = "E:/2025/event_flick_flare/datasets/DAVIS/test/full_randomFlare-2025_06_25_14_52_42.aedat4"
    
    # Convert to Unix paths
    clean_file = clean_file.replace('\\', '/').replace('E:', '/mnt/e')
    flare_file = flare_file.replace('\\', '/').replace('E:', '/mnt/e')
    
    print("="*60)
    print("AEDAT4 Loading Test")
    print("="*60)
    print(f"Clean file: {clean_file}")
    print(f"Flare file: {flare_file}")
    
    try:
        print("\nLoading clean data...")
        events_clean = load_events(clean_file)
        print(f"✓ Clean data loaded: {len(events_clean):,} events")
        if len(events_clean) > 0:
            print(f"  Time range: {events_clean['t'].min():.3f}s to {events_clean['t'].max():.3f}s")
            print(f"  Spatial range: X[{events_clean['x'].min()}, {events_clean['x'].max()}], Y[{events_clean['y'].min()}, {events_clean['y'].max()}]")
            print(f"  Polarity distribution: {np.bincount(events_clean['p'])}")
        
        print("\nLoading flare data...")
        events_flare = load_events(flare_file)
        print(f"✓ Flare data loaded: {len(events_flare):,} events")
        if len(events_flare) > 0:
            print(f"  Time range: {events_flare['t'].min():.3f}s to {events_flare['t'].max():.3f}s")
            print(f"  Spatial range: X[{events_flare['x'].min()}, {events_flare['x'].max()}], Y[{events_flare['y'].min()}, {events_flare['y'].max()}]")
            print(f"  Polarity distribution: {np.bincount(events_flare['p'])}")
        
        if len(events_clean) > 0 and len(events_flare) > 0:
            print("\nTesting metrics calculation on first 10,000 events...")
            
            # Take first 10k events for quick test
            test_clean = events_clean[:min(10000, len(events_clean))]
            test_flare = events_flare[:min(10000, len(events_flare))]
            
            print(f"Computing metrics between {len(test_clean):,} and {len(test_flare):,} events...")
            
            metrics = calculate_all_metrics(test_clean, test_flare)
            
            print("✓ Metrics calculated successfully:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "="*60)
        print("✅ AEDAT4 loading test completed successfully!")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install the aedat library: pip install aedat")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please check that the file paths are correct and accessible from WSL")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_aedat4_loading()
    sys.exit(0 if success else 1)