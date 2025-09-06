#!/usr/bin/env python3
"""
Quick test script for H5 evaluation functionality.
Tests a single pair to verify the implementation works.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_events
from metrics import calculate_all_metrics

def test_single_pair():
    """Test loading and evaluating a single H5 pair."""
    
    # Test with the first available pair
    flare_dir = Path("Datasets/simu/background_with_flare_events_testoutput")
    gt_dir = Path("Datasets/simu/background_with_light_events_test")
    
    # Find first available pair
    flare_files = list(flare_dir.glob("*.h5"))
    if not flare_files:
        print("ERROR: No H5 files found in flare directory")
        return False
    
    flare_file = flare_files[0]
    sample_id = flare_file.stem.replace("_bg_flare", "")
    gt_file = gt_dir / f"{sample_id}_bg_light.h5"
    
    if not gt_file.exists():
        print(f"ERROR: No matching ground truth file for {sample_id}")
        return False
    
    print(f"Testing with pair: {sample_id}")
    print(f"Flare file: {flare_file}")
    print(f"GT file: {gt_file}")
    print()
    
    try:
        # Test loading flare events
        print("Loading flare events...")
        events_flare = load_events(str(flare_file))
        print(f"Loaded {len(events_flare):,} flare events")
        
        # Test loading GT events  
        print("Loading ground truth events...")
        events_gt = load_events(str(gt_file))
        print(f"Loaded {len(events_gt):,} ground truth events")
        
        # Test metrics calculation
        print("Computing metrics...")
        metrics = calculate_all_metrics(events_flare, events_gt)
        
        print("\nResults:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_pair()
    sys.exit(0 if success else 1)