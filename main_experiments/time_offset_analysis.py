#!/usr/bin/env python3
"""
Time Offset Analysis Script

This script performs a specialized temporal alignment analysis between two event streams.
It compares the first 0.1s of clean data (no flare) with the first 0.1s of flare data
at various time offsets (0s, 0.001s, 0.002s, ..., 0.099s) to measure the impact
of temporal misalignment on downstream metrics.

Usage:
    python time_offset_analysis.py --clean_file path/to/clean.aedat4 --flare_file path/to/flare.aedat4
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Import our framework modules
from data_loader import load_events
from metrics import calculate_all_metrics


def extract_time_window(events: np.ndarray, start_time: float, duration: float) -> np.ndarray:
    """
    Extract events within a specific time window.
    
    Args:
        events (np.ndarray): Event data as structured array
        start_time (float): Start time in seconds
        duration (float): Duration in seconds
        
    Returns:
        np.ndarray: Events within the specified time window
    """
    end_time = start_time + duration
    mask = (events['t'] >= start_time) & (events['t'] < end_time)
    return events[mask].copy()


def shift_timestamps(events: np.ndarray, offset: float) -> np.ndarray:
    """
    Shift all timestamps by a given offset.
    
    Args:
        events (np.ndarray): Event data as structured array
        offset (float): Time offset in seconds
        
    Returns:
        np.ndarray: Events with shifted timestamps
    """
    events_shifted = events.copy()
    events_shifted['t'] = events_shifted['t'] + offset
    return events_shifted


def run_time_offset_analysis(clean_file_path: str, 
                            flare_file_path: str,
                            analysis_duration: float = 0.1,
                            offset_step: float = 0.001,
                            max_offset: float = 0.099,
                            output_dir: str = "time_offset_results") -> pd.DataFrame:
    """
    Run time offset analysis between clean and flare data.
    
    Args:
        clean_file_path (str): Path to clean (no flare) event data
        flare_file_path (str): Path to flare event data
        analysis_duration (float): Duration to analyze in seconds (default: 0.1s)
        offset_step (float): Step size for time offsets in seconds (default: 0.001s)
        max_offset (float): Maximum offset to test in seconds (default: 0.099s)
        output_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: Results of all offset analyses
    """
    print("="*80)
    print("TIME OFFSET ANALYSIS")
    print("="*80)
    print(f"Clean data: {clean_file_path}")
    print(f"Flare data: {flare_file_path}")
    print(f"Analysis duration: {analysis_duration}s")
    print(f"Offset range: 0s to {max_offset}s (step: {offset_step}s)")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading event data...")
    start_time = time.time()
    
    print("  Loading clean data...")
    events_clean = load_events(clean_file_path)
    
    print("  Loading flare data...")
    events_flare = load_events(flare_file_path)
    
    load_time = time.time() - start_time
    print(f"  Data loading completed in {load_time:.2f}s")
    print(f"  Clean events: {len(events_clean):,}")
    print(f"  Flare events: {len(events_flare):,}")
    
    # Extract base time windows (first 0.1s of each dataset)
    print(f"\nExtracting base time windows ({analysis_duration}s)...")
    clean_window = extract_time_window(events_clean, 0.0, analysis_duration)
    flare_window = extract_time_window(events_flare, 0.0, analysis_duration)
    
    print(f"  Clean window events: {len(clean_window):,}")
    print(f"  Flare window events: {len(flare_window):,}")
    
    if len(clean_window) == 0 or len(flare_window) == 0:
        raise ValueError("One of the time windows contains no events. Check your data and time range.")
    
    # Generate offset values
    offsets = np.arange(0.0, max_offset + offset_step/2, offset_step)
    print(f"\nGenerating {len(offsets)} offset comparisons...")
    
    results = []
    analysis_start_time = time.time()
    
    for i, offset in enumerate(offsets):
        if i % 10 == 0:  # Progress update every 10 iterations
            print(f"  Processing offset {i+1}/{len(offsets)}: {offset:.3f}s")
        
        try:
            # Shift the flare window by the current offset
            flare_shifted = shift_timestamps(flare_window, offset)
            
            # Calculate metrics between clean and shifted flare data
            metrics = calculate_all_metrics(clean_window, flare_shifted)
            
            # Add offset information to results
            result_record = {
                'offset_seconds': offset,
                'offset_milliseconds': offset * 1000,
                'clean_event_count': len(clean_window),
                'flare_event_count': len(flare_shifted),
                **metrics
            }
            results.append(result_record)
            
        except Exception as e:
            print(f"    Warning: Failed to process offset {offset:.3f}s: {e}")
            # Add failed record with NaN values
            result_record = {
                'offset_seconds': offset,
                'offset_milliseconds': offset * 1000,
                'clean_event_count': len(clean_window),
                'flare_event_count': len(flare_window),
                'error': str(e)
            }
            results.append(result_record)
    
    analysis_time = time.time() - analysis_start_time
    print(f"\nAnalysis completed in {analysis_time:.2f}s")
    
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    # Save results
    csv_path = Path(output_dir) / "time_offset_analysis_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    if 'chamfer_distance' in results_df.columns:
        chamfer_stats = results_df['chamfer_distance'].describe()
        print("Chamfer Distance Statistics:")
        print(f"  Mean: {chamfer_stats['mean']:.6f}")
        print(f"  Std:  {chamfer_stats['std']:.6f}")
        print(f"  Min:  {chamfer_stats['min']:.6f}")
        print(f"  Max:  {chamfer_stats['max']:.6f}")
        
        # Find best and worst offsets
        best_offset = results_df.loc[results_df['chamfer_distance'].idxmin(), 'offset_seconds']
        worst_offset = results_df.loc[results_df['chamfer_distance'].idxmax(), 'offset_seconds']
        print(f"  Best alignment: {best_offset:.3f}s")
        print(f"  Worst alignment: {worst_offset:.3f}s")
    
    if 'gaussian_distance' in results_df.columns:
        gaussian_stats = results_df['gaussian_distance'].describe()
        print("\nGaussian Distance Statistics:")
        print(f"  Mean: {gaussian_stats['mean']:.6f}")
        print(f"  Std:  {gaussian_stats['std']:.6f}")
        print(f"  Min:  {gaussian_stats['min']:.6f}")
        print(f"  Max:  {gaussian_stats['max']:.6f}")
    
    # Calculate variability impact
    if 'chamfer_distance' in results_df.columns:
        min_chamfer = results_df['chamfer_distance'].min()
        max_chamfer = results_df['chamfer_distance'].max()
        variability_impact = (max_chamfer - min_chamfer) / min_chamfer * 100
        print(f"\nTemporal misalignment impact on Chamfer Distance: {variability_impact:.2f}%")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f}s")
    
    return results_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Time Offset Analysis for Event Data")
    parser.add_argument('--clean_file', type=str, required=True,
                       help='Path to clean (no flare) event data file')
    parser.add_argument('--flare_file', type=str, required=True,
                       help='Path to flare event data file')
    parser.add_argument('--duration', type=float, default=0.1,
                       help='Analysis duration in seconds (default: 0.1)')
    parser.add_argument('--offset_step', type=float, default=0.001,
                       help='Time offset step in seconds (default: 0.001)')
    parser.add_argument('--max_offset', type=float, default=0.099,
                       help='Maximum offset in seconds (default: 0.099)')
    parser.add_argument('--output_dir', type=str, default='time_offset_results',
                       help='Output directory (default: time_offset_results)')
    
    args = parser.parse_args()
    
    # Convert Windows paths to Unix format if needed
    clean_file = args.clean_file.replace('\\', '/')
    flare_file = args.flare_file.replace('\\', '/')
    
    try:
        results_df = run_time_offset_analysis(
            clean_file_path=clean_file,
            flare_file_path=flare_file,
            analysis_duration=args.duration,
            offset_step=args.offset_step,
            max_offset=args.max_offset,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {args.output_dir}/")
        print("You can now analyze the CSV file for temporal alignment sensitivity.")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())