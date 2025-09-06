#!/usr/bin/env python3
"""
Fast Batch Evaluation Script for Simulated H5 Event Pairs

Optimized version that processes only a subset of samples for quick testing.
This is useful for debugging and getting initial results quickly.

Usage:
    python evaluate_simu_pairs_fast.py [--num-samples 5] [--output results/]
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_events
from metrics import calculate_all_metrics


def find_paired_files(flare_dir: str, gt_dir: str, max_samples: int = None) -> List[Tuple[str, str, str]]:
    """
    Find paired H5 files, optionally limited to max_samples.
    """
    flare_path = Path(flare_dir)
    gt_path = Path(gt_dir)
    
    if not flare_path.exists():
        raise FileNotFoundError(f"Flare directory not found: {flare_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    
    # Get all flare files
    flare_files = {f.stem: f for f in flare_path.glob("*.h5")}
    gt_files = {f.stem: f for f in gt_path.glob("*.h5")}
    
    paired_files = []
    
    for flare_stem, flare_file in flare_files.items():
        if "_bg_flare" in flare_stem:
            sample_id = flare_stem.replace("_bg_flare", "")
            gt_stem = f"{sample_id}_bg_light"
            
            if gt_stem in gt_files:
                paired_files.append((
                    sample_id,
                    str(flare_file),
                    str(gt_files[gt_stem])
                ))
    
    paired_files.sort(key=lambda x: x[0])  # Sort by sample ID
    
    # Limit to max_samples if specified
    if max_samples is not None and len(paired_files) > max_samples:
        paired_files = paired_files[:max_samples]
        print(f"Limited to first {max_samples} samples for fast testing")
    
    return paired_files


def evaluate_single_pair(sample_id: str, flare_path: str, gt_path: str, verbose: bool = False) -> Dict:
    """Evaluate a single pair of H5 files."""
    try:
        if verbose:
            print(f"  Processing {sample_id}...")
            
        start_time = time.time()
        
        # Load both files
        load_start = time.time()
        events_flare = load_events(flare_path)
        events_gt = load_events(gt_path)
        load_time = time.time() - load_start
        
        if verbose:
            print(f"    Loaded: flare={len(events_flare):,}, gt={len(events_gt):,} events")
        
        # Calculate metrics
        metrics_start = time.time()
        metrics = calculate_all_metrics(events_flare, events_gt)
        metrics_time = time.time() - metrics_start
        
        total_time = time.time() - start_time
        
        # Add metadata
        result = {
            'sample_id': sample_id,
            'flare_file': flare_path,
            'gt_file': gt_path,
            'loading_time_s': load_time,
            'metrics_time_s': metrics_time,
            'total_time_s': total_time,
            **metrics
        }
        
        if verbose:
            print(f"    ✓ Completed in {total_time:.2f}s")
            print(f"    Chamfer: {metrics.get('chamfer_distance', 'N/A'):.4f}")
            print(f"    Gaussian: {metrics.get('gaussian_distance', 'N/A'):.4f}")
            
        return result
        
    except Exception as e:
        error_result = {
            'sample_id': sample_id,
            'flare_file': flare_path,
            'gt_file': gt_path,
            'error': str(e),
            'failed': True
        }
        
        if verbose:
            print(f"    ✗ Failed: {e}")
            
        return error_result


def run_batch_evaluation(flare_dir: str, gt_dir: str, max_samples: int = None, verbose: bool = True) -> pd.DataFrame:
    """Run batch evaluation on paired files."""
    if verbose:
        print("="*80)
        print("FAST BATCH EVALUATION OF SIMULATED H5 EVENT PAIRS")
        print("="*80)
    
    # Find paired files
    paired_files = find_paired_files(flare_dir, gt_dir, max_samples)
    
    if verbose:
        print(f"Found {len(paired_files)} paired files to process")
        print(f"  Flare directory: {flare_dir}")
        print(f"  Ground truth directory: {gt_dir}")
        print()
    
    # Evaluate each pair
    results = []
    batch_start_time = time.time()
    
    for i, (sample_id, flare_path, gt_path) in enumerate(paired_files):
        if verbose:
            print(f"[{i+1}/{len(paired_files)}] Evaluating {sample_id}")
            
        result = evaluate_single_pair(sample_id, flare_path, gt_path, verbose)
        results.append(result)
    
    batch_time = time.time() - batch_start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    if verbose:
        print()
        print("="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        failed_count = df.get('failed', pd.Series([False]*len(df))).sum()
        success_count = len(df) - failed_count
        
        print(f"Total pairs: {len(df)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {success_count/len(df)*100:.1f}%")
        print(f"Total time: {batch_time:.2f}s")
        print(f"Avg time per pair: {batch_time/len(df):.2f}s")
        
        if success_count > 0:
            success_df = df[df.get('failed', pd.Series([False]*len(df))) == False]
            
            print()
            print("METRIC STATISTICS:")
            print("-" * 50)
            
            for metric in ['chamfer_distance', 'gaussian_distance', 'event_count_ratio', 'temporal_overlap']:
                if metric in success_df.columns:
                    values = success_df[metric].dropna()
                    if len(values) > 0:
                        print(f"{metric}:")
                        print(f"  Mean: {values.mean():.6f}")
                        print(f"  Std:  {values.std():.6f}")
                        print(f"  Min:  {values.min():.6f}")
                        print(f"  Max:  {values.max():.6f}")
                        print(f"  Median: {values.median():.6f}")
                        print()
    
    return df


def save_results(df: pd.DataFrame, output_dir: str, verbose: bool = True) -> None:
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save complete results
    csv_path = output_path / "simu_pairs_fast_evaluation.csv"
    df.to_csv(csv_path, index=False)
    
    # Save summary
    summary_path = output_path / "simu_pairs_fast_summary.txt"
    
    failed_count = df.get('failed', pd.Series([False]*len(df))).sum()
    success_count = len(df) - failed_count
    
    with open(summary_path, 'w') as f:
        f.write("FAST EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total pairs: {len(df)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Success rate: {success_count/len(df)*100:.1f}%\n\n")
        
        if success_count > 0:
            success_df = df[df.get('failed', pd.Series([False]*len(df))) == False]
            f.write("METRIC STATISTICS:\n")
            f.write("-"*30 + "\n")
            
            for metric in ['chamfer_distance', 'gaussian_distance', 'event_count_ratio', 'temporal_overlap']:
                if metric in success_df.columns:
                    values = success_df[metric].dropna()
                    if len(values) > 0:
                        f.write(f"\n{metric}:\n")
                        f.write(f"  Mean:   {values.mean():.6f}\n")
                        f.write(f"  Std:    {values.std():.6f}\n")
                        f.write(f"  Min:    {values.min():.6f}\n")
                        f.write(f"  Max:    {values.max():.6f}\n")
                        f.write(f"  Median: {values.median():.6f}\n")
    
    if verbose:
        print(f"Results saved to: {csv_path}")
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Fast batch evaluation (limited samples)")
    
    parser.add_argument('--flare-dir', '-f',
                       default='Datasets/simu/background_with_flare_events_testoutput',
                       help='Flare results directory')
    
    parser.add_argument('--gt-dir', '-g', 
                       default='Datasets/simu/background_with_light_events_test',
                       help='Ground truth directory')
    
    parser.add_argument('--output', '-o',
                       default='results/simu_fast/',
                       help='Output directory')
    
    parser.add_argument('--num-samples', '-n', type=int, default=5,
                       help='Number of sample pairs to evaluate (default: 5)')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce verbosity')
    
    args = parser.parse_args()
    
    try:
        results_df = run_batch_evaluation(
            flare_dir=args.flare_dir,
            gt_dir=args.gt_dir,
            max_samples=args.num_samples,
            verbose=not args.quiet
        )
        
        save_results(results_df, args.output, not args.quiet)
        
        if not args.quiet:
            print("\n✓ Fast evaluation completed!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()