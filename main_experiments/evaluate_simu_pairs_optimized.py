#!/usr/bin/env python3
"""
Optimized H5 Event Pairs Evaluation Script

Features:
- Only computes chamfer_distance and gaussian_distance
- Results saved to results/ folder
- Generates CSV with averages row
- Clean and focused output

Usage:
    python evaluate_simu_pairs_optimized.py [--num-samples N] [--quiet]
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
from metrics import chamfer_distance_loss, gaussian_distance_loss


def find_paired_files(flare_dir: str, gt_dir: str, max_samples: int = None) -> List[Tuple[str, str, str]]:
    """Find paired H5 files."""
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
        print(f"Limited to first {max_samples} samples")
    
    return paired_files


def evaluate_single_pair(sample_id: str, flare_path: str, gt_path: str, verbose: bool = False) -> Dict:
    """Evaluate a single pair - only compute the two main metrics."""
    try:
        if verbose:
            print(f"  Processing {sample_id}...")
            
        start_time = time.time()
        
        # Load both files
        events_flare = load_events(flare_path)
        events_gt = load_events(gt_path)
        
        if verbose:
            print(f"    Events: flare={len(events_flare):,}, gt={len(events_gt):,}")
        
        # Calculate only the two required metrics
        chamfer_dist = chamfer_distance_loss(events_flare, events_gt)
        gaussian_dist = gaussian_distance_loss(events_flare, events_gt)
        
        total_time = time.time() - start_time
        
        result = {
            'sample_id': sample_id,
            'chamfer_distance': chamfer_dist,
            'gaussian_distance': gaussian_dist
        }
        
        if verbose:
            print(f"    ✓ Completed in {total_time:.2f}s")
            print(f"    Chamfer: {chamfer_dist:.4f}, Gaussian: {gaussian_dist:.4f}")
            
        return result
        
    except Exception as e:
        if verbose:
            print(f"    ✗ Failed: {e}")
            
        return {
            'sample_id': sample_id,
            'chamfer_distance': np.nan,
            'gaussian_distance': np.nan,
            'error': str(e)
        }


def run_evaluation(flare_dir: str, gt_dir: str, max_samples: int = None, verbose: bool = True) -> pd.DataFrame:
    """Run evaluation and return results DataFrame."""
    if verbose:
        print("="*60)
        print("H5 EVENT PAIRS EVALUATION")
        print("="*60)
    
    # Find paired files
    paired_files = find_paired_files(flare_dir, gt_dir, max_samples)
    
    if verbose:
        print(f"Found {len(paired_files)} paired files to process")
        print()
    
    # Evaluate each pair
    results = []
    batch_start_time = time.time()
    
    for i, (sample_id, flare_path, gt_path) in enumerate(paired_files):
        if verbose:
            print(f"[{i+1}/{len(paired_files)}] {sample_id}")
            
        result = evaluate_single_pair(sample_id, flare_path, gt_path, verbose)
        results.append(result)
    
    batch_time = time.time() - batch_start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate and add averages row
    numeric_cols = ['chamfer_distance', 'gaussian_distance']
    averages = df[numeric_cols].mean(skipna=True)
    
    avg_row = pd.DataFrame([{
        'sample_id': 'AVERAGE',
        'chamfer_distance': averages['chamfer_distance'],
        'gaussian_distance': averages['gaussian_distance']
    }])
    
    df_with_avg = pd.concat([df, avg_row], ignore_index=True)
    
    # Print summary
    if verbose:
        successful = df.dropna(subset=['chamfer_distance']).shape[0]
        failed = len(df) - successful
        
        print()
        print("="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total pairs: {len(df)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(df)*100:.1f}%")
        print(f"Total processing time: {batch_time:.1f}s")
        
        if successful > 0:
            valid_df = df.dropna(subset=['chamfer_distance'])
            print()
            print("METRIC STATISTICS:")
            print("-" * 30)
            print(f"Chamfer Distance:")
            print(f"  Mean: {averages['chamfer_distance']:.4f}")
            print(f"  Std:  {valid_df['chamfer_distance'].std():.4f}")
            print(f"  Min:  {valid_df['chamfer_distance'].min():.4f}")
            print(f"  Max:  {valid_df['chamfer_distance'].max():.4f}")
            print()
            print(f"Gaussian Distance:")
            print(f"  Mean: {averages['gaussian_distance']:.4f}")
            print(f"  Std:  {valid_df['gaussian_distance'].std():.4f}")
            print(f"  Min:  {valid_df['gaussian_distance'].min():.4f}")
            print(f"  Max:  {valid_df['gaussian_distance'].max():.4f}")
    
    return df_with_avg


def save_results(df: pd.DataFrame, output_dir: str = "results", verbose: bool = True):
    """Save results to CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV with averages
    csv_path = output_path / "h5_pairs_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"\nResults saved to: {csv_path}")
        
        # Print averages for quick reference
        avg_row = df[df['sample_id'] == 'AVERAGE'].iloc[0]
        print("\nFINAL AVERAGES:")
        print(f"  Chamfer Distance: {avg_row['chamfer_distance']:.6f}")
        print(f"  Gaussian Distance: {avg_row['gaussian_distance']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Optimized H5 pairs evaluation")
    
    parser.add_argument('--flare-dir', '-f',
                       default='Datasets/simu/background_with_flare_events_testoutput',
                       help='Flare results directory')
    
    parser.add_argument('--gt-dir', '-g', 
                       default='Datasets/simu/background_with_light_events_test',
                       help='Ground truth directory')
    
    parser.add_argument('--output', '-o',
                       default='results',
                       help='Output directory (default: results)')
    
    parser.add_argument('--num-samples', '-n', type=int,
                       help='Limit number of samples (default: all)')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce verbosity')
    
    args = parser.parse_args()
    
    try:
        results_df = run_evaluation(
            flare_dir=args.flare_dir,
            gt_dir=args.gt_dir,
            max_samples=args.num_samples,
            verbose=not args.quiet
        )
        
        save_results(results_df, args.output, not args.quiet)
        
        if not args.quiet:
            print("\n✓ Evaluation completed successfully!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()