#!/usr/bin/env python3
"""
EVK4 Method Evaluation Script

Evaluates EVK4 methods against ground truth (target folder).
Adapted from evaluate_all_methods.py for EVK4 data structure.

Data structure:
- Ground truth: Datasets/EVK4_result/target/ (defocus_*ms_sample*.h5)
- Methods: Datasets/EVK4_result/{input,inputefr,inputpfda,inputpfdb,output_*,outputbaseline}/

Features:
- Comprehensive metrics: traditional (Chamfer, Gaussian) + voxel (PMSE, F1) + utility (ratios, overlap)
- Dynamic method discovery
- Comprehensive results table with averages
- Progress tracking and timing

Default metrics:
- Traditional: chamfer_distance, gaussian_distance
- Voxel MSE: voxel_mse, pmse_2, pmse_4
- Voxel F1: rf1, tf1, tpf1
- Utility: event_count_ratio, temporal_overlap

Usage:
    python evaluate_evk4_methods.py [--evk4-dir DIR] [--num-samples N] [--metrics METRICS] [--quiet]
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
try:
    from metrics import calculate_metrics, get_available_metrics
    METRICS_AVAILABLE = True
except:
    METRICS_AVAILABLE = False
    print("Warning: Full metrics module not available, using basic metrics")


def calculate_basic_metrics(events_est, events_gt):
    """Basic metrics calculation using sklearn"""
    from sklearn.neighbors import NearestNeighbors
    
    if len(events_est) == 0 or len(events_gt) == 0:
        return {'chamfer_distance': float('inf'), 'gaussian_distance': float('inf')}
    
    # Limit events for performance
    max_events = 50000
    if len(events_est) > max_events:
        indices = np.random.choice(len(events_est), max_events, replace=False)
        events_est = events_est[indices]
    if len(events_gt) > max_events:
        indices = np.random.choice(len(events_gt), max_events, replace=False)
        events_gt = events_gt[indices]
    
    # Build coordinate matrices
    coords_est = np.column_stack([events_est['x'], events_est['y']])
    coords_gt = np.column_stack([events_gt['x'], events_gt['y']])
    
    # Chamfer distance (estimated -> ground truth)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords_gt)
    distances, _ = nbrs.kneighbors(coords_est)
    chamfer_distance = np.mean(distances)
    
    # Gaussian distance (simplified)
    gaussian_distance = np.mean(np.exp(-distances.flatten() / (2 * 0.4**2)))
    
    return {
        'chamfer_distance': float(chamfer_distance),
        'gaussian_distance': float(gaussian_distance)
    }


def discover_evk4_methods_and_gt(evk4_dir: str) -> Tuple[str, List[str]]:
    """
    Discover ground truth folder and all method folders for EVK4.
    
    Returns:
        Tuple of (gt_folder_path, list_of_method_folder_paths)
    """
    evk4_path = Path(evk4_dir)
    
    # Ground truth is always 'target' folder
    gt_folder = evk4_path / 'target'
    if not gt_folder.exists():
        raise ValueError(f"Ground truth folder not found: {gt_folder}")
    
    # Find all method folders (exclude target)
    method_folders = []
    for item in evk4_path.iterdir():
        if item.is_dir() and item.name != 'target':
            # Check if folder contains H5 files
            h5_files = list(item.glob('*.h5'))
            if h5_files:
                method_folders.append(str(item))
                print(f"Found method: {item.name} ({len(h5_files)} files)")
    
    if not method_folders:
        raise ValueError(f"No method folders found in {evk4_dir}")
    
    print(f"Ground truth: {gt_folder.name}")
    print(f"Found {len(method_folders)} method folders")
    
    return str(gt_folder), method_folders


def get_sample_pairs_evk4(gt_folder: str, method_folders: List[str], num_samples: int = None) -> List[Dict]:
    """
    Get sample pairs for EVK4 evaluation.
    
    Returns:
        List of sample dictionaries with gt_file and method_files
    """
    gt_path = Path(gt_folder)
    gt_files = sorted(list(gt_path.glob('*ms_sample*.h5')))
    
    if not gt_files:
        raise ValueError(f"No EVK4 H5 files found in ground truth folder: {gt_folder}")
    
    # Limit samples if specified
    if num_samples and num_samples < len(gt_files):
        gt_files = gt_files[:num_samples]
    
    samples = []
    for gt_file in gt_files:
        # Extract sample identifier from full filename
        gt_name = gt_file.stem  # e.g., defocus_25ms_sample1 or redandfull_sixflare_12ms_sample1
        
        sample_dict = {
            'sample_id': gt_name,  # Use full filename as sample_id
            'gt_file': str(gt_file)
        }
        
        # Find corresponding method files
        for method_folder in method_folders:
            method_path = Path(method_folder)
            method_name = method_path.name
            
            # Look for exact matching filename in method folder
            method_file = method_path / f"{gt_name}.h5"
            if method_file.exists():
                sample_dict[f'{method_name}_file'] = str(method_file)
            else:
                print(f"Warning: Missing {method_name} file for sample {gt_name}")
        
        samples.append(sample_dict)
    
    print(f"Found {len(samples)} sample pairs")
    return samples


def load_checkpoint_evk4(checkpoint_file: str, quiet: bool = False) -> Tuple[pd.DataFrame, set]:
    """Load checkpoint data if exists.

    Returns:
        Tuple of (existing_df, completed_sample_ids)
    """
    checkpoint_path = Path(checkpoint_file)

    if not checkpoint_path.exists():
        if not quiet:
            print("No checkpoint found, starting fresh evaluation")
        return None, set()

    try:
        existing_df = pd.read_csv(checkpoint_path)
        # Filter out AVERAGE row if present
        existing_df = existing_df[existing_df['sample_id'] != 'AVERAGE']
        completed_ids = set(existing_df['sample_id'].tolist())

        if not quiet:
            print(f"✓ Checkpoint loaded: {len(completed_ids)} samples already completed")
            print(f"  Resuming from checkpoint: {checkpoint_path}")

        return existing_df, completed_ids
    except Exception as e:
        if not quiet:
            print(f"Warning: Failed to load checkpoint: {e}")
        return None, set()


def save_incremental_result_evk4(result_row: Dict, checkpoint_file: str, is_first: bool = False):
    """Save single result row incrementally to CSV.

    Args:
        result_row: Dictionary with sample results
        checkpoint_file: Path to checkpoint CSV
        is_first: Whether this is the first row (write header)
    """
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame([result_row])

    # Write with or without header
    mode = 'w' if is_first else 'a'
    df.to_csv(checkpoint_path, mode=mode, header=is_first, index=False)


def evaluate_evk4_methods(evk4_dir: str = "Datasets/EVK4_result",
                         num_samples: int = None,
                         selected_metrics: List[str] = None,
                         output_dir: str = "results",
                         quiet: bool = False,
                         checkpoint_file: str = None) -> pd.DataFrame:
    """
    Evaluate all EVK4 methods against ground truth with checkpoint support.

    Args:
        evk4_dir: EVK4 directory path
        num_samples: Limit number of samples
        selected_metrics: Metrics to calculate
        output_dir: Output directory
        quiet: Reduce verbosity
        checkpoint_file: Path to checkpoint file (enables resume)
    """

    if not quiet:
        print("="*80)
        print("EVK4 METHOD EVALUATION")
        print("="*80)
        print(f"EVK4 directory: {evk4_dir}")
        if num_samples:
            print(f"Sample limit: {num_samples}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Discover methods and ground truth
    gt_folder, method_folders = discover_evk4_methods_and_gt(evk4_dir)
    method_names = [Path(folder).name for folder in method_folders]
    
    # Get available metrics with comprehensive defaults
    default_metrics = [
        # 传统距离指标
        'chamfer_distance', 'gaussian_distance',
        # Voxel MSE指标 (voxel_mse最严格，pmse_2中等，pmse_4最宽松)
        'voxel_mse', 'pmse_2', 'pmse_4',
        # Voxel F1指标
        'rf1', 'tf1', 'tpf1',
        # 其他有用指标
        'event_count_ratio', 'temporal_overlap'
    ]
    
    if METRICS_AVAILABLE:
        available_metrics = get_available_metrics()
        if selected_metrics is None:
            selected_metrics = default_metrics
        
        # Validate selected metrics，只保留可用的
        selected_metrics = [m for m in selected_metrics if m in available_metrics]
        if not selected_metrics:
            print("Warning: No valid metrics found, using basic fallback")
            selected_metrics = ['chamfer_distance', 'gaussian_distance']
    else:
        selected_metrics = ['chamfer_distance', 'gaussian_distance']
    
    if not quiet:
        print(f"Using metrics: {selected_metrics}")
    
    # Get sample pairs
    samples = get_sample_pairs_evk4(gt_folder, method_folders, num_samples)

    # Load checkpoint if enabled
    existing_df = None
    completed_ids = set()
    if checkpoint_file:
        existing_df, completed_ids = load_checkpoint_evk4(checkpoint_file, quiet)

    # Filter out already completed samples
    remaining_samples = [s for s in samples if s['sample_id'] not in completed_ids]

    if not quiet:
        print(f"Total samples: {len(samples)}")
        if checkpoint_file:
            print(f"Already completed: {len(completed_ids)}")
            print(f"Remaining to process: {len(remaining_samples)}")
        print("Starting evaluation...")

    # If all samples completed, return existing results
    if checkpoint_file and len(remaining_samples) == 0:
        if not quiet:
            print("✓ All samples already completed!")
        return existing_df

    # Evaluate each remaining sample
    results = []
    start_time = time.time()

    for i, sample in enumerate(remaining_samples):
        sample_id = sample['sample_id']

        if not quiet:
            completed_count = len(completed_ids) + i
            total_count = len(samples)
            print(f"\n[{completed_count+1}/{total_count}] Processing sample: {sample_id}")
        
        # Load ground truth
        try:
            gt_events = load_events(sample['gt_file'])
            if not quiet:
                print(f"  GT events: {len(gt_events):,}")
        except Exception as e:
            print(f"  Error loading GT: {e}")
            continue
        
        # Prepare result row
        result_row = {'sample_id': sample_id}
        
        # Evaluate each method
        for method_name in method_names:
            method_file_key = f"{method_name}_file"
            
            if method_file_key not in sample:
                # Fill with NaN for missing methods
                for metric in selected_metrics:
                    result_row[f"{method_name}_{metric}"] = np.nan
                continue
            
            try:
                # Load method results
                method_events = load_events(sample[method_file_key])
                
                if not quiet:
                    print(f"  {method_name} events: {len(method_events):,}")
                
                # Calculate metrics
                if METRICS_AVAILABLE:
                    metrics = calculate_metrics(method_events, gt_events, selected_metrics)
                else:
                    # Fallback basic metrics
                    metrics = calculate_basic_metrics(method_events, gt_events)
                
                # Add to result row
                for metric in selected_metrics:
                    result_row[f"{method_name}_{metric}"] = metrics.get(metric, np.nan)
                
            except Exception as e:
                print(f"  Error evaluating {method_name}: {e}")
                # Fill with NaN for failed evaluations
                for metric in selected_metrics:
                    result_row[f"{method_name}_{metric}"] = np.nan
        
        results.append(result_row)

        # Save checkpoint incrementally
        if checkpoint_file:
            is_first_write = (existing_df is None and i == 0)
            save_incremental_result_evk4(result_row, checkpoint_file, is_first=is_first_write)

    # Combine existing and new results
    if existing_df is not None:
        new_df = pd.DataFrame(results)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(results)
    
    # Calculate averages
    numeric_columns = [col for col in df.columns if col != 'sample_id']
    averages = {'sample_id': 'AVERAGE'}
    for col in numeric_columns:
        averages[col] = df[col].mean()
    
    # Add averages row
    df = pd.concat([df, pd.DataFrame([averages])], ignore_index=True)
    
    # Save results
    output_file = os.path.join(output_dir, 'evk4_evaluation_results.csv')
    df.to_csv(output_file, index=False)
    
    elapsed_time = time.time() - start_time
    
    if not quiet:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Samples processed: {len(samples)}")
        print(f"Methods evaluated: {method_names}")
        print(f"Metrics calculated: {selected_metrics}")
        print(f"Results saved to: {output_file}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Average time per sample: {elapsed_time/len(samples):.2f}s")
        
        # Show summary statistics
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        avg_row = df[df['sample_id'] == 'AVERAGE'].iloc[0]
        for method in method_names:
            print(f"\n{method}:")
            for metric in selected_metrics:
                col_name = f"{method}_{metric}"
                if col_name in avg_row:
                    value = avg_row[col_name]
                    if not np.isnan(value):
                        print(f"  {metric}: {value:.6f}")
    
    return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="EVK4 Method Evaluation")
    parser.add_argument('--evk4-dir', type=str, default='Datasets/EVK4_result',
                       help='EVK4 directory path (default: Datasets/EVK4_result)')
    parser.add_argument('--num-samples', type=int,
                       help='Limit number of samples to evaluate (default: all)')
    parser.add_argument('--metrics', type=str, nargs='+',
                       help='Metrics to calculate (default: comprehensive set including voxel metrics)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--list-metrics', action='store_true',
                       help='List available metrics and exit')
    
    args = parser.parse_args()
    
    if args.list_metrics:
        available_metrics = get_available_metrics()
        print("Available metrics:")
        for metric in available_metrics:
            print(f"  {metric}")
        return
    
    try:
        # Enable checkpoint by default (save to output directory)
        checkpoint_path = Path(args.output) / "evk4_evaluation_results.csv"

        df = evaluate_evk4_methods(
            evk4_dir=args.evk4_dir,
            num_samples=args.num_samples,
            selected_metrics=args.metrics,
            output_dir=args.output,
            quiet=args.quiet,
            checkpoint_file=str(checkpoint_path)
        )
        
        if not args.quiet:
            print(f"\n✅ EVK4 evaluation completed successfully!")
            print(f"Results shape: {df.shape}")
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()