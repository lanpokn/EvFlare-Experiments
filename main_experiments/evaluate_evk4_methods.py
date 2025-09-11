#!/usr/bin/env python3
"""
EVK4 Method Evaluation Script

Evaluates EVK4 methods against ground truth (target folder).
Adapted from evaluate_all_methods.py for EVK4 data structure.

Data structure:
- Ground truth: Datasets/EVk4_result/target/ (evk4_*ms_sample*.h5)
- Methods: Datasets/EVk4_result/{baseline,input,inputpfds,unet3d}/

Features:
- Comprehensive metrics: traditional (Chamfer, Gaussian) + voxel (PMSE, F1) + utility (ratios, overlap)
- Dynamic method discovery
- Comprehensive results table with averages
- Progress tracking and timing

Default metrics:
- Traditional: chamfer_distance, gaussian_distance
- Voxel PMSE: pmse_2, pmse_4 
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
        # Extract sample identifier (e.g., "25ms_sample1" from "defocus_25ms_sample1.h5" or "evk4_25ms_sample1.h5")
        gt_name = gt_file.stem  # defocus_25ms_sample1 or evk4_25ms_sample1
        sample_id = gt_name.replace('evk4_', '').replace('defocus_', '')  # 25ms_sample1
        
        sample_dict = {
            'sample_id': sample_id,
            'gt_file': str(gt_file)
        }
        
        # Find corresponding method files
        for method_folder in method_folders:
            method_path = Path(method_folder)
            method_name = method_path.name
            
            # Look for matching file in method folder (try both defocus_ and evk4_ prefixes)
            method_file_defocus = method_path / f"defocus_{sample_id}.h5"
            method_file_evk4 = method_path / f"evk4_{sample_id}.h5"
            method_file = method_file_defocus if method_file_defocus.exists() else method_file_evk4
            if method_file.exists():
                sample_dict[f'{method_name}_file'] = str(method_file)
            else:
                print(f"Warning: Missing {method_name} file for sample {sample_id}")
        
        samples.append(sample_dict)
    
    print(f"Found {len(samples)} sample pairs")
    return samples


def evaluate_evk4_methods(evk4_dir: str = "Datasets/EVk4_result", 
                         num_samples: int = None,
                         selected_metrics: List[str] = None,
                         output_dir: str = "results",
                         quiet: bool = False) -> pd.DataFrame:
    """
    Evaluate all EVK4 methods against ground truth.
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
        # Voxel PMSE指标  
        'pmse_2', 'pmse_4',
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
    
    if not quiet:
        print(f"Evaluating {len(samples)} samples across {len(method_names)} methods")
        print("Starting evaluation...")
    
    # Evaluate each sample
    results = []
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        
        if not quiet:
            print(f"\n[{i+1}/{len(samples)}] Processing sample: {sample_id}")
        
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
    
    # Create DataFrame
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
    parser.add_argument('--evk4-dir', type=str, default='Datasets/EVk4_result',
                       help='EVK4 directory path (default: Datasets/EVk4_result)')
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
        df = evaluate_evk4_methods(
            evk4_dir=args.evk4_dir,
            num_samples=args.num_samples,
            selected_metrics=args.metrics,
            output_dir=args.output,
            quiet=args.quiet
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