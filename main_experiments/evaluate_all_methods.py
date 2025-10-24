#!/usr/bin/env python3
"""
Multi-Method H5 Event Evaluation Script

Evaluates all methods against ground truth (target folder).
Automatically discovers method folders and generates comprehensive comparison table.

Features:
- Ground truth: target/ (*_bg_light.h5)
- Methods: All other folders in simu/ directory (folder name = method name)
- Default metrics: chamfer_distance, gaussian_distance, pger, voxel_mse, pmse_2, pmse_4, rf1, tf1, tpf1
- Results saved to results/ folder with method comparison table
- Supports dynamic addition of new method folders

Usage:
    python evaluate_all_methods.py [--simu-dir DIR] [--num-samples N] [--quiet]
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
from metrics import calculate_metrics, get_available_metrics, set_kernel_config


def discover_methods_and_gt(simu_dir: str) -> Tuple[str, List[str]]:
    """
    Discover ground truth folder and all method folders.
    
    Returns:
        Tuple of (gt_folder_path, list_of_method_folder_paths)
    """
    simu_path = Path(simu_dir)
    if not simu_path.exists():
        raise FileNotFoundError(f"Simu directory not found: {simu_dir}")
    
    # Ground truth folder (fixed name)
    gt_folder_name = "target"
    gt_folder_path = simu_path / gt_folder_name
    
    if not gt_folder_path.exists():
        raise FileNotFoundError(f"Ground truth folder not found: {gt_folder_path}")
    
    # Find all method folders (exclude ground truth folder)
    method_folders = []
    for folder in simu_path.iterdir():
        if folder.is_dir() and folder.name != gt_folder_name:
            # Check if folder contains H5 files
            if any(folder.glob("*.h5")):
                method_folders.append(str(folder))
    
    if not method_folders:
        raise ValueError("No method folders found")
    
    method_folders.sort()  # Sort for consistent ordering
    return str(gt_folder_path), method_folders


def extract_sample_ids_from_folder(folder_path: str) -> List[str]:
    """Extract sample IDs from H5 files in a folder."""
    folder = Path(folder_path)
    sample_ids = set()
    
    for h5_file in folder.glob("*.h5"):
        # Extract sample ID from filename
        # e.g., composed_00504_bg_light.h5 -> composed_00504
        # e.g., composed_00504_bg_flare.h5 -> composed_00504
        stem = h5_file.stem
        
        # Remove common suffixes to get sample ID
        for suffix in ["_bg_light", "_bg_flare", "_output", "_result"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        
        sample_ids.add(stem)
    
    return sorted(list(sample_ids))


def find_matching_files(sample_id: str, gt_folder: str, method_folders: List[str]) -> Dict[str, str]:
    """
    Find matching files for a sample ID across all folders.
    
    Returns:
        Dict mapping folder_name -> file_path, with 'ground_truth' as special key
    """
    matches = {}
    
    # Find ground truth file
    gt_path = Path(gt_folder)
    gt_files = list(gt_path.glob(f"{sample_id}_*.h5"))
    if gt_files:
        matches['ground_truth'] = str(gt_files[0])
    
    # Find method files
    for method_folder in method_folders:
        method_path = Path(method_folder)
        method_name = method_path.name
        
        # Look for files matching sample_id
        method_files = list(method_path.glob(f"{sample_id}_*.h5"))
        if method_files:
            matches[method_name] = str(method_files[0])
    
    return matches


def evaluate_sample_across_methods(sample_id: str, file_matches: Dict[str, str],
                                  metric_names: List[str] = None, verbose: bool = False,
                                  target_method: str = None) -> Dict:
    """
    Evaluate one sample across all methods against ground truth.

    Args:
        sample_id (str): Sample identifier
        file_matches (Dict[str, str]): Mapping of method names to file paths
        metric_names (List[str], optional): Specific metrics to calculate
        verbose (bool): Print detailed progress
        target_method (str, optional): Only evaluate this method (skip others)

    Returns:
        Dict with results for each method
    """
    if 'ground_truth' not in file_matches:
        return {'sample_id': sample_id, 'error': 'No ground truth file found'}

    try:
        # Load ground truth
        gt_events = load_events(file_matches['ground_truth'])

        if verbose:
            print(f"    GT events: {len(gt_events):,}")

        results = {'sample_id': sample_id}

        # Evaluate each method against ground truth
        for method_name, method_file in file_matches.items():
            if method_name == 'ground_truth':
                continue  # Skip ground truth itself

            # If target_method specified, skip other methods
            if target_method is not None and method_name != target_method:
                continue
            
            try:
                # Load method results
                method_events = load_events(method_file)
                
                # Calculate metrics using the new metric system
                if metric_names is None:
                    # Default to comprehensive metric set: traditional + voxel + sanity checks
                    selected_metrics = ['chamfer_distance', 'gaussian_distance', 'pger', 'voxel_mse', 'pmse_2', 'pmse_4', 'rf1', 'tf1', 'tpf1']
                else:
                    selected_metrics = metric_names
                
                method_metrics = calculate_metrics(method_events, gt_events, selected_metrics)
                
                # Add method prefix to metric names for final results
                for metric_name, value in method_metrics.items():
                    if not metric_name.endswith('_count'):  # Skip count metrics for method prefixing
                        results[f'{method_name}_{metric_name}'] = value
                
                if verbose:
                    metric_str = ', '.join([f"{k}={v:.4f}" for k, v in method_metrics.items() 
                                          if not k.endswith('_count')])
                    print(f"    {method_name}: {metric_str}")
                    
            except Exception as e:
                if verbose:
                    print(f"    {method_name}: Failed - {e}")
                # Set failed metrics to NaN
                if metric_names is None:
                    selected_metrics = ['chamfer_distance', 'gaussian_distance', 'pger', 'voxel_mse', 'pmse_2', 'pmse_4', 'rf1', 'tf1', 'tpf1']
                else:
                    selected_metrics = metric_names
                for metric_name in selected_metrics:
                    results[f'{method_name}_{metric_name}'] = np.nan
        
        return results
        
    except Exception as e:
        return {'sample_id': sample_id, 'error': str(e)}


def load_checkpoint(checkpoint_file: str, verbose: bool = True) -> Tuple[pd.DataFrame, set]:
    """Load checkpoint data if exists.

    Returns:
        Tuple of (existing_df, completed_sample_ids)
    """
    checkpoint_path = Path(checkpoint_file)

    if not checkpoint_path.exists():
        if verbose:
            print("No checkpoint found, starting fresh evaluation")
        return None, set()

    try:
        existing_df = pd.read_csv(checkpoint_path)
        # Filter out AVERAGE row if present
        existing_df = existing_df[existing_df['sample_id'] != 'AVERAGE']
        completed_ids = set(existing_df['sample_id'].tolist())

        if verbose:
            print(f"✓ Checkpoint loaded: {len(completed_ids)} samples already completed")
            print(f"  Resuming from checkpoint: {checkpoint_path}")

        return existing_df, completed_ids
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to load checkpoint: {e}")
        return None, set()


def save_incremental_result(result_row: Dict, checkpoint_file: str, is_first: bool = False):
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


def run_multi_method_evaluation(simu_dir: str = "Datasets/simu",
                               max_samples: int = None,
                               metric_names: List[str] = None,
                               verbose: bool = True,
                               checkpoint_file: str = None,
                               target_method: str = None) -> pd.DataFrame:
    """Run evaluation across all methods with checkpoint support.

    Args:
        simu_dir: Directory containing method folders
        max_samples: Maximum number of samples to process
        metric_names: List of specific metrics to calculate
        verbose: Print progress information
        checkpoint_file: Path to checkpoint file (enables resume capability)
        target_method: If specified, only evaluate this method (default: all methods)
    """

    if verbose:
        print("="*80)
        print("MULTI-METHOD H5 EVENT EVALUATION")
        print("="*80)
        if metric_names:
            print(f"Selected metrics: {', '.join(metric_names)}")
        else:
            print("Using default metrics: chamfer_distance, gaussian_distance, pger, voxel_mse, pmse_2, pmse_4, rf1, tf1, tpf1")
            print(f"Available metrics: {', '.join(get_available_metrics().keys())}")

    # Discover methods and ground truth
    gt_folder, method_folders = discover_methods_and_gt(simu_dir)
    method_names = [Path(f).name for f in method_folders]

    if verbose:
        print(f"Ground truth: {Path(gt_folder).name}")
        if target_method:
            if target_method in method_names:
                print(f"Target method: {target_method} (only this method will be evaluated)")
            else:
                raise ValueError(f"Method '{target_method}' not found. Available: {', '.join(method_names)}")
        else:
            print(f"Methods found: {len(method_names)}")
            for name in method_names:
                print(f"  - {name}")
        print()

    # Get all sample IDs from ground truth folder
    sample_ids = extract_sample_ids_from_folder(gt_folder)

    if max_samples is not None and len(sample_ids) > max_samples:
        sample_ids = sample_ids[:max_samples]
        if verbose:
            print(f"Limited to first {max_samples} samples")

    # Load checkpoint if enabled
    existing_df = None
    completed_ids = set()
    if checkpoint_file:
        existing_df, completed_ids = load_checkpoint(checkpoint_file, verbose)

    # Filter out already completed samples
    remaining_ids = [sid for sid in sample_ids if sid not in completed_ids]

    if verbose:
        print(f"Total samples: {len(sample_ids)}")
        if checkpoint_file:
            print(f"Already completed: {len(completed_ids)}")
            print(f"Remaining to process: {len(remaining_ids)}")
        print()

    # If all samples completed, return existing results
    if checkpoint_file and len(remaining_ids) == 0:
        if verbose:
            print("✓ All samples already completed!")
        return existing_df

    # Evaluate each remaining sample
    results = []
    start_time = time.time()

    for i, sample_id in enumerate(remaining_ids):
        if verbose:
            completed_count = len(completed_ids) + i
            total_count = len(sample_ids)
            print(f"[{completed_count+1}/{total_count}] {sample_id}")

        # Find matching files across all folders
        file_matches = find_matching_files(sample_id, gt_folder, method_folders)

        if verbose:
            print(f"    Found files: {len(file_matches)-1} methods + GT")

        # Evaluate this sample
        sample_result = evaluate_sample_across_methods(sample_id, file_matches, metric_names, verbose, target_method)
        results.append(sample_result)

        # Save checkpoint incrementally
        if checkpoint_file:
            is_first_write = (existing_df is None and i == 0)
            save_incremental_result(sample_result, checkpoint_file, is_first=is_first_write)

    total_time = time.time() - start_time

    # Combine existing and new results
    if existing_df is not None:
        new_df = pd.DataFrame(results)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(results)
    
    # Calculate averages for each method - use ALL columns, not just selected metrics
    method_columns = []
    # Get all metric columns (exclude sample_id and any error columns)
    for col in df.columns:
        if col != 'sample_id' and not col.endswith('_error') and not col.endswith('_count'):
            method_columns.append(col)
    
    # Add averages row
    averages = {'sample_id': 'AVERAGE'}
    for col in method_columns:
        if col in df.columns:
            averages[col] = df[col].mean(skipna=True)
    
    avg_df = pd.DataFrame([averages])
    df_with_avg = pd.concat([df, avg_df], ignore_index=True)
    
    # Print summary
    if verbose:
        print()
        print("="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        successful = len(df.dropna(subset=method_columns[:2] if method_columns else []))  # Check first method
        
        print(f"Total samples: {len(df)}")
        print(f"Methods evaluated: {len(method_names)}")
        print(f"Processing time: {total_time:.1f}s")
        
        if successful > 0:
            print()
            print("AVERAGE METRICS BY METHOD:")
            print("-" * 50)
            for method_name in method_names:
                print(f"{method_name}:")
                # Find all metrics for this method
                method_cols = [col for col in df.columns if col.startswith(method_name + '_')]
                for col in sorted(method_cols):
                    metric_name = col[len(method_name + '_'):]
                    avg_value = df[col].mean(skipna=True)
                    print(f"  {metric_name.replace('_', ' ').title()}: {avg_value:.6f}")
                print()
    
    return df_with_avg


def save_results(df: pd.DataFrame, output_dir: str = "results", verbose: bool = True):
    """Save multi-method results to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive results
    csv_path = output_path / "multi_method_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"Results saved to: {csv_path}")
        
        # Print final averages summary
        avg_row = df[df['sample_id'] == 'AVERAGE'].iloc[0]
        print("\nFINAL AVERAGES SUMMARY:")
        print("="*50)
        
        # Group by method and extract all metrics
        method_metrics = {}
        for col in df.columns:
            if col != 'sample_id' and not col.endswith('_count'):
                # Parse method name and metric name from column
                # Handle different metric naming patterns
                if '_' in col:
                    # Known metric patterns for proper parsing
                    known_metrics = ['chamfer_distance', 'gaussian_distance', 'temporal_overlap', 
                                   'event_count_ratio', 'pger', 'pmse_2', 'pmse_4', 
                                   'rf1', 'tf1', 'tpf1']
                    
                    method_name = None
                    metric_full_name = None
                    
                    # Try to match known metric patterns
                    for metric in known_metrics:
                        if col.endswith('_' + metric):
                            method_name = col[:-len('_' + metric)]
                            metric_full_name = metric
                            break
                    
                    # Fallback: split from the right
                    if method_name is None:
                        parts = col.rsplit('_', 1)
                        if len(parts) == 2:
                            method_name = parts[0]
                            metric_full_name = parts[1]
                    
                    if method_name and metric_full_name:
                        if method_name not in method_metrics:
                            method_metrics[method_name] = {}
                        method_metrics[method_name][metric_full_name] = avg_row[col]
        
        for method_name, metrics in method_metrics.items():
            print(f"{method_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name.replace('_', ' ').title()}: {value:.6f}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Multi-method H5 evaluation against ground truth")
    
    parser.add_argument('--simu-dir', '-d',
                       default='Datasets/simu',
                       help='Simu directory containing method folders (default: %(default)s)')
    
    parser.add_argument('--output', '-o',
                       default='results',
                       help='Output directory (default: %(default)s)')
    
    parser.add_argument('--num-samples', '-n', type=int,
                       help='Limit number of samples (default: all)')
    
    parser.add_argument('--metrics', '-m', nargs='+',
                       help='Specific metrics to calculate (e.g., --metrics chamfer_distance gaussian_distance)')
    
    parser.add_argument('--list-metrics', action='store_true',
                       help='List all available metrics and exit')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce verbosity')

    parser.add_argument('--method', type=str,
                       help='Only evaluate this specific method (e.g., inputefr). Default: all methods')

    # Kernel configuration arguments
    parser.add_argument('--kernel-sampling', type=str, choices=['on', 'off'], default='off',
                       help='Enable/disable kernel sampling (default: off - fine cubes don\'t need sampling)')

    parser.add_argument('--kernel-max-events', type=int, default=10000,
                       help='Max events per cube before sampling (default: 10000)')

    parser.add_argument('--kernel-verbose', type=str, choices=['on', 'off'], default='off',
                       help='Show kernel progress tracking (default: off)')

    args = parser.parse_args()

    try:
        # Handle --list-metrics
        if args.list_metrics:
            available_metrics = get_available_metrics()
            print("Available metrics:")
            for name, info in available_metrics.items():
                print(f"  {name}: {info['description']} (category: {info['category']})")
            return

        # Configure kernel before running evaluation
        set_kernel_config(
            enabled=(args.kernel_sampling == 'on'),
            max_events=args.kernel_max_events,
            verbose=(args.kernel_verbose == 'on')
        )

        # Enable checkpoint by default (save to output directory)
        checkpoint_path = Path(args.output) / "multi_method_evaluation_results.csv"

        results_df = run_multi_method_evaluation(
            simu_dir=args.simu_dir,
            max_samples=args.num_samples,
            metric_names=args.metrics,
            verbose=not args.quiet,
            checkpoint_file=str(checkpoint_path),
            target_method=args.method
        )
        
        save_results(results_df, args.output, not args.quiet)
        
        if not args.quiet:
            print("✓ Multi-method evaluation completed successfully!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()