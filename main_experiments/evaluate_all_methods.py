#!/usr/bin/env python3
"""
Multi-Method H5 Event Evaluation Script

Evaluates all methods against ground truth (background_with_light_events_test).
Automatically discovers method folders and generates comprehensive comparison table.

Features:
- Ground truth: background_with_light_events_test/ (*_bg_light.h5)
- Methods: All other folders in simu/ directory (any naming pattern)
- Only computes chamfer_distance and gaussian_distance
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
from metrics import chamfer_distance_loss, gaussian_distance_loss


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
    gt_folder_name = "background_with_light_events_test"
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


def evaluate_sample_across_methods(sample_id: str, file_matches: Dict[str, str], verbose: bool = False) -> Dict:
    """
    Evaluate one sample across all methods against ground truth.
    
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
            
            try:
                # Load method results
                method_events = load_events(method_file)
                
                # Calculate metrics
                chamfer_dist = chamfer_distance_loss(method_events, gt_events)
                gaussian_dist = gaussian_distance_loss(method_events, gt_events)
                
                results[f'{method_name}_chamfer_distance'] = chamfer_dist
                results[f'{method_name}_gaussian_distance'] = gaussian_dist
                
                if verbose:
                    print(f"    {method_name}: CD={chamfer_dist:.4f}, GD={gaussian_dist:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"    {method_name}: Failed - {e}")
                results[f'{method_name}_chamfer_distance'] = np.nan
                results[f'{method_name}_gaussian_distance'] = np.nan
        
        return results
        
    except Exception as e:
        return {'sample_id': sample_id, 'error': str(e)}


def run_multi_method_evaluation(simu_dir: str = "Datasets/simu", 
                               max_samples: int = None, 
                               verbose: bool = True) -> pd.DataFrame:
    """Run evaluation across all methods."""
    
    if verbose:
        print("="*80)
        print("MULTI-METHOD H5 EVENT EVALUATION")
        print("="*80)
    
    # Discover methods and ground truth
    gt_folder, method_folders = discover_methods_and_gt(simu_dir)
    method_names = [Path(f).name for f in method_folders]
    
    if verbose:
        print(f"Ground truth: {Path(gt_folder).name}")
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
    
    if verbose:
        print(f"Processing {len(sample_ids)} samples")
        print()
    
    # Evaluate each sample
    results = []
    start_time = time.time()
    
    for i, sample_id in enumerate(sample_ids):
        if verbose:
            print(f"[{i+1}/{len(sample_ids)}] {sample_id}")
        
        # Find matching files across all folders
        file_matches = find_matching_files(sample_id, gt_folder, method_folders)
        
        if verbose:
            print(f"    Found files: {len(file_matches)-1} methods + GT")
        
        # Evaluate this sample
        sample_result = evaluate_sample_across_methods(sample_id, file_matches, verbose)
        results.append(sample_result)
    
    total_time = time.time() - start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate averages for each method
    method_columns = []
    for method_name in method_names:
        chamfer_col = f'{method_name}_chamfer_distance'
        gaussian_col = f'{method_name}_gaussian_distance'
        if chamfer_col in df.columns:
            method_columns.extend([chamfer_col, gaussian_col])
    
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
                chamfer_col = f'{method_name}_chamfer_distance'
                gaussian_col = f'{method_name}_gaussian_distance'
                
                if chamfer_col in df.columns and gaussian_col in df.columns:
                    chamfer_avg = df[chamfer_col].mean(skipna=True)
                    gaussian_avg = df[gaussian_col].mean(skipna=True)
                    print(f"{method_name}:")
                    print(f"  Chamfer Distance: {chamfer_avg:.6f}")
                    print(f"  Gaussian Distance: {gaussian_avg:.6f}")
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
        
        # Group by method
        method_metrics = {}
        for col in df.columns:
            if col.endswith('_chamfer_distance'):
                method_name = col[:-len('_chamfer_distance')]
                if method_name not in method_metrics:
                    method_metrics[method_name] = {}
                method_metrics[method_name]['chamfer'] = avg_row[col]
            elif col.endswith('_gaussian_distance'):
                method_name = col[:-len('_gaussian_distance')]
                if method_name not in method_metrics:
                    method_metrics[method_name] = {}
                method_metrics[method_name]['gaussian'] = avg_row[col]
        
        for method_name, metrics in method_metrics.items():
            print(f"{method_name}:")
            if 'chamfer' in metrics:
                print(f"  Chamfer Distance: {metrics['chamfer']:.6f}")
            if 'gaussian' in metrics:
                print(f"  Gaussian Distance: {metrics['gaussian']:.6f}")
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
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce verbosity')
    
    args = parser.parse_args()
    
    try:
        results_df = run_multi_method_evaluation(
            simu_dir=args.simu_dir,
            max_samples=args.num_samples,
            verbose=not args.quiet
        )
        
        save_results(results_df, args.output, not args.quiet)
        
        if not args.quiet:
            print("✓ Multi-method evaluation completed successfully!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()