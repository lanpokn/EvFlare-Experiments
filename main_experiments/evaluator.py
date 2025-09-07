"""
Main Experiment Evaluator Module

This module contains the core evaluator that orchestrates the entire experimental pipeline:
- Loading data from various sources
- Running different glare removal methods
- Computing evaluation metrics
- Collecting and organizing results
"""

import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from data_loader import load_events
from methods import MethodResult
from metrics import calculate_all_metrics, calculate_metrics


class MainExperimentEvaluator:
    """
    Core coordinator for the main experiment pipeline.
    
    This class handles the complete evaluation workflow:
    1. Load method results and ground truth event data
    2. Compute metrics comparing results to ground truth  
    3. Collect and organize all results
    
    Note: The actual glare removal processing is done externally.
    This framework only evaluates pre-processed results.
    """
    
    def __init__(self, method_results: List[MethodResult], verbose: bool = True, 
                 metric_names: Optional[List[str]] = None):
        """
        Initialize the evaluator with a list of method results to evaluate.
        
        Args:
            method_results (List[MethodResult]): List of method results to evaluate
            verbose (bool): Whether to print detailed progress information
            metric_names (List[str], optional): Specific metrics to calculate. If None, uses all.
        """
        self.method_results = method_results
        self.verbose = verbose
        self.metric_names = metric_names
        self.results = []
        self.failed_evaluations = []
        
        if self.verbose:
            print(f"Initialized evaluator with {len(method_results)} method results:")
            for result in method_results:
                print(f"  - {result.name}")
            if metric_names:
                print(f"Selected metrics: {', '.join(metric_names)}")
            else:
                print("Using all available metrics")

    def run_single_evaluation(self, 
                            gt_file_path: str, 
                            sample_name: str,
                            save_intermediate: bool = False,
                            output_dir: Optional[str] = None) -> bool:
        """
        Run a complete evaluation for a single sample against all method results.
        
        Args:
            gt_file_path (str): Path to ground truth (clean background) events file
            sample_name (str): Descriptive name for this sample
            save_intermediate (bool): Whether to save intermediate results
            output_dir (str, optional): Directory to save intermediate results
            
        Returns:
            bool: True if evaluation completed successfully, False otherwise
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting evaluation for sample: {sample_name}")
            print(f"{'='*60}")
        
        # 1. Load ground truth data
        try:
            if self.verbose:
                print(f"Loading ground truth events from: {gt_file_path}")
            start_time = time.time()
            events_gt = load_events(gt_file_path)
            
            load_time = time.time() - start_time
            if self.verbose:
                print(f"Ground truth loading completed in {load_time:.2f}s")
                print(f"  Ground truth events: {len(events_gt):,}")
                
        except Exception as e:
            error_msg = f"Error loading ground truth data for {sample_name}: {e}"
            print(f"ERROR: {error_msg}")
            self.failed_evaluations.append({
                'sample_name': sample_name,
                'error': error_msg,
                'error_type': 'data_loading',
                'traceback': traceback.format_exc()
            })
            return False

        # 2. Evaluate each method result
        sample_success = True
        for i, method_result in enumerate(self.method_results):
            if self.verbose:
                print(f"\n[{i+1}/{len(self.method_results)}] Evaluating method: {method_result.name}")
            
            method_start_time = time.time()
            
            try:
                # 3. Load method result
                if self.verbose:
                    print(f"  Loading method results...")
                load_start = time.time()
                events_est = method_result.get_events()
                load_time = time.time() - load_start
                
                if self.verbose:
                    print(f"  Method result loading completed in {load_time:.2f}s")
                    print(f"  Estimated events: {len(events_est):,}")
                
                # 4. Calculate metrics
                if self.verbose:
                    print(f"  Computing metrics...")
                metrics_start = time.time()
                if self.metric_names:
                    metrics = calculate_metrics(events_est, events_gt, self.metric_names)
                else:
                    metrics = calculate_all_metrics(events_est, events_gt)
                metrics_time = time.time() - metrics_start
                
                method_total_time = time.time() - method_start_time
                
                # Add timing information to metrics
                metrics.update({
                    'result_loading_time_s': load_time,
                    'metrics_computation_time_s': metrics_time,
                    'total_evaluation_time_s': method_total_time
                })
                
                if self.verbose:
                    print(f"  Metrics computation completed in {metrics_time:.2f}s")
                    print(f"  Key metrics:")
                    for key, value in metrics.items():
                        if not key.endswith('_time_s') and not key.endswith('_count'):
                            print(f"    {key}: {value:.6f}")
                
                # 5. Save intermediate results if requested
                if save_intermediate and output_dir:
                    self._save_intermediate_results(
                        events_est, sample_name, method_result.name, output_dir
                    )
                
                # 6. Record results
                result_record = {
                    'sample_name': sample_name,
                    'method_name': method_result.name,
                    'method_config': method_result.get_config(),
                    'gt_file_path': gt_file_path,
                    **metrics
                }
                self.results.append(result_record)
                
                if self.verbose:
                    print(f"  ✓ Method {method_result.name} completed successfully")
                    
            except Exception as e:
                error_msg = f"Error evaluating {sample_name} with {method_result.name}: {e}"
                print(f"  ✗ ERROR: {error_msg}")
                
                # Record the failure but continue with other methods
                failure_record = {
                    'sample_name': sample_name,
                    'method_name': method_result.name,
                    'error': error_msg,
                    'error_type': 'method_evaluation',
                    'traceback': traceback.format_exc()
                }
                self.failed_evaluations.append(failure_record)
                sample_success = False
        
        if self.verbose:
            status = "✓ COMPLETED" if sample_success else "⚠ COMPLETED WITH ERRORS"
            print(f"\n{status}: Sample {sample_name}")
            
        return sample_success

    def run_batch_evaluation(self, 
                           evaluation_samples: List[Dict[str, str]], 
                           save_intermediate: bool = False,
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run evaluation on multiple samples in batch.
        
        Args:
            evaluation_samples (List[Dict]): List of dicts with keys 'name', 'gt'
            save_intermediate (bool): Whether to save intermediate results  
            output_dir (str, optional): Directory to save intermediate results
            
        Returns:
            Dict[str, Any]: Summary of batch evaluation results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"STARTING BATCH EVALUATION")
            print(f"{'='*80}")
            print(f"Total samples: {len(evaluation_samples)}")
            print(f"Total methods: {len(self.method_results)}")
            print(f"Total evaluations: {len(evaluation_samples) * len(self.method_results)}")
        
        if save_intermediate and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        batch_start_time = time.time()
        successful_samples = 0
        
        for i, sample in enumerate(evaluation_samples):
            if self.verbose:
                print(f"\n[{i+1}/{len(evaluation_samples)}] Processing batch item")
                
            success = self.run_single_evaluation(
                gt_file_path=sample['gt'], 
                sample_name=sample['name'],
                save_intermediate=save_intermediate,
                output_dir=output_dir
            )
            
            if success:
                successful_samples += 1
        
        batch_total_time = time.time() - batch_start_time
        
        # Generate summary
        summary = {
            'total_samples': len(evaluation_samples),
            'successful_samples': successful_samples,
            'failed_samples': len(evaluation_samples) - successful_samples,
            'total_methods': len(self.method_results),
            'total_evaluations_attempted': len(evaluation_samples) * len(self.method_results),
            'successful_evaluations': len(self.results),
            'failed_evaluations': len(self.failed_evaluations),
            'batch_processing_time_s': batch_total_time,
            'success_rate': successful_samples / len(evaluation_samples) if evaluation_samples else 0
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"BATCH EVALUATION SUMMARY")
            print(f"{'='*80}")
            for key, value in summary.items():
                print(f"{key}: {value}")
        
        return summary

    def get_results_as_dataframe(self):
        """
        Return collected results as a pandas DataFrame for analysis.
        
        Returns:
            pd.DataFrame: Results organized in tabular format
            
        Note:
            Requires pandas to be installed. If not available, returns None.
        """
        try:
            import pandas as pd
            if not self.results:
                print("Warning: No results available to convert to DataFrame")
                return pd.DataFrame()
            return pd.DataFrame(self.results)
        except ImportError:
            print("Warning: pandas not available. Install with 'pip install pandas'")
            return None

    def save_results_to_csv(self, output_path: str) -> bool:
        """
        Save results to CSV file.
        
        Args:
            output_path (str): Path where to save the CSV file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            df = self.get_results_as_dataframe()
            if df is not None and not df.empty:
                df.to_csv(output_path, index=False)
                if self.verbose:
                    print(f"Results saved to: {output_path}")
                return True
            else:
                print("Warning: No results to save")
                return False
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    def get_failure_summary(self) -> Dict[str, Any]:
        """
        Get summary of failed evaluations for debugging.
        
        Returns:
            Dict[str, Any]: Summary of failures organized by type
        """
        if not self.failed_evaluations:
            return {'total_failures': 0}
            
        failures_by_type = {}
        failures_by_sample = {}
        failures_by_method = {}
        
        for failure in self.failed_evaluations:
            error_type = failure.get('error_type', 'unknown')
            sample_name = failure.get('sample_name', 'unknown')
            method_name = failure.get('method_name', 'unknown')
            
            failures_by_type[error_type] = failures_by_type.get(error_type, 0) + 1
            failures_by_sample[sample_name] = failures_by_sample.get(sample_name, 0) + 1
            failures_by_method[method_name] = failures_by_method.get(method_name, 0) + 1
        
        return {
            'total_failures': len(self.failed_evaluations),
            'failures_by_type': failures_by_type,
            'failures_by_sample': failures_by_sample,
            'failures_by_method': failures_by_method,
            'detailed_failures': self.failed_evaluations
        }

    def _save_intermediate_results(self, events_est: np.ndarray, 
                                 sample_name: str, method_name: str, 
                                 output_dir: str) -> None:
        """Save intermediate processing results for later analysis."""
        try:
            output_path = Path(output_dir) / f"{sample_name}_{method_name}_result.npy"
            np.save(str(output_path), events_est)
            if self.verbose:
                print(f"    Saved intermediate result to: {output_path}")
        except Exception as e:
            print(f"    Warning: Could not save intermediate result: {e}")

    def reset_results(self) -> None:
        """Clear all collected results and start fresh."""
        self.results.clear()
        self.failed_evaluations.clear()
        if self.verbose:
            print("Results cleared - evaluator reset to initial state")