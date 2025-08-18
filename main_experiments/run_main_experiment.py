#!/usr/bin/env python3
"""
Main Experiment Runner

This script demonstrates how to use the experiment framework to evaluate
pre-processed glare removal results against ground truth data.

Usage:
    python run_main_experiment.py [--config config.json] [--output results/]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import our framework modules
from methods import MethodResult
from evaluator import MainExperimentEvaluator


def create_example_method_results() -> List[MethodResult]:
    """
    Create example method results for demonstration.
    
    Each MethodResult points to a pre-processed result file from a different
    glare removal approach.
    
    Returns:
        List of MethodResult objects pointing to processed data files
    """
    method_results = []
    
    # Example method results - REPLACE WITH YOUR ACTUAL RESULT PATHS
    method_results.extend([
        MethodResult(
            name="MyAwesomeModel_v1", 
            result_file_path="results/my_model/sample01_processed.aedat4",
            model_version="1.0", 
            processing_date="2025-01-15"
        ),
        MethodResult(
            name="BaselineFilter_temporal", 
            result_file_path="results/baseline_temporal/sample01_filtered.h5",
            filter_type="temporal", 
            threshold=0.1
        ),
        MethodResult(
            name="CompetitorMethod_A", 
            result_file_path="results/competitor_a/sample01_result.npy",
            paper_reference="Smith et al. 2024"
        ),
        MethodResult(
            name="CompetitorMethod_B", 
            result_file_path="results/competitor_b/sample01_output.aedat4",
            paper_reference="Jones et al. 2024"
        ),
    ])
    
    return method_results


def create_example_evaluation_samples() -> List[Dict[str, str]]:
    """
    Create example evaluation samples for demonstration.
    
    Each sample specifies the ground truth file for comparison.
    The method results are loaded separately and evaluated against each sample.
    
    Returns:
        List of evaluation samples with 'name' and 'gt' keys
    """
    
    # Example evaluation samples - REPLACE WITH YOUR ACTUAL DATA PATHS
    evaluation_samples = [
        {
            'name': 'sample_01_indoor',
            'gt': 'data/ground_truth/indoor_scene_01_clean.aedat4'  # Clean background only
        },
        {
            'name': 'sample_02_outdoor', 
            'gt': 'data/ground_truth/outdoor_drive_02_clean.h5'
        },
        {
            'name': 'sample_03_office',
            'gt': '/mnt/shared_storage/ground_truth/office_clean.aedat4'
        },
        # Add more evaluation samples as needed...
    ]
    
    return evaluation_samples


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON file.
    
    Args:
        config_path (str): Path to JSON configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_example_config_file(output_path: str = "example_config.json") -> None:
    """
    Create an example configuration file for reference.
    
    Args:
        output_path (str): Where to save the example config file
    """
    example_config = {
        "experiment_name": "glare_removal_evaluation",
        "method_results": [
            {
                "name": "MyAwesomeModel_v1",
                "result_file_path": "results/my_model/processed_data.aedat4",
                "metadata": {
                    "model_version": "1.0",
                    "processing_date": "2025-01-15"
                }
            },
            {
                "name": "BaselineFilter_temporal", 
                "result_file_path": "results/baseline/filtered_data.h5",
                "metadata": {
                    "filter_type": "temporal",
                    "threshold": 0.1
                }
            }
        ],
        "evaluation_samples": [
            {
                "name": "test_sample_01",
                "gt": "path/to/ground_truth_data.aedat4"
            }
        ],
        "output": {
            "results_dir": "results/",
            "save_intermediate": True,
            "save_csv": True
        },
        "options": {
            "verbose": True
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example configuration saved to: {output_path}")


def main():
    """
    Main execution function - demonstrates the complete evaluation workflow.
    """
    parser = argparse.ArgumentParser(description="Run main glare removal experiment")
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    parser.add_argument('--create-example-config', action='store_true', 
                       help='Create example configuration file and exit')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_example_config:
        create_example_config_file()
        return
    
    print("="*80)
    print("Event Glare Removal - Main Experiment Runner")
    print("="*80)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration or use defaults
        if args.config:
            print(f"Loading configuration from: {args.config}")
            config = load_config_from_file(args.config)
            # TODO: Parse config to create methods and evaluation pairs
            print("WARNING: Config file parsing not yet implemented")
            print("Using example methods and data pairs...")
        
        # For now, use example setup
        print("\nInitializing method results...")
        method_results = create_example_method_results()
        
        print("\nSetting up evaluation samples...")  
        evaluation_samples = create_example_evaluation_samples()
        
        print(f"\nNote: This example uses placeholder data paths.")
        print(f"Update the paths in create_example_method_results() and create_example_evaluation_samples()")
        print(f"to point to your actual processed results and ground truth data.")
        print(f"The framework supports .aedat4, .h5, and .npy files.")
        
        # Create evaluator
        print(f"\nCreating evaluator...")
        evaluator = MainExperimentEvaluator(method_results=method_results, verbose=args.verbose)
        
        # Option 1: Run single evaluation (for testing)
        print(f"\n" + "="*60)
        print("OPTION 1: Single Evaluation Example")
        print("="*60)
        print("To run a single evaluation, uncomment and modify the following lines:")
        print("# success = evaluator.run_single_evaluation(")
        print("#     gt_file_path='path/to/your/ground_truth_data.aedat4',")
        print("#     sample_name='test_sample',")
        print("#     save_intermediate=True,")
        print("#     output_dir=str(output_dir)")
        print("# )")
        
        # Option 2: Run batch evaluation
        print(f"\n" + "="*60)  
        print("OPTION 2: Batch Evaluation Example")
        print("="*60)
        print("To run batch evaluation, first ensure your data paths are correct,")
        print("then uncomment the following lines:")
        print("# batch_summary = evaluator.run_batch_evaluation(")
        print("#     evaluation_samples=evaluation_samples,")
        print("#     save_intermediate=True,")
        print("#     output_dir=str(output_dir)")
        print("# )")
        print("# ")
        print("# # Save results")
        print("# results_csv = output_dir / 'main_experiment_results.csv'")
        print("# evaluator.save_results_to_csv(str(results_csv))")
        print("# ")
        print("# # Get summary of any failures")
        print("# failure_summary = evaluator.get_failure_summary()")
        print("# if failure_summary['total_failures'] > 0:")
        print("#     print(f'\\nFailure Summary: {failure_summary}')")
        
        # Demonstrate framework without running actual experiments
        print(f"\n" + "="*60)
        print("FRAMEWORK DEMONSTRATION")  
        print("="*60)
        print("The framework is now ready to use!")
        print(f"Method results configured: {len(method_results)}")
        for i, result in enumerate(method_results):
            print(f"  {i+1}. {result.name}")
        
        print(f"\nEvaluation samples configured: {len(evaluation_samples)}")
        for i, sample in enumerate(evaluation_samples[:3]):  # Show first 3
            print(f"  {i+1}. {sample['name']}")
        if len(evaluation_samples) > 3:
            print(f"  ... and {len(evaluation_samples)-3} more")
            
        print(f"\nOutput directory: {output_dir}")
        print("\nTo run actual evaluations:")
        print("1. Update method result paths in create_example_method_results()")
        print("2. Update ground truth paths in create_example_evaluation_samples()")  
        print("3. Ensure AEDAT4/H5 file loading is implemented if needed")
        print("4. Uncomment evaluation code above")
        print("5. Run the script")
        
        print(f"\nFramework files created successfully!")
        print("Ready for actual implementation and experiments.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()