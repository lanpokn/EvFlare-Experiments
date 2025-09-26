#!/usr/bin/env python3
"""
3DGS Render and Evaluate Existing Models
Renders images from trained 3DGS models and calculates evaluation metrics
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse

def find_trained_models(weights_dir):
    """Find all trained models with iteration_7000"""
    models = []
    if not os.path.exists(weights_dir):
        return models
    
    for model_dir in os.listdir(weights_dir):
        model_path = os.path.join(weights_dir, model_dir)
        checkpoint_path = os.path.join(model_path, "point_cloud", "iteration_7000", "point_cloud.ply")
        
        if os.path.exists(checkpoint_path):
            models.append(model_dir)
            print(f"Found trained model: {model_dir}")
    
    return models

def run_command(cmd, description=""):
    """Run a shell command and return success status"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"ERROR: {description} failed")
            print(f"STDERR: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return False

def copy_model_for_rendering(source_path, temp_path):
    """Copy trained model to temporary location for rendering"""
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    
    try:
        shutil.copytree(source_path, temp_path)
        return True
    except Exception as e:
        print(f"ERROR: Failed to copy model: {e}")
        return False

def render_model(model_name, temp_output_dir, gaussian_splatting_dir, dataset_dir):
    """Render test set for a specific model"""
    render_cmd = [
        "python", 
        os.path.join(gaussian_splatting_dir, "render.py"),
        "-m", temp_output_dir,
        "-s", dataset_dir,  # Add source dataset path!
        "--skip_train",
        "--grayscale"
    ]
    
    return run_command(render_cmd, f"Rendering {model_name}")

def backup_renders(temp_output_dir, render_dest):
    """Backup rendered images to final location"""
    render_source = os.path.join(temp_output_dir, "test", "ours_7000", "renders")
    
    print(f"DEBUG: Looking for renders at: {render_source}")
    
    if not os.path.exists(render_source):
        print(f"ERROR: No renders found at {render_source}")
        
        # Debug: Show complete directory structure
        print("DEBUG: Complete temp directory structure:")
        for root, dirs, files in os.walk(temp_output_dir):
            level = root.replace(temp_output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.endswith('.png'):
                    print(f"{subindent}{file} (PNG FILE)")
                else:
                    print(f"{subindent}{file}")
        
        # Try to find PNG files anywhere
        png_files = []
        for root, dirs, files in os.walk(temp_output_dir):
            for file in files:
                if file.endswith('.png'):
                    png_files.append(os.path.join(root, file))
        
        print(f"DEBUG: Found {len(png_files)} PNG files total:")
        for png in png_files:
            rel_path = os.path.relpath(png, temp_output_dir)
            print(f"  - {rel_path}")
        
        return False
    
    if os.path.exists(render_dest):
        shutil.rmtree(render_dest)
    
    try:
        shutil.copytree(render_source, render_dest)
        
        # Count images
        png_files = list(Path(render_dest).glob("*.png"))
        print(f"SUCCESS: {len(png_files)} images backed up to {render_dest}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to backup renders: {e}")
        return False

def calculate_metrics(temp_output_dir, metrics_file, gaussian_splatting_dir):
    """Calculate evaluation metrics"""
    metrics_cmd = [
        "python",
        os.path.join(gaussian_splatting_dir, "metrics.py"),
        "-m", temp_output_dir,
        "--grayscale"
    ]
    
    try:
        result = subprocess.run(metrics_cmd, capture_output=True, text=True)
        
        with open(metrics_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"SUCCESS: Metrics calculated and saved to {metrics_file}")
            return True
        else:
            print(f"WARNING: Metrics calculation failed, check {metrics_file}")
            return False
    except Exception as e:
        print(f"ERROR: Failed to calculate metrics: {e}")
        return False

def parse_metrics_file(metrics_file):
    """Parse metrics from output file"""
    metrics = {"SSIM": "nan", "PSNR": "nan", "LPIPS": "nan"}
    
    if not os.path.exists(metrics_file):
        return metrics
    
    try:
        with open(metrics_file, 'r') as f:
            content = f.read()
            
        for line in content.split('\n'):
            if "SSIM" in line and ":" in line:
                metrics["SSIM"] = line.split(':')[1].strip()
            elif "PSNR" in line and ":" in line:
                metrics["PSNR"] = line.split(':')[1].strip()
            elif "LPIPS" in line and ":" in line:
                metrics["LPIPS"] = line.split(':')[1].strip()
    except Exception as e:
        print(f"WARNING: Failed to parse metrics file {metrics_file}: {e}")
    
    return metrics

def generate_comparison_report(success_models, metrics_output_dir, dataset_name, method_name):
    """Generate a comprehensive comparison report"""
    comparison_file = os.path.join(metrics_output_dir, "comparison_report.txt")
    json_report_file = os.path.join(metrics_output_dir, "comparison_report.json")
    
    # Text report
    with open(comparison_file, 'w') as f:
        f.write(f"3DGS Model Comparison Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Models Processed: {len(success_models)}\n\n")
        
        all_metrics = {}
        
        for model_name in success_models:
            metrics_file = os.path.join(metrics_output_dir, f"{model_name}_metrics.txt")
            metrics = parse_metrics_file(metrics_file)
            all_metrics[model_name] = metrics
            
            f.write(f"Model: {model_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  SSIM : {metrics['SSIM']}\n")
            f.write(f"  PSNR : {metrics['PSNR']}\n")
            f.write(f"  LPIPS: {metrics['LPIPS']}\n\n")
    
    # JSON report for programmatic access
    report_data = {
        "dataset": dataset_name,
        "method": method_name,
        "timestamp": datetime.now().isoformat(),
        "models_processed": len(success_models),
        "results": all_metrics
    }
    
    with open(json_report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"SUCCESS: Comparison reports saved:")
    print(f"  Text: {comparison_file}")
    print(f"  JSON: {json_report_file}")

def main():
    parser = argparse.ArgumentParser(description="Render and evaluate existing 3DGS models")
    parser.add_argument("--dataset", default="lego2", help="Dataset name")
    parser.add_argument("--method", default="spade_e2vid", help="Method name")
    parser.add_argument("--weights-dir", help="Custom weights directory path")
    parser.add_argument("--debug", action="store_true", help="Keep temp directories for debugging")
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_dir = f"datasets/{args.dataset}"
    results_dir = f"{dataset_dir}/3dgs_results"
    weights_dir = args.weights_dir or f"{results_dir}/weights"
    gaussian_splatting_dir = "gaussian-splatting"
    
    print("=" * 50)
    print("3DGS Render and Evaluate Existing Models")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Weights Directory: {weights_dir}")
    print("=" * 50)
    
    # Validation
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return 1
    
    if not os.path.exists(weights_dir):
        print(f"ERROR: Weights directory not found: {weights_dir}")
        print("No trained models available for rendering")
        return 1
    
    # Find available models
    trained_models = find_trained_models(weights_dir)
    if not trained_models:
        print("ERROR: No trained models found with iteration_7000")
        return 1
    
    print(f"Found {len(trained_models)} trained models: {trained_models}")
    
    # Setup output directories
    render_output_dir = f"{results_dir}/final_renders"
    metrics_output_dir = f"{results_dir}/final_metrics"
    
    os.makedirs(render_output_dir, exist_ok=True)
    os.makedirs(metrics_output_dir, exist_ok=True)
    
    print(f"Output directories:")
    print(f"  Renders: {render_output_dir}")
    print(f"  Metrics: {metrics_output_dir}")
    
    # Process each model
    success_models = []
    failed_models = []
    
    for model_name in trained_models:
        print(f"\n{'='*40}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*40}")
        
        # Setup paths
        source_weights = os.path.join(weights_dir, model_name)
        temp_output_dir = f"{gaussian_splatting_dir}/temp_render_{model_name}"
        
        try:
            # Copy model for rendering
            if not copy_model_for_rendering(source_weights, temp_output_dir):
                failed_models.append(model_name)
                continue
            
            # Render test set
            if not render_model(model_name, temp_output_dir, gaussian_splatting_dir, dataset_dir):
                failed_models.append(model_name)
                continue
            
            # Backup renders
            render_dest = os.path.join(render_output_dir, model_name)
            if not backup_renders(temp_output_dir, render_dest):
                failed_models.append(model_name)
                continue
            
            # Calculate metrics
            metrics_file = os.path.join(metrics_output_dir, f"{model_name}_metrics.txt")
            calculate_metrics(temp_output_dir, metrics_file, gaussian_splatting_dir)
            
            success_models.append(model_name)
            print(f"SUCCESS: {model_name} processing completed")
            
        finally:
            # Cleanup (unless debugging)
            if not args.debug and os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
            elif args.debug:
                print(f"DEBUG: Temp directory preserved: {temp_output_dir}")
    
    # Generate final report
    print(f"\n{'='*50}")
    print("Render and Evaluation Complete")
    print(f"{'='*50}")
    print(f"Total models processed: {len(trained_models)}")
    print(f"Successful renders: {len(success_models)}")
    print(f"Failed renders: {len(failed_models)}")
    
    if failed_models:
        print(f"Failed models: {failed_models}")
    if success_models:
        print(f"Successful models: {success_models}")
    
    if success_models:
        generate_comparison_report(success_models, metrics_output_dir, args.dataset, args.method)
    
    print(f"\nResults saved to:")
    print(f"  Renders: {render_output_dir}")
    print(f"  Metrics: {metrics_output_dir}")
    
    return 0 if success_models else 1

if __name__ == "__main__":
    sys.exit(main())