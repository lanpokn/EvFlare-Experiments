#!/usr/bin/env python3
"""
Debug Render Script - Find out what's actually happening
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def debug_render_single_model():
    """Debug a single model rendering process"""
    
    # Setup paths
    weights_dir = "datasets/lego2/3dgs_results/weights"
    gaussian_splatting_dir = "gaussian-splatting"
    
    # Find first available model
    if not os.path.exists(weights_dir):
        print(f"ERROR: No weights directory at {weights_dir}")
        return False
    
    available_models = [d for d in os.listdir(weights_dir) 
                       if os.path.isdir(os.path.join(weights_dir, d))]
    
    if not available_models:
        print(f"ERROR: No model directories found in {weights_dir}")
        return False
    
    model_name = available_models[0]
    print(f"Using model: {model_name}")
    
    # Setup paths
    source_weights = os.path.join(weights_dir, model_name)
    temp_output_dir = f"{gaussian_splatting_dir}/debug_render_{model_name}"
    
    print(f"Source weights: {source_weights}")
    print(f"Temp output dir: {temp_output_dir}")
    
    try:
        # Step 1: Copy model
        print("\n=== Step 1: Copying model ===")
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        
        shutil.copytree(source_weights, temp_output_dir)
        print(f"SUCCESS: Model copied to {temp_output_dir}")
        
        # Check contents
        print("Model contents:")
        for root, dirs, files in os.walk(temp_output_dir):
            level = root.replace(temp_output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Step 2: Run render command
        print(f"\n=== Step 2: Running render command ===")
        dataset_dir = "datasets/lego2"  # Original dataset path
        render_cmd = [
            "python", 
            os.path.join(gaussian_splatting_dir, "render.py"),
            "-m", temp_output_dir,
            "-s", dataset_dir,  # Add source dataset path!
            "--skip_train",
            "--grayscale"
        ]
        
        print(f"Command: {' '.join(render_cmd)}")
        result = subprocess.run(render_cmd, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr if result.stderr else "(no stderr)")
        
        # Also check if render.py expects source dataset
        print(f"\n=== Checking dataset requirements ===")
        cfg_args_path = os.path.join(temp_output_dir, "cfg_args")
        if os.path.exists(cfg_args_path):
            print("Contents of cfg_args:")
            with open(cfg_args_path, 'r') as f:
                cfg_content = f.read()
                print(cfg_content)
                
                # Check if it has source_path
                if "source_path" in cfg_content:
                    print("WARNING: cfg_args contains source_path - render.py might need original dataset")
        else:
            print("WARNING: No cfg_args file found")
        
        # Step 3: Check output structure
        print(f"\n=== Step 3: Checking output structure ===")
        print("Complete directory structure after rendering:")
        for root, dirs, files in os.walk(temp_output_dir):
            level = root.replace(temp_output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size} bytes)")
        
        # Step 4: Look for expected render path
        print(f"\n=== Step 4: Looking for expected render outputs ===")
        expected_render_path = os.path.join(temp_output_dir, "test", "ours_7000", "renders")
        expected_gt_path = os.path.join(temp_output_dir, "test", "ours_7000", "gt")
        
        print(f"Expected render path: {expected_render_path}")
        print(f"Expected GT path: {expected_gt_path}")
        
        if os.path.exists(expected_render_path):
            png_files = list(Path(expected_render_path).glob("*.png"))
            print(f"SUCCESS: Found {len(png_files)} PNG files in render path")
            if png_files:
                print(f"First few files: {[f.name for f in png_files[:5]]}")
        else:
            print(f"ERROR: Render path does not exist")
        
        if os.path.exists(expected_gt_path):
            gt_files = list(Path(expected_gt_path).glob("*.png"))
            print(f"SUCCESS: Found {len(gt_files)} GT files")
        else:
            print(f"ERROR: GT path does not exist")
        
        # Step 5: Look for any PNG files anywhere
        print(f"\n=== Step 5: Searching for any PNG files ===")
        all_pngs = list(Path(temp_output_dir).rglob("*.png"))
        print(f"Total PNG files found: {len(all_pngs)}")
        for png in all_pngs:
            rel_path = png.relative_to(temp_output_dir)
            file_size = png.stat().st_size
            print(f"  {rel_path} ({file_size} bytes)")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"ERROR: Exception occurred: {e}")
        return False
    
    finally:
        # Keep temp directory for debugging
        print(f"\n=== Debug Complete ===")
        print(f"Temp directory preserved for inspection: {temp_output_dir}")
        print("Note: Temp directory kept for debugging purposes")

if __name__ == "__main__":
    print("=== 3DGS Render Debug Script ===")
    success = debug_render_single_model()
    print(f"\nDebug completed. Success: {success}")