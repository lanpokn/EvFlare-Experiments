#!/usr/bin/env python3
"""
Video generation script for 3D reconstruction image sequences
Converts image sequences from four directories to MP4 videos
"""

import cv2
import os
import sys
from pathlib import Path
import argparse

def create_video_from_images(image_dir, output_video_path, fps=10):
    """
    Create a video from a sequence of images
    
    Args:
        image_dir (str): Directory containing images
        output_video_path (str): Path for output video file
        fps (int): Frames per second for output video
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        return False
    
    # Get sorted list of PNG files
    image_files = sorted([f for f in image_dir.glob("*.png")])
    
    if not image_files:
        print(f"Error: No PNG files found in {image_dir}")
        return False
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"Error: Cannot read first image {image_files[0]}")
        return False
    
    height, width, _ = first_img.shape
    print(f"Video dimensions: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Cannot create video writer for {output_video_path}")
        return False
    
    # Process all images
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read image {img_path}")
            continue
        
        # Ensure image has correct dimensions
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")
    
    video_writer.release()
    print(f"Video created successfully: {output_video_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create videos from image sequences")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--output-dir", type=str, default="videos", help="Output directory for videos")
    args = parser.parse_args()
    
    # Base directory
    base_dir = Path(__file__).parent / "datasets" / "lego"
    
    # Define image directories and corresponding video names
    sequences = [
        ("train", "lego_train_flare.mp4"),
        ("test", "lego_test_normal.mp4"),
        ("reconstruction/evreal_e2vid", "lego_reconstruction_e2vid.mp4"),
        ("reconstruction/evreal_firenet", "lego_reconstruction_firenet.mp4")
    ]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating videos at {args.fps} FPS")
    print(f"Output directory: {output_dir.absolute()}")
    
    success_count = 0
    total_count = len(sequences)
    
    for rel_path, video_name in sequences:
        image_dir = base_dir / rel_path
        output_path = output_dir / video_name
        
        print(f"\n--- Processing {rel_path} ---")
        print(f"Input: {image_dir}")
        print(f"Output: {output_path}")
        
        if create_video_from_images(image_dir, output_path, args.fps):
            success_count += 1
        else:
            print(f"Failed to create video for {rel_path}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully created {success_count}/{total_count} videos")
    
    if success_count == total_count:
        print("All videos created successfully!")
        return 0
    else:
        print(f"Failed to create {total_count - success_count} videos")
        return 1

if __name__ == "__main__":
    sys.exit(main())