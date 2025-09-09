#!/usr/bin/env python3
"""
DAVIS Dataset Generator
生成对齐的DAVIS炫光/无炫光数据集，格式与DSEC一致

从两个AEDAT4文件（无炫光和有炫光）中生成成对的H5文件：
1. 时间对齐到重叠区间
2. 随机选择10个起始点，每个提取100ms数据
3. 存储为H5格式，与DSEC格式完全一致
4. 输出到指定的input/target目录结构

Usage:
    python generate_davis_dataset.py
"""

import argparse
import time
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, List
import random

# Import our framework modules
from data_loader import load_events


def align_time_ranges(events_clean: np.ndarray, events_flare: np.ndarray) -> Tuple[float, float]:
    """
    找到两个事件流的时间重叠区间
    
    Args:
        events_clean: 无炫光事件数据
        events_flare: 有炫光事件数据
        
    Returns:
        tuple: (overlap_start, overlap_end) 重叠区间的起始和结束时间
    """
    clean_start, clean_end = events_clean['t'].min(), events_clean['t'].max()
    flare_start, flare_end = events_flare['t'].min(), events_flare['t'].max()
    
    # 计算重叠区间
    overlap_start = max(clean_start, flare_start)
    overlap_end = min(clean_end, flare_end)
    
    print(f"Clean time range: {clean_start:.3f}s to {clean_end:.3f}s ({clean_end-clean_start:.3f}s)")
    print(f"Flare time range: {flare_start:.3f}s to {flare_end:.3f}s ({flare_end-flare_start:.3f}s)")
    print(f"Overlap range: {overlap_start:.3f}s to {overlap_end:.3f}s ({overlap_end-overlap_start:.3f}s)")
    
    if overlap_start >= overlap_end:
        raise ValueError("No time overlap found between the two event streams")
    
    return overlap_start, overlap_end


def extract_events_window(events: np.ndarray, start_time: float, duration: float) -> np.ndarray:
    """
    从事件流中提取指定时间窗口的数据
    
    Args:
        events: 事件数据
        start_time: 起始时间 (秒)
        duration: 持续时间 (秒)
        
    Returns:
        提取的事件数据，时间戳重新对齐到从0开始
    """
    end_time = start_time + duration
    mask = (events['t'] >= start_time) & (events['t'] < end_time)
    window_events = events[mask].copy()
    
    # 重新对齐时间戳到从0开始
    if len(window_events) > 0:
        window_events['t'] = window_events['t'] - start_time
    
    return window_events


def save_events_to_h5(events: np.ndarray, file_path: str):
    """
    将事件数据保存为H5格式，与DSEC格式完全一致
    
    Args:
        events: 结构化事件数组 [('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]
        file_path: 输出文件路径
        
    Note:
        H5格式规范：
        - events/t: 时间戳 (微秒, uint64)
        - events/x: X坐标 (uint16) 
        - events/y: Y坐标 (uint16)
        - events/p: 极性 (0/1, uint8)
    """
    # 确保输出目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        # 创建events组
        events_group = f.create_group('events')
        
        # 转换时间戳：秒 -> 微秒，float64 -> uint64
        t_microseconds = (events['t'] * 1e6).astype(np.uint64)
        
        # 转换极性：-1/+1 -> 0/1
        p_binary = np.where(events['p'] == 1, 1, 0).astype(np.uint8)
        
        # 保存数据
        events_group.create_dataset('t', data=t_microseconds, dtype=np.uint64)
        events_group.create_dataset('x', data=events['x'].astype(np.uint16), dtype=np.uint16)
        events_group.create_dataset('y', data=events['y'].astype(np.uint16), dtype=np.uint16)
        events_group.create_dataset('p', data=p_binary, dtype=np.uint8)
        
    print(f"  Saved {len(events):,} events to {file_path}")


def generate_random_sampling_points(overlap_start: float, overlap_end: float, 
                                   duration: float, num_samples: int, 
                                   seed: int = 42) -> List[float]:
    """
    在重叠区间内生成随机采样起始点
    
    Args:
        overlap_start: 重叠区间开始时间
        overlap_end: 重叠区间结束时间  
        duration: 每个样本的持续时间
        num_samples: 样本数量
        seed: 随机种子
        
    Returns:
        随机起始时间点列表
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 确保有足够空间进行采样
    available_range = overlap_end - overlap_start - duration
    if available_range <= 0:
        raise ValueError(f"Overlap range ({overlap_end-overlap_start:.3f}s) is too small for {duration}s samples")
    
    # 生成随机起始点
    start_points = []
    for i in range(num_samples):
        start_time = overlap_start + random.uniform(0, available_range)
        start_points.append(start_time)
    
    # 排序便于处理
    start_points.sort()
    
    print(f"Generated {num_samples} random sampling points:")
    for i, start_time in enumerate(start_points):
        print(f"  Sample {i+1}: {start_time:.3f}s to {start_time+duration:.3f}s")
    
    return start_points


def generate_davis_dataset(clean_file_path: str, 
                          flare_file_path: str,
                          output_base_dir: str = "/mnt/e/2025/event_flick_flare/experiments/main_experiments/Datasets/DAVIS",
                          num_samples: int = 10,
                          sample_duration: float = 0.1,
                          seed: int = 42):
    """
    生成DAVIS数据集：从AEDAT4文件生成成对的H5文件
    
    Args:
        clean_file_path: 无炫光AEDAT4文件路径
        flare_file_path: 有炫光AEDAT4文件路径
        output_base_dir: 输出基础目录
        num_samples: 生成样本数量
        sample_duration: 每个样本持续时间(秒)
        seed: 随机种子
    """
    print("="*80)
    print("DAVIS DATASET GENERATOR")
    print("="*80)
    print(f"Clean (target): {clean_file_path}")
    print(f"Flare (input):  {flare_file_path}")
    print(f"Output dir:     {output_base_dir}")
    print(f"Samples:        {num_samples} × {sample_duration*1000:.0f}ms")
    print(f"Random seed:    {seed}")
    
    start_time = time.time()
    
    # 1. 加载数据
    print("\n" + "="*50)
    print("STEP 1: Loading AEDAT4 files")
    print("="*50)
    
    events_clean = load_events(clean_file_path)
    events_flare = load_events(flare_file_path)
    
    print(f"✓ Loaded clean data: {len(events_clean):,} events")
    print(f"✓ Loaded flare data: {len(events_flare):,} events")
    
    # 2. 时间对齐
    print("\n" + "="*50)
    print("STEP 2: Time alignment")
    print("="*50)
    
    overlap_start, overlap_end = align_time_ranges(events_clean, events_flare)
    overlap_duration = overlap_end - overlap_start
    
    if overlap_duration < num_samples * sample_duration:
        print(f"Warning: Overlap duration ({overlap_duration:.3f}s) may not be sufficient")
        print(f"         for {num_samples} non-overlapping samples of {sample_duration}s each")
    
    # 3. 生成随机采样点
    print("\n" + "="*50)
    print("STEP 3: Random sampling points generation")
    print("="*50)
    
    start_points = generate_random_sampling_points(
        overlap_start, overlap_end, sample_duration, num_samples, seed
    )
    
    # 4. 创建输出目录
    print("\n" + "="*50)
    print("STEP 4: Dataset generation")
    print("="*50)
    
    input_dir = Path(output_base_dir) / "input"    # 炫光数据 (需要去炫光处理)
    target_dir = Path(output_base_dir) / "target"  # 无炫光数据 (真值)
    
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input dir:  {input_dir}")
    print(f"Target dir: {target_dir}")
    
    # 5. 生成样本对
    for i, start_time in enumerate(start_points):
        sample_id = i + 1
        print(f"\nGenerating sample {sample_id}/{num_samples} (start: {start_time:.3f}s)...")
        
        # 提取时间窗口
        clean_window = extract_events_window(events_clean, start_time, sample_duration)
        flare_window = extract_events_window(events_flare, start_time, sample_duration)
        
        print(f"  Clean window: {len(clean_window):,} events")
        print(f"  Flare window: {len(flare_window):,} events")
        
        if len(clean_window) == 0 or len(flare_window) == 0:
            print(f"  Warning: Empty window for sample {sample_id}, skipping")
            continue
        
        # 生成文件名：full_起始时间ms_sample编号.h5
        start_time_ms = int(start_time * 1000)
        filename = f"full_{start_time_ms:05d}ms_sample{sample_id:02d}.h5"
        
        # 保存文件
        input_path = input_dir / filename    # 炫光数据 -> input
        target_path = target_dir / filename  # 无炫光数据 -> target
        
        save_events_to_h5(flare_window, str(input_path))   # 炫光数据作为输入
        save_events_to_h5(clean_window, str(target_path))  # 无炫光数据作为目标
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Generated:     {num_samples} sample pairs")
    print(f"Input files:   {input_dir}/*.h5")
    print(f"Target files:  {target_dir}/*.h5")
    print(f"Total time:    {total_time:.2f}s")
    print(f"File format:   H5 with DSEC-compatible structure")
    print("\nFiles ready for flare removal algorithms and evaluation!")


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="Generate DAVIS dataset from AEDAT4 files")
    parser.add_argument('--clean_file', type=str, 
                       default="/mnt/e/2025/event_flick_flare/datasets/DAVIS/test/redlightandfull_noFlare-2025_06_25_15_23_32.aedat4",
                       help='Path to clean (no flare) AEDAT4 file')
    parser.add_argument('--flare_file', type=str,
                       default="/mnt/e/2025/event_flick_flare/datasets/DAVIS/test/redlightandfull_sixFlare-2025_06_25_15_24_11.aedat4", 
                       help='Path to flare AEDAT4 file')
    parser.add_argument('--output_dir', type=str,
                       default="/mnt/e/2025/event_flick_flare/experiments/main_experiments/Datasets/DAVIS",
                       help='Output base directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sample pairs to generate')
    parser.add_argument('--duration', type=float, default=0.1,
                       help='Duration of each sample in seconds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    try:
        generate_davis_dataset(
            clean_file_path=args.clean_file,
            flare_file_path=args.flare_file, 
            output_base_dir=args.output_dir,
            num_samples=args.num_samples,
            sample_duration=args.duration,
            seed=args.seed
        )
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())