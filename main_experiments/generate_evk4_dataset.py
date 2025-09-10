#!/usr/bin/env python3
"""
EVK4数据集生成器
从EVK4格式的H5文件生成DSEC格式的训练数据集
- noflare作为target（真值）
- sixflare作为input（带炫光）
- 时间对齐后随机采样100ms片段
- 空间裁剪重映射到640×480分辨率
- 每对数据使用不同的时间起点和裁剪区域
"""

import os
import json
import glob
import argparse
import time
import numpy as np

# 设置ECF插件路径
os.environ['HDF5_PLUGIN_PATH'] = '/home/lanpoknlanpokn/miniconda3/envs/Umain/lib/hdf5/plugin'

import h5py
from pathlib import Path
import random
from typing import Tuple, Dict

def read_evk4_events(hdf5_path: str) -> Dict[str, np.ndarray]:
    """
    读取EVK4格式的H5文件 - 使用基础方法
    """
    print(f"Reading EVK4 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        grp = f.get('CD', f)  # EVK4 HDF5 存储在 group 'CD'
        ev = grp['events']
        
        # 直接按照用户提供的方式读取
        x = ev['x'][:].astype(np.int32)
        y = ev['y'][:].astype(np.int32)
        p = ev['p'][:].astype(np.int8)
        ts = ev['t'][:].astype(np.float64)
        
        events = {
            'x': x,
            'y': y, 
            'p': p,
            't': ts
        }
        
        print(f"Successfully loaded {len(events['x'])} events")
        return events

def find_time_overlap(events1: Dict, events2: Dict) -> Tuple[float, float]:
    """
    找到两个事件流的时间重叠区间
    """
    start1, end1 = events1['t'][0], events1['t'][-1]
    start2, end2 = events2['t'][0], events2['t'][-1]
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    return overlap_start, overlap_end

def time_align_events(events: Dict, start_time: float, end_time: float) -> Dict[str, np.ndarray]:
    """
    根据时间范围对齐事件数据
    """
    t = events['t']
    mask = (t >= start_time) & (t <= end_time)
    
    aligned_events = {
        'x': events['x'][mask],
        'y': events['y'][mask], 
        'p': events['p'][mask],
        't': events['t'][mask] - start_time  # 时间戳归零
    }
    
    return aligned_events

def get_sensor_size(events: Dict) -> Tuple[int, int]:
    """
    获取传感器尺寸
    """
    max_x = int(events['x'].max())
    max_y = int(events['y'].max())
    return max_x + 1, max_y + 1

def crop_and_remap_events(events: Dict, sensor_size: Tuple[int, int], 
                         target_size: Tuple[int, int] = (640, 480)) -> Dict[str, np.ndarray]:
    """
    裁剪并重映射事件数据到目标分辨率
    
    Args:
        events: 原始事件数据
        sensor_size: 原始传感器尺寸 (width, height)
        target_size: 目标尺寸 (640, 480)
    
    Returns:
        重映射后的事件数据
    """
    orig_width, orig_height = sensor_size
    target_width, target_height = target_size
    
    # 计算裁剪区域，确保包含原中心点
    center_x = orig_width // 2
    center_y = orig_height // 2
    
    # 随机偏移，但保证裁剪区域包含中心点
    max_offset_x = min(center_x, orig_width - target_width - center_x) if orig_width > target_width else 0
    max_offset_y = min(center_y, orig_height - target_height - center_y) if orig_height > target_height else 0
    
    if max_offset_x > 0:
        offset_x = random.randint(-max_offset_x, max_offset_x)
    else:
        offset_x = 0
        
    if max_offset_y > 0:
        offset_y = random.randint(-max_offset_y, max_offset_y)
    else:
        offset_y = 0
    
    # 计算裁剪窗口
    crop_x_start = center_x - target_width // 2 + offset_x
    crop_x_end = crop_x_start + target_width
    crop_y_start = center_y - target_height // 2 + offset_y  
    crop_y_end = crop_y_start + target_height
    
    # 确保边界合法
    crop_x_start = max(0, min(crop_x_start, orig_width - target_width))
    crop_x_end = crop_x_start + target_width
    crop_y_start = max(0, min(crop_y_start, orig_height - target_height))
    crop_y_end = crop_y_start + target_height
    
    print(f"Crop region: x[{crop_x_start}:{crop_x_end}], y[{crop_y_start}:{crop_y_end}]")
    print(f"Original size: {orig_width}×{orig_height}, Target size: {target_width}×{target_height}")
    
    # 应用裁剪mask
    x = events['x']
    y = events['y']
    
    mask = (x >= crop_x_start) & (x < crop_x_end) & (y >= crop_y_start) & (y < crop_y_end)
    
    # 重映射坐标
    cropped_events = {
        'x': (x[mask] - crop_x_start).astype(np.uint16),
        'y': (y[mask] - crop_y_start).astype(np.uint16),
        'p': events['p'][mask],
        't': events['t'][mask]
    }
    
    return cropped_events

def save_dsec_format(events: Dict, output_path: str, sample_id: int):
    """
    保存为DSEC格式的H5文件
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换极性：确保0/1格式  
    polarity = events['p'].copy()
    polarity[polarity != 1] = 0  # 非1的都变成0
    
    # 转换时间戳为微秒
    timestamps_us = (events['t'] * 1e6).astype(np.int64)
    
    with h5py.File(output_path, 'w') as f:
        events_group = f.create_group('events')
        events_group.create_dataset('t', data=timestamps_us)
        events_group.create_dataset('x', data=events['x'])  
        events_group.create_dataset('y', data=events['y'])
        events_group.create_dataset('p', data=polarity)
        
    print(f"Saved sample {sample_id}: {len(events['x'])} events to {output_path}")

def generate_random_sample_times(overlap_start: float, overlap_end: float, 
                               num_samples: int = 10, sample_duration: float = 0.1) -> list:
    """
    在重叠区间内生成随机的采样时间点
    确保每个100ms样本都在有效范围内
    """
    available_duration = overlap_end - overlap_start - sample_duration
    
    if available_duration <= 0:
        raise ValueError(f"重叠区间太短({overlap_end - overlap_start:.3f}s)，无法生成{sample_duration}s的样本")
    
    sample_starts = []
    for i in range(num_samples):
        # 生成随机起始时间
        random_start = overlap_start + random.uniform(0, available_duration)
        sample_starts.append(random_start)
        print(f"Sample {i+1} time range: {random_start:.3f}s - {random_start + sample_duration:.3f}s")
    
    return sample_starts

def main():
    """主函数"""
    # 设置路径
    noflare_path = "Datasets/EVK4/full_noFlare_part_defocus.hdf5"
    sixflare_path = "Datasets/EVK4/full_sixFlare_part_defocus.hdf5" 
    output_dir = "evk4_dataset_dsec_format"
    
    # 创建输出目录
    input_dir = os.path.join(output_dir, "input")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    print("=== Loading EVK4 Data ===")
    print("Loading noflare events...")
    noflare_events = read_evk4_events(noflare_path)
    print(f"NoFlare events: {len(noflare_events['x'])} events")
    print(f"Time range: {noflare_events['t'][0]:.6f}s - {noflare_events['t'][-1]:.6f}s")
    
    print("Loading sixflare events...")
    sixflare_events = read_evk4_events(sixflare_path)  
    print(f"SixFlare events: {len(sixflare_events['x'])} events")
    print(f"Time range: {sixflare_events['t'][0]:.6f}s - {sixflare_events['t'][-1]:.6f}s")
    
    # 获取传感器尺寸
    sensor_size = get_sensor_size(noflare_events)
    print(f"Sensor size: {sensor_size[0]}×{sensor_size[1]}")
    
    print("\n=== Finding Time Overlap ===")
    overlap_start, overlap_end = find_time_overlap(noflare_events, sixflare_events)
    overlap_duration = overlap_end - overlap_start
    print(f"Time overlap: {overlap_start:.6f}s - {overlap_end:.6f}s")
    print(f"Overlap duration: {overlap_duration:.3f}s")
    
    # 生成随机采样时间点
    print("\n=== Generating Random Sample Times ===")
    sample_starts = generate_random_sample_times(overlap_start, overlap_end, num_samples=10)
    
    print("\n=== Processing Samples ===")
    for i, sample_start in enumerate(sample_starts):
        sample_end = sample_start + 0.1  # 100ms
        sample_id = i + 1
        
        print(f"\n--- Processing Sample {sample_id} ---")
        print(f"Time range: {sample_start:.3f}s - {sample_end:.3f}s")
        
        # 时间对齐
        noflare_aligned = time_align_events(noflare_events, sample_start, sample_end)
        sixflare_aligned = time_align_events(sixflare_events, sample_start, sample_end)
        
        print(f"Aligned events - NoFlare: {len(noflare_aligned['x'])}, SixFlare: {len(sixflare_aligned['x'])}")
        
        # 空间裁剪重映射（每个样本使用不同的随机裁剪区域）
        print("Applying spatial crop and remap...")
        noflare_cropped = crop_and_remap_events(noflare_aligned, sensor_size)
        sixflare_cropped = crop_and_remap_events(sixflare_aligned, sensor_size)
        
        print(f"Cropped events - NoFlare: {len(noflare_cropped['x'])}, SixFlare: {len(sixflare_cropped['x'])}")
        
        # 保存为DSEC格式
        target_file = os.path.join(target_dir, f"full_{int(sample_start*1000)}ms_sample{sample_id}.h5")
        input_file = os.path.join(input_dir, f"full_{int(sample_start*1000)}ms_sample{sample_id}.h5")
        
        save_dsec_format(noflare_cropped, target_file, f"target_{sample_id}")
        save_dsec_format(sixflare_cropped, input_file, f"input_{sample_id}")
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Generated 10 pairs of training data")
    print(f"Target files (clean): {target_dir}/")  
    print(f"Input files (with flare): {input_dir}/")
    print(f"Format: DSEC-compatible HDF5")
    print(f"Resolution: 640×480")
    print(f"Duration per sample: 100ms")

if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    random.seed(42)
    np.random.seed(42)
    
    main()