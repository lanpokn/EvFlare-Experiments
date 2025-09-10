#!/usr/bin/env python3
"""
从NPY格式的EVK4数据生成DSEC格式的训练数据集
- noflare作为target（真值）
- sixflare作为input（带炫光）
- 时间对齐后随机采样100ms片段
- 空间裁剪重映射到640×480分辨率
- 每对数据使用不同的时间起点和裁剪区域
"""

import os
import json
import numpy as np
import h5py
from pathlib import Path
import random
from typing import Tuple, Dict, Optional, List

def read_evk4_npy_events(npy_folder: str) -> Dict[str, np.ndarray]:
    """
    读取NPY格式的EVK4事件数据
    """
    print(f"Reading NPY folder: {npy_folder}")
    
    # 读取事件数据
    events_ts = np.load(os.path.join(npy_folder, 'events_ts.npy'))
    events_xy = np.load(os.path.join(npy_folder, 'events_xy.npy'))
    events_p = np.load(os.path.join(npy_folder, 'events_p.npy'))
    
    # 读取元数据
    with open(os.path.join(npy_folder, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    sensor_resolution = metadata.get('sensor_resolution', [720, 1280])  # [height, width]
    
    # 提取x, y坐标
    x = events_xy[:, 0].astype(np.int32)
    y = events_xy[:, 1].astype(np.int32)
    
    # 时间戳已经是秒为单位
    t = events_ts.astype(np.float64)
    
    # 极性数据 - 检查格式并标准化为-1/+1
    p = events_p.astype(np.int8)
    unique_polarities = np.unique(p)
    print(f"Original polarity values: {unique_polarities}")
    
    # 标准化极性值：确保是-1和+1
    if set(unique_polarities) == {0, 1}:
        # 如果是0/1格式，转换为-1/+1
        p = p * 2 - 1
        print("Converted polarities from {0,1} to {-1,+1}")
    elif set(unique_polarities) != {-1, 1}:
        print(f"Warning: Unexpected polarity values: {unique_polarities}")
    
    events = {
        'x': x,
        'y': y,
        'p': p,
        't': t,
        'sensor_size': tuple(reversed(sensor_resolution))  # (width, height)
    }
    
    print(f"Successfully loaded {len(x)} events")
    print(f"Sensor size: {events['sensor_size']} (W×H)")
    print(f"Time range: {t.min():.3f}s - {t.max():.3f}s")
    print(f"Duration: {t.max() - t.min():.3f}s")
    print(f"X range: {x.min()} - {x.max()}")
    print(f"Y range: {y.min()} - {y.max()}")
    print(f"Final polarity values: {np.unique(p)}")
    
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

def calculate_polarity_ratios(events: Dict, start_time: float) -> List[float]:
    """
    计算四个时间窗口的正极性占比
    时间窗口: 0-2.5ms, 2.5-5ms, 5-7.5ms, 7.5-10ms
    
    Args:
        events: 事件数据
        start_time: 开始时间
    
    Returns:
        四个时间窗口的正极性占比列表
    """
    time_windows = [
        (start_time, start_time + 0.0025),      # 0-2.5ms
        (start_time + 0.0025, start_time + 0.005),   # 2.5-5ms 
        (start_time + 0.005, start_time + 0.0075),   # 5-7.5ms
        (start_time + 0.0075, start_time + 0.01)     # 7.5-10ms
    ]
    
    ratios = []
    t = events['t']
    p = events['p']
    
    for window_start, window_end in time_windows:
        # 提取时间窗口内的事件
        mask = (t >= window_start) & (t < window_end)
        window_events = p[mask]
        
        if len(window_events) == 0:
            ratios.append(0.5)  # 默认50%，避免空窗口问题
        else:
            # 计算正极性占比
            positive_count = np.sum(window_events == 1)
            ratio = positive_count / len(window_events)
            ratios.append(ratio)
    
    return ratios

def find_optimal_alignment_polarity(events1: Dict, events2: Dict, sample_start: float) -> float:
    """
    使用正极性占比进行时间对齐
    
    Args:
        events1: 参考事件流（通常是干净数据），固定从sample_start开始
        events2: 待对齐事件流（通常是带炫光数据），测试不同起始点
        sample_start: 基准采样起始时间
    
    Returns:
        最佳偏移量（秒）
    """
    
    # 计算参考数据的正极性占比（固定从sample_start开始）
    ref_ratios = calculate_polarity_ratios(events1, sample_start)
    print(f"参考数据正极性占比: {[f'{r:.3f}' for r in ref_ratios]}")
    
    best_offset = 0.0
    best_difference = float('inf')
    
    # 搜索20步，每步0.5ms
    for step in range(20):
        offset = step * 0.0005  # 0.5ms步长
        test_start = sample_start + offset
        
        # 确保在有效时间范围内
        if test_start + 0.01 > events2['t'][-1]:  # 需要10ms的数据
            break
            
        # 计算测试数据的正极性占比
        test_ratios = calculate_polarity_ratios(events2, test_start)
        
        # 计算四个窗口占比差距之和
        difference = sum(abs(r1 - r2) for r1, r2 in zip(ref_ratios, test_ratios))
        
        print(f"偏移 {offset*1000:.1f}ms: 测试占比 {[f'{r:.3f}' for r in test_ratios]}, 差距和: {difference:.6f}")
        
        if difference < best_difference:
            best_difference = difference
            best_offset = offset
    
    print(f"最佳对齐偏移: {best_offset*1000:.1f}ms, 最小差距和: {best_difference:.6f}")
    return best_offset

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
        't': events['t'][mask] - start_time,  # 时间戳归零
        'sensor_size': events['sensor_size']
    }
    
    return aligned_events

def crop_and_remap_events(events: Dict, target_size: Tuple[int, int] = (640, 480), 
                         crop_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    裁剪并重映射事件数据到目标分辨率
    
    Args:
        events: 原始事件数据
        target_size: 目标尺寸 (640, 480)
        crop_seed: 随机种子，确保每对数据使用不同的裁剪区域
    
    Returns:
        重映射后的事件数据
    """
    # 如果提供了种子，设置随机种子以获得不同的裁剪区域
    if crop_seed is not None:
        np.random.seed(crop_seed)
        random.seed(crop_seed)
    orig_width, orig_height = events['sensor_size']
    target_width, target_height = target_size
    
    # 计算裁剪区域，确保包含原中心点
    center_x = orig_width // 2
    center_y = orig_height // 2
    
    # 计算最大允许的随机偏移范围
    # 裁剪窗口的左上角可以在 (0, 0) 到 (orig_width-target_width, orig_height-target_height) 之间
    # 中心裁剪的起始点是 (center_x - target_width//2, center_y - target_height//2)
    center_crop_x_start = center_x - target_width // 2
    center_crop_y_start = center_y - target_height // 2
    
    max_offset_x = min(center_crop_x_start, (orig_width - target_width) - center_crop_x_start) if orig_width > target_width else 0
    max_offset_y = min(center_crop_y_start, (orig_height - target_height) - center_crop_y_start) if orig_height > target_height else 0
    
    print(f"Max offset range - X: ±{max_offset_x}, Y: ±{max_offset_y}")
    
    if max_offset_x > 0:
        offset_x = random.randint(-max_offset_x, max_offset_x)
    else:
        offset_x = 0
        
    if max_offset_y > 0:
        offset_y = random.randint(-max_offset_y, max_offset_y)
    else:
        offset_y = 0
    
    print(f"Applied offset - X: {offset_x}, Y: {offset_y}")
    
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
    
    print(f"Events after cropping: {len(cropped_events['x'])} (from {len(x)})")
    
    return cropped_events

def save_dsec_format(events: Dict, output_path: str, sample_id: int):
    """
    保存为DSEC格式的H5文件
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换极性：DSEC格式使用0/1
    polarity = events['p'].copy()
    polarity[polarity == -1] = 0  # -1变成0
    polarity[polarity == 1] = 1   # +1保持1
    
    # 转换时间戳为微秒
    timestamps_us = (events['t'] * 1e6).astype(np.int64)
    
    with h5py.File(output_path, 'w') as f:
        events_group = f.create_group('events')
        # 使用gzip压缩，与DSEC格式一致
        events_group.create_dataset('t', data=timestamps_us, compression='gzip', compression_opts=6)
        events_group.create_dataset('x', data=events['x'], compression='gzip', compression_opts=6)  
        events_group.create_dataset('y', data=events['y'], compression='gzip', compression_opts=6)
        events_group.create_dataset('p', data=polarity, compression='gzip', compression_opts=6)
        
    print(f"Saved sample {sample_id}: {len(events['x'])} events to {output_path}")

def generate_random_sample_times(overlap_start: float, overlap_end: float, 
                               num_samples: int = 10, sample_duration: float = 0.1) -> list:
    """
    在重叠区间内生成均匀分布的采样时间点
    确保样本在整个可用时间范围内均匀分布
    """
    available_duration = overlap_end - overlap_start - sample_duration
    
    if available_duration <= 0:
        raise ValueError(f"重叠区间太短({overlap_end - overlap_start:.3f}s)，无法生成{sample_duration}s的样本")
    
    # 使用均匀分布而非完全随机，确保覆盖整个时间范围
    sample_starts = []
    
    # 将可用时间分成num_samples个区间，每个区间内随机选择
    interval_size = available_duration / num_samples
    
    for i in range(num_samples):
        # 每个样本在其分配的时间区间内随机选择起始点
        interval_start = overlap_start + i * interval_size
        interval_end = interval_start + interval_size
        
        # 在区间内随机选择（如果区间太小就取区间开始）
        if interval_size > 0.01:  # 如果区间>10ms，随机选择
            random_start = random.uniform(interval_start, min(interval_end, overlap_start + available_duration))
        else:
            random_start = interval_start
            
        sample_starts.append(random_start)
        print(f"Sample {i+1} time range: {random_start:.3f}s - {random_start + sample_duration:.3f}s")
    
    return sorted(sample_starts)  # 按时间排序

def main():
    """主函数"""
    # 设置路径
    noflare_npy_folder = "Datasets/EVK4/full_noFlare_part_defocus"
    sixflare_npy_folder = "Datasets/EVK4/full_sixFlare_part_defocus" 
    output_dir = "evk4_dataset_dsec_format_from_npy"
    
    # 创建输出目录
    input_dir = os.path.join(output_dir, "input")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    print("=== Loading EVK4 NPY Data ===")
    print("Loading noflare events...")
    noflare_events = read_evk4_npy_events(noflare_npy_folder)
    
    print("\nLoading sixflare events...")
    sixflare_events = read_evk4_npy_events(sixflare_npy_folder)
    
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
        
        # 极性占比对齐：找到最佳时间偏移
        print("Finding optimal alignment using polarity ratios...")
        try:
            best_offset = find_optimal_alignment_polarity(noflare_events, sixflare_events, sample_start)
        except Exception as e:
            print(f"警告: 极性对齐失败，使用零偏移: {e}")
            best_offset = 0.0
        
        # 应用指标对齐
        noflare_aligned = time_align_events(noflare_events, sample_start, sample_end)
        sixflare_aligned = time_align_events(sixflare_events, sample_start + best_offset, sample_end + best_offset)
        
        print(f"Aligned events - NoFlare: {len(noflare_aligned['x'])}, SixFlare: {len(sixflare_aligned['x'])}")
        
        # 空间裁剪重映射（每个样本使用不同的随机裁剪区域）
        print("Applying spatial crop and remap...")
        crop_seed = sample_id * 42  # 每对数据使用不同的裁剪种子
        noflare_cropped = crop_and_remap_events(noflare_aligned, crop_seed=crop_seed)
        sixflare_cropped = crop_and_remap_events(sixflare_aligned, crop_seed=crop_seed)
        
        print(f"Final events - NoFlare: {len(noflare_cropped['x'])}, SixFlare: {len(sixflare_cropped['x'])}")
        
        # 保存为DSEC格式
        target_file = os.path.join(target_dir, f"evk4_{int(sample_start*1000)}ms_sample{sample_id}.h5")
        input_file = os.path.join(input_dir, f"evk4_{int(sample_start*1000)}ms_sample{sample_id}.h5")
        
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