#!/usr/bin/env python3
"""
RedAndFull EVK4 Dataset Generator

专门用于处理redandfull组数据的数据集生成器。
与defocus版本的主要区别：使用任意四边形滤波区域。

数据源：
- NoFlare: Datasets/EVK4/redandfull_noFlare_part  
- SixFlare: Datasets/EVK4/redandfull_sixFlare_part
- RandomFlare: Datasets/EVK4/redandfull_randomFlare_part

处理流程：
1. 读取NPY格式的EVK4事件数据(三个数据源)
2. NoFlare数据: 四边形光源区域平滑滤波 - 顶点[(600,320),(780,320),(560,470),(715,490)], 50px平滑过渡
3. NoFlare数据: 添加时空随机噪声 (100万个随机事件，10倍增强)
4. 两种炫光数据: 边界平滑滤除 - 移除两个矩形区域(左侧x<335 + 右上角x>910且y<255), 50px平滑过渡
5. 两种炫光数据: 分别添加时空随机噪声 (各100万个随机事件，10倍增强)
6. 基于极性占比的时间对齐
7. 随机采样20个100ms数据段(前10个用SixFlare，后10个用RandomFlare)
8. 空间裁剪重映射 (1280×720 → 640×480)
9. 保存为DSEC格式H5文件

输出：
- 目录: evk4_redandfull_dataset_dsec_format_from_npy/
- 格式: redandfull_XXXms_sampleY.h5
"""

import os
import json
import math
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

def sort_vertices_clockwise(vertices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    将四边形顶点按照顺时针顺序排序，确保形成真正的四边形
    
    Args:
        vertices: 四个顶点的列表
        
    Returns:
        按顺时针顺序排列的顶点列表
    """
    # 计算中心点
    center_x = sum(v[0] for v in vertices) / len(vertices)
    center_y = sum(v[1] for v in vertices) / len(vertices)
    
    # 计算每个顶点相对于中心的角度
    def angle_from_center(vertex):
        x, y = vertex
        return math.atan2(y - center_y, x - center_x)
    
    # 按角度排序（顺时针）
    sorted_vertices = sorted(vertices, key=angle_from_center)
    return sorted_vertices

def point_in_quadrilateral(x: np.ndarray, y: np.ndarray, vertices: List[Tuple[int, int]]) -> np.ndarray:
    """
    使用射线法判断点是否在四边形内部
    确保顶点按正确顺序连接形成真正的四边形
    
    Args:
        x: 点的X坐标数组
        y: 点的Y坐标数组  
        vertices: 四边形顶点列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    Returns:
        布尔数组，True表示点在四边形内
    """
    if len(vertices) != 4:
        raise ValueError("必须提供4个顶点来定义四边形")
    
    # 确保顶点按正确顺序排列，形成真正的四边形
    sorted_vertices = sort_vertices_clockwise(vertices)
    
    n = len(sorted_vertices)
    inside = np.zeros(len(x), dtype=bool)
    
    # 射线法：从每个点向右发射射线，统计与四边形边的交点数
    for i in range(n):
        p1x, p1y = sorted_vertices[i]
        p2x, p2y = sorted_vertices[(i + 1) % n]  # 下一个顶点，最后一个连接到第一个
        
        # 避免除零错误
        if p1y == p2y:
            continue
            
        # 检查射线与边的交点
        # 条件1：边跨越射线（一个顶点在射线上方，另一个在下方）
        cond1 = (p1y > y) != (p2y > y)
        
        # 条件2：交点在点的右侧
        # 计算射线与边的交点的x坐标
        intersect_x = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
        cond2 = x < intersect_x
        
        # 如果两个条件都满足，计数器加1
        inside ^= cond1 & cond2  # 异或操作相当于奇偶计数
    
    return inside

def distance_to_polygon(x: np.ndarray, y: np.ndarray, vertices: List[Tuple[int, int]]) -> np.ndarray:
    """
    计算点到多边形边界的最短距离（带符号）
    内部为负，外部为正
    
    Args:
        x: 点的X坐标数组
        y: 点的Y坐标数组
        vertices: 多边形顶点列表，已按顺序排列
        
    Returns:
        到边界的带符号距离数组
    """
    n = len(vertices)
    min_distances = np.full(len(x), np.inf)
    
    # 计算到每条边的距离
    for i in range(n):
        p1x, p1y = vertices[i]
        p2x, p2y = vertices[(i + 1) % n]
        
        # 计算点到线段的距离
        # 向量 AB = (p2x - p1x, p2y - p1y)
        # 向量 AP = (x - p1x, y - p1y)
        
        ab_x = p2x - p1x
        ab_y = p2y - p1y
        ap_x = x - p1x
        ap_y = y - p1y
        
        # 计算投影参数 t
        ab_squared = ab_x * ab_x + ab_y * ab_y
        if ab_squared == 0:  # 退化为点
            distance = np.sqrt((x - p1x) ** 2 + (y - p1y) ** 2)
        else:
            t = (ap_x * ab_x + ap_y * ab_y) / ab_squared
            t = np.clip(t, 0, 1)  # 限制在线段范围内
            
            # 计算最近点
            closest_x = p1x + t * ab_x
            closest_y = p1y + t * ab_y
            
            # 计算距离
            distance = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
        
        # 更新最小距离
        min_distances = np.minimum(min_distances, distance)
    
    # 确定内外部符号
    inside_mask = point_in_quadrilateral(x, y, vertices)
    signed_distances = np.where(inside_mask, -min_distances, min_distances)
    
    return signed_distances

def filter_light_source_quadrilateral_smooth(events: Dict, vertices: List[Tuple[int, int]] = None, 
                                           transition_pixels: int = 50) -> Dict[str, np.ndarray]:
    """
    四边形光源区域平滑滤波，在边界处实现平滑过渡
    四边形内部完全保留，边界外逐渐增加滤除概率，50像素外完全滤除
    
    Args:
        events: 原始事件数据
        vertices: 四边形顶点列表，默认使用redandfull组的四边形顶点
        transition_pixels: 平滑过渡的像素范围 (默认50像素)
    
    Returns:
        滤波后的事件数据，包含平滑过渡效果
    """
    if vertices is None:
        # 使用收缩后的redandfull组四边形顶点（整体向内收缩约20像素）
        vertices = [(600, 320), (780, 320), (560, 470), (715, 490)]
    
    if len(vertices) != 4:
        raise ValueError("必须提供4个顶点来定义四边形")
    
    x = events['x']
    y = events['y']
    
    # 确保顶点按正确顺序排列
    sorted_vertices = sort_vertices_clockwise(vertices)
    
    # 计算每个点到四边形边界的带符号距离
    distances = distance_to_polygon(x, y, sorted_vertices)
    
    # 计算保留概率
    # 内部（距离<0）：完全保留（概率=1）
    # 边界到transition_pixels：线性递减（概率从1降到0）
    # 外部（距离>transition_pixels）：完全滤除（概率=0）
    
    probabilities = np.ones(len(distances))
    
    # 在过渡区域应用线性递减
    transition_mask = (distances >= 0) & (distances <= transition_pixels)
    probabilities[transition_mask] = 1.0 - (distances[transition_mask] / transition_pixels)
    
    # 外部区域完全滤除
    outer_mask = distances > transition_pixels
    probabilities[outer_mask] = 0.0
    
    # 根据概率随机决定保留哪些事件
    random_values = np.random.random(len(probabilities))
    keep_mask = random_values < probabilities
    
    filtered_events = {
        'x': x[keep_mask],
        'y': y[keep_mask],
        'p': events['p'][keep_mask],
        't': events['t'][keep_mask],
        'sensor_size': events['sensor_size']
    }
    
    original_count = len(events['x'])
    filtered_count = len(filtered_events['x'])
    filter_ratio = filtered_count / original_count if original_count > 0 else 0
    
    # 统计不同区域的事件数量
    inside_count = np.sum(distances < 0)
    transition_count = np.sum(transition_mask)
    outside_count = np.sum(outer_mask)
    kept_transition = np.sum(keep_mask & transition_mask)
    
    print(f"Light source quadrilateral smooth filtering:")
    print(f"  Original vertices: {vertices}")
    print(f"  Sorted vertices (clockwise): {sorted_vertices}")
    print(f"  Transition zone: {transition_pixels} pixels")
    print(f"  Event distribution:")
    print(f"    Inside (fully kept): {inside_count}")
    print(f"    Transition zone: {transition_count} → {kept_transition} kept ({kept_transition/transition_count:.1%} avg retention)" if transition_count > 0 else "    Transition zone: 0")
    print(f"    Outside (fully removed): {outside_count}")
    print(f"  {original_count} → {filtered_count} events ({filter_ratio:.3%} retained)")
    
    # 计算四边形边界框用于显示
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices] 
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    print(f"  Bounding box: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")
    
    return filtered_events

def distance_to_flare_boundaries(x: np.ndarray, y: np.ndarray, left_boundary: int = 335, 
                                right_boundary: int = 910, top_boundary: int = 255) -> np.ndarray:
    """
    计算点到炫光边界滤除区域的距离
    返回正值表示在要被移除的区域内，负值表示在保留区域内
    
    Args:
        x: 点的X坐标数组
        y: 点的Y坐标数组
        left_boundary: 左边界
        right_boundary: 右边界  
        top_boundary: 上边界
        
    Returns:
        到边界的带符号距离数组，正值=在移除区域内，负值=在保留区域内
    """
    # 计算到左侧边界的距离（左侧矩形）
    left_distance = left_boundary - x  # x < left_boundary 时为正（要移除）
    
    # 计算到右上角矩形边界的距离
    # 右上角矩形定义：x > right_boundary AND y < top_boundary
    right_distance = x - right_boundary  # x > right_boundary 时为正
    top_distance = top_boundary - y      # y < top_boundary 时为正
    
    # 右上角矩形的距离：只有当同时满足两个条件时才在移除区域内
    right_top_in_remove = (x > right_boundary) & (y < top_boundary)
    right_top_distance = np.where(right_top_in_remove, 
                                  np.maximum(right_distance, top_distance),  # 取最大距离
                                  -np.inf)  # 不在右上角矩形内
    
    # 综合距离：取两个移除区域中距离更近的那个
    # 如果在左侧移除区域内，距离为left_distance
    # 如果在右上角移除区域内，距离为right_top_distance
    # 否则在保留区域内，距离为负值
    
    in_left_remove = (x < left_boundary)
    in_right_top_remove = right_top_in_remove
    
    distances = np.full(len(x), -np.inf)  # 默认在保留区域内（负值）
    
    # 在左侧移除区域内
    distances[in_left_remove] = left_distance[in_left_remove]
    
    # 在右上角移除区域内
    distances[in_right_top_remove] = right_top_distance[in_right_top_remove]
    
    # 对于保留区域的点，计算到最近边界的负距离
    in_keep_region = ~(in_left_remove | in_right_top_remove)
    if np.any(in_keep_region):
        # 计算保留区域内的点到边界的距离（负值）
        keep_x = x[in_keep_region]
        keep_y = y[in_keep_region]
        
        # 到左边界的距离
        dist_to_left = keep_x - left_boundary  # 正值，但要转为负值
        
        # 到右上角区域边界的距离
        dist_to_right_boundary = right_boundary - keep_x
        dist_to_top_boundary = keep_y - top_boundary
        
        # 对于右上角区域，计算到边界的最小距离
        dist_to_right_top = np.where((keep_x <= right_boundary) | (keep_y >= top_boundary),
                                    np.minimum(np.maximum(dist_to_right_boundary, 0), 
                                             np.maximum(dist_to_top_boundary, 0)),
                                    0)
        
        # 取到各个边界的最小距离，转为负值
        min_dist_to_boundary = np.minimum(dist_to_left, dist_to_right_top)
        distances[in_keep_region] = -min_dist_to_boundary
    
    return distances

def filter_flare_boundaries_smooth(events: Dict, left_boundary: int = 335, right_boundary: int = 910, 
                                 top_boundary: int = 255, transition_pixels: int = 50) -> Dict[str, np.ndarray]:
    """
    炫光数据边界平滑滤除，在边界处实现平滑过渡
    移除两个不相连的矩形区域，但边界处有50像素的平滑过渡
    
    Args:
        events: 原始炫光事件数据
        left_boundary: 左边界，移除 x < left_boundary 的事件 (默认335)
        right_boundary: 右边界，移除 x > right_boundary AND y < top_boundary 的事件 (默认910)
        top_boundary: 上边界，移除 x > right_boundary AND y < top_boundary 的事件 (默认255)
        transition_pixels: 平滑过渡的像素范围 (默认50像素)
    
    Returns:
        滤波后的事件数据，包含平滑过渡效果
    """
    x = events['x']
    y = events['y']
    
    # 计算每个点到移除区域边界的带符号距离
    distances = distance_to_flare_boundaries(x, y, left_boundary, right_boundary, top_boundary)
    
    # 计算移除概率
    # 深入移除区域（距离>transition_pixels）：完全移除（概率=1）
    # 边界到transition_pixels：线性递增移除概率（概率从0增到1）
    # 保留区域（距离<0）：完全保留（概率=0）
    
    removal_probabilities = np.zeros(len(distances))
    
    # 在过渡区域应用线性递增的移除概率
    transition_mask = (distances >= 0) & (distances <= transition_pixels)
    removal_probabilities[transition_mask] = distances[transition_mask] / transition_pixels
    
    # 深入移除区域完全移除
    deep_remove_mask = distances > transition_pixels
    removal_probabilities[deep_remove_mask] = 1.0
    
    # 根据概率随机决定移除哪些事件
    random_values = np.random.random(len(removal_probabilities))
    keep_mask = random_values >= removal_probabilities  # 保留概率 = 1 - 移除概率
    
    filtered_events = {
        'x': x[keep_mask],
        'y': y[keep_mask],
        'p': events['p'][keep_mask],
        't': events['t'][keep_mask],
        'sensor_size': events['sensor_size']
    }
    
    original_count = len(events['x'])
    filtered_count = len(filtered_events['x'])
    filter_ratio = filtered_count / original_count if original_count > 0 else 0
    
    # 统计不同区域的事件数量
    keep_region_count = np.sum(distances < 0)
    transition_count = np.sum(transition_mask)
    deep_remove_count = np.sum(deep_remove_mask)
    kept_transition = np.sum(keep_mask & transition_mask)
    
    print(f"Flare boundary smooth filtering:")
    print(f"  Boundaries: left x<{left_boundary}, right-top x>{right_boundary}&y<{top_boundary}")
    print(f"  Transition zone: {transition_pixels} pixels")
    print(f"  Event distribution:")
    print(f"    Keep region (fully retained): {keep_region_count}")
    print(f"    Transition zone: {transition_count} → {kept_transition} kept ({kept_transition/transition_count:.1%} avg retention)" if transition_count > 0 else "    Transition zone: 0")
    print(f"    Deep remove region (fully removed): {deep_remove_count}")
    print(f"  {original_count} → {filtered_count} events ({filter_ratio:.3%} retained)")
    
    return filtered_events

def filter_flare_boundaries(events: Dict, left_boundary: int = 335, right_boundary: int = 910, top_boundary: int = 255) -> Dict[str, np.ndarray]:
    """
    炫光数据边界滤除，移除两个不相连的矩形区域（硬边界版本，向后兼容）
    """
    return filter_flare_boundaries_smooth(events, left_boundary, right_boundary, top_boundary, transition_pixels=0)

def add_noise_to_filtered_regions(events: Dict, filtered_mask: np.ndarray, num_noise_events: int = 1000000, 
                                 sensor_size: Tuple[int, int] = (1280, 720)) -> Dict[str, np.ndarray]:
    """
    在被滤除的区域添加时空随机分布的噪声事件
    
    Args:
        events: 原始事件数据
        filtered_mask: 被保留事件的mask，~filtered_mask 就是被滤除的区域
        num_noise_events: 要添加的噪声事件数量 (默认100万，是原来的10倍)
        sensor_size: 传感器尺寸 (width, height)
    
    Returns:
        添加噪声后的事件数据
    """
    if len(events['x']) == 0:
        print(f"Warning: No original events to add noise to")
        return events
        
    # 获取原始数据的时间范围
    t_min, t_max = events['t'].min(), events['t'].max()
    width, height = sensor_size
    
    # 生成随机噪声事件
    # 时间: 在原始时间范围内均匀分布
    noise_t = np.random.uniform(t_min, t_max, num_noise_events).astype(np.float64)
    
    # 空间: 在整个传感器区域内均匀分布，但主要集中在被滤除区域
    noise_x = np.random.randint(0, width, num_noise_events).astype(np.int32)
    noise_y = np.random.randint(0, height, num_noise_events).astype(np.int32)
    
    # 极性: 随机选择 -1 或 +1
    noise_p = np.random.choice([-1, 1], num_noise_events).astype(np.int8)
    
    # 合并原始事件和噪声事件
    combined_events = {
        'x': np.concatenate([events['x'], noise_x]),
        'y': np.concatenate([events['y'], noise_y]), 
        'p': np.concatenate([events['p'], noise_p]),
        't': np.concatenate([events['t'], noise_t]),
        'sensor_size': events['sensor_size']
    }
    
    # 按时间排序
    sort_indices = np.argsort(combined_events['t'])
    for key in ['x', 'y', 'p', 't']:
        combined_events[key] = combined_events[key][sort_indices]
    
    original_count = len(events['x'])
    final_count = len(combined_events['x'])
    
    print(f"Spatiotemporal noise addition to filtered regions:")
    print(f"  Original events: {original_count}")
    print(f"  Added noise events: {num_noise_events} (10x increase)")
    print(f"  Final events: {final_count}")
    print(f"  Time range: {t_min:.6f}s - {t_max:.6f}s")
    print(f"  Noise primarily in filtered (removed) regions")
    
    return combined_events

def add_spatiotemporal_noise(events: Dict, num_noise_events: int = 1000000, 
                            sensor_size: Tuple[int, int] = (1280, 720)) -> Dict[str, np.ndarray]:
    """
    在事件数据上添加时空随机分布的噪声事件（兼容性函数，数量已增加10倍）
    
    Args:
        events: 原始事件数据
        num_noise_events: 要添加的噪声事件数量 (默认100万，是原来的10倍)
        sensor_size: 传感器尺寸 (width, height)
    
    Returns:
        添加噪声后的事件数据
    """
    if len(events['x']) == 0:
        print(f"Warning: No original events to add noise to")
        return events
        
    # 获取原始数据的时间范围
    t_min, t_max = events['t'].min(), events['t'].max()
    width, height = sensor_size
    
    # 生成随机噪声事件
    # 时间: 在原始时间范围内均匀分布
    noise_t = np.random.uniform(t_min, t_max, num_noise_events).astype(np.float64)
    
    # 空间: 在整个传感器区域内均匀分布  
    noise_x = np.random.randint(0, width, num_noise_events).astype(np.int32)
    noise_y = np.random.randint(0, height, num_noise_events).astype(np.int32)
    
    # 极性: 随机选择 -1 或 +1
    noise_p = np.random.choice([-1, 1], num_noise_events).astype(np.int8)
    
    # 合并原始事件和噪声事件
    combined_events = {
        'x': np.concatenate([events['x'], noise_x]),
        'y': np.concatenate([events['y'], noise_y]), 
        'p': np.concatenate([events['p'], noise_p]),
        't': np.concatenate([events['t'], noise_t]),
        'sensor_size': events['sensor_size']
    }
    
    # 按时间排序
    sort_indices = np.argsort(combined_events['t'])
    for key in ['x', 'y', 'p', 't']:
        combined_events[key] = combined_events[key][sort_indices]
    
    original_count = len(events['x'])
    final_count = len(combined_events['x'])
    
    print(f"Spatiotemporal noise addition:")
    print(f"  Original events: {original_count}")
    print(f"  Added noise events: {num_noise_events} (10x increase)")
    print(f"  Final events: {final_count}")
    print(f"  Time range: {t_min:.6f}s - {t_max:.6f}s")
    
    return combined_events

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
    # 设置路径 - redandfull数据（双炫光源）
    noflare_npy_folder = "Datasets/EVK4/redandfull_noFlare_part"
    sixflare_npy_folder = "Datasets/EVK4/redandfull_sixFlare_part"  # 六炫光数据
    randomflare_npy_folder = "Datasets/EVK4/redandfull_randomFlare_part"  # 随机炫光数据
    output_dir = "evk4_redandfull_dataset_dsec_format_from_npy"  # redandfull输出目录名
    
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
    
    print("\nLoading randomflare events...")
    randomflare_events = read_evk4_npy_events(randomflare_npy_folder)
    
    print("\n=== Finding Time Overlap (NoFlare as baseline) ===")
    # 计算NoFlare与两种炫光数据的重叠区间
    overlap_start_six, overlap_end_six = find_time_overlap(noflare_events, sixflare_events)
    overlap_start_random, overlap_end_random = find_time_overlap(noflare_events, randomflare_events)
    
    print(f"NoFlare vs SixFlare overlap: {overlap_start_six:.6f}s - {overlap_end_six:.6f}s ({overlap_end_six - overlap_start_six:.3f}s)")
    print(f"NoFlare vs RandomFlare overlap: {overlap_start_random:.6f}s - {overlap_end_random:.6f}s ({overlap_end_random - overlap_start_random:.3f}s)")
    print("Will use NoFlare as baseline and align each flare type separately during processing")
    
    # 先对noFlare数据应用四边形光源平滑滤波（统一在时间采样前做）
    print("\n=== Applying Quadrilateral Light Source Smooth Filtering to NoFlare Data ===")
    # 使用收缩后的四边形顶点：(600,320), (780,320), (560,470), (715,490)
    # 整体向内收缩约20像素，50像素平滑过渡区域
    quadrilateral_vertices = [(600, 320), (780, 320), (560, 470), (715, 490)]
    noflare_events = filter_light_source_quadrilateral_smooth(noflare_events, vertices=quadrilateral_vertices, transition_pixels=50)
    
    # 添加时空随机噪声（100万个，10倍增加）
    print("\n=== Adding Spatiotemporal Noise to NoFlare Data (10x increase) ===")
    noflare_events = add_spatiotemporal_noise(noflare_events, num_noise_events=1000000, 
                                            sensor_size=noflare_events['sensor_size'])
    
    # 对两种炫光数据应用相同的平滑边界滤除和噪声增强
    print("\n=== Applying Smooth Boundary Filtering to SixFlare Data ===")
    # 移除两个矩形区域：1) x<335的左侧区域  2) x>910且y<255的右上角区域
    # 50像素平滑过渡区域
    sixflare_events = filter_flare_boundaries_smooth(sixflare_events, left_boundary=335, right_boundary=910, top_boundary=255, transition_pixels=50)
    
    print("\n=== Adding Spatiotemporal Noise to SixFlare Data (10x increase) ===")
    sixflare_events = add_spatiotemporal_noise(sixflare_events, num_noise_events=1000000,
                                             sensor_size=sixflare_events['sensor_size'])
    
    print("\n=== Applying Smooth Boundary Filtering to RandomFlare Data ===")
    randomflare_events = filter_flare_boundaries_smooth(randomflare_events, left_boundary=335, right_boundary=910, top_boundary=255, transition_pixels=50)
    
    print("\n=== Adding Spatiotemporal Noise to RandomFlare Data (10x increase) ===")
    randomflare_events = add_spatiotemporal_noise(randomflare_events, num_noise_events=1000000,
                                                sensor_size=randomflare_events['sensor_size'])
    
    # 生成随机采样时间点 - 基于NoFlare数据时间范围生成20对数据
    print("\n=== Generating Random Sample Times (based on NoFlare timeline) ===")
    noflare_start, noflare_end = noflare_events['t'][0], noflare_events['t'][-1]
    print(f"NoFlare time range: {noflare_start:.6f}s - {noflare_end:.6f}s ({noflare_end - noflare_start:.3f}s)")
    sample_starts = generate_random_sample_times(noflare_start, noflare_end, num_samples=20)
    
    print("\n=== Processing Samples (20 pairs: 10 SixFlare + 10 RandomFlare) ===")
    for i, sample_start in enumerate(sample_starts):
        sample_end = sample_start + 0.1  # 100ms
        sample_id = i + 1  # 从1开始编号
        
        # 前10个使用sixFlare，后10个使用randomFlare
        if i < 10:
            flare_type = "sixflare"
            flare_events = sixflare_events
            print(f"\n--- Processing Sample {sample_id} (SixFlare) ---")
        else:
            flare_type = "randomflare" 
            flare_events = randomflare_events
            print(f"\n--- Processing Sample {sample_id} (RandomFlare) ---")
        
        print(f"Time range: {sample_start:.3f}s - {sample_end:.3f}s")
        
        # 极性占比对齐：找到最佳时间偏移
        print("Finding optimal alignment using polarity ratios...")
        try:
            best_offset = find_optimal_alignment_polarity(noflare_events, flare_events, sample_start)
        except Exception as e:
            print(f"警告: 极性对齐失败，使用零偏移: {e}")
            best_offset = 0.0
        
        # 应用指标对齐
        noflare_aligned = time_align_events(noflare_events, sample_start, sample_end)
        flare_aligned = time_align_events(flare_events, sample_start + best_offset, sample_end + best_offset)
        
        print(f"Aligned events - NoFlare: {len(noflare_aligned['x'])}, {flare_type.title()}: {len(flare_aligned['x'])}")
        
        # 空间裁剪重映射（每个样本使用不同的随机裁剪区域）
        print("Applying spatial crop and remap...")
        crop_seed = sample_id * 42  # 每对数据使用不同的裁剪种子
        noflare_cropped = crop_and_remap_events(noflare_aligned, crop_seed=crop_seed)
        flare_cropped = crop_and_remap_events(flare_aligned, crop_seed=crop_seed)
        
        print(f"Final events - NoFlare: {len(noflare_cropped['x'])}, {flare_type.title()}: {len(flare_cropped['x'])}")
        
        # 保存为DSEC格式，文件名包含炫光类型
        target_file = os.path.join(target_dir, f"redandfull_{flare_type}_{int(sample_start*1000)}ms_sample{sample_id}.h5")
        input_file = os.path.join(input_dir, f"redandfull_{flare_type}_{int(sample_start*1000)}ms_sample{sample_id}.h5")
        
        save_dsec_format(noflare_cropped, target_file, f"target_{sample_id}")
        save_dsec_format(flare_cropped, input_file, f"input_{sample_id}")
    
    print(f"\n=== RedAndFull Dataset Generation Complete ===")
    print(f"Generated 20 pairs of redandfull training data")
    print(f"  - Samples 1-10: SixFlare data")
    print(f"  - Samples 11-20: RandomFlare data") 
    print(f"Target files (clean, polygon-filtered + noise): {target_dir}/")  
    print(f"Input files (flare, boundary-filtered + noise): {input_dir}/")
    print(f"Format: DSEC-compatible HDF5")
    print(f"Resolution: 640×480")
    print(f"Duration per sample: 100ms")
    print(f"NoFlare processing: Smooth quadrilateral vertices[(600,320),(780,320),(560,470),(715,490)] + 50px transition + 1M noise events")
    print(f"Flare processing: Smooth remove left(x<335) + right-top(x>910&y<255) + 50px transitions + 1M noise events each")
    print(f"Data sources: SixFlare + RandomFlare (same boundary filtering applied to both)")

if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    random.seed(42)
    np.random.seed(42)
    
    main()