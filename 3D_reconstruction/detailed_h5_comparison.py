#!/usr/bin/env python3
"""
详细的H5文件对比分析
对比修复前后的数据，并与原始DVS数据验证
"""

import h5py
import numpy as np
import os

def load_dvs_reference():
    """加载原始DVS数据作为参考"""
    dvs_file = "/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego/events_dvs/lego_train_events.txt"
    
    if not os.path.exists(dvs_file):
        print(f"❌ 原始DVS文件不存在: {dvs_file}")
        return None
        
    print(f"📖 加载原始DVS数据: {dvs_file}")
    try:
        # DVS格式: timestamp_us, x, y, polarity
        data = np.loadtxt(dvs_file, delimiter=' ')
        print(f"  加载成功: {len(data):,} 个事件")
        return {
            't': data[:, 0],  # timestamp_us
            'x': data[:, 1],  # x
            'y': data[:, 2],  # y  
            'p': data[:, 3]   # polarity
        }
    except Exception as e:
        print(f"❌ 加载DVS文件失败: {e}")
        return None

def analyze_h5_file(filepath, label):
    """详细分析H5文件"""
    print(f"\n{'='*60}")
    print(f"分析 {label}")
    print(f"文件: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在")
        return None
        
    try:
        with h5py.File(filepath, 'r') as f:
            events = f['events']
            data = {}
            
            print(f"📊 基本信息:")
            print(f"  文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
            
            for field in ['t', 'x', 'y', 'p']:
                if field in events:
                    field_data = events[field][:]
                    data[field] = field_data
                    
                    print(f"\n{field}字段:")
                    print(f"  数据量: {len(field_data):,}")
                    print(f"  数据类型: {field_data.dtype}")
                    print(f"  范围: [{field_data.min()}, {field_data.max()}]")
                    print(f"  前5个值: {field_data[:5]}")
                    print(f"  后5个值: {field_data[-5:]}")
                    
                    # 数据分布统计
                    if field in ['x', 'y']:
                        unique_count = len(np.unique(field_data))
                        print(f"  唯一值数量: {unique_count}")
                    elif field == 'p':
                        unique_values, counts = np.unique(field_data, return_counts=True)
                        print(f"  极性分布: {dict(zip(unique_values, counts))}")
                        
            return data
            
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None

def compare_datasets(data1, data2, label1, label2):
    """对比两个数据集"""
    print(f"\n{'🔍 数据集对比'}")
    print(f"{label1} vs {label2}")
    print(f"{'='*60}")
    
    if data1 is None or data2 is None:
        print("❌ 无法对比，存在空数据集")
        return
        
    for field in ['t', 'x', 'y', 'p']:
        if field in data1 and field in data2:
            d1, d2 = data1[field], data2[field]
            
            print(f"\n{field}字段对比:")
            print(f"  {label1}: 范围[{d1.min()}, {d1.max()}], 数量{len(d1):,}")
            print(f"  {label2}: 范围[{d2.min()}, {d2.max()}], 数量{len(d2):,}")
            
            if len(d1) == len(d2):
                # 检查数据是否完全相同
                if np.array_equal(d1, d2):
                    print(f"  ✅ 数据完全相同")
                else:
                    diff_count = np.sum(d1 != d2)
                    print(f"  ❌ 数据不同: {diff_count:,} 个位置不同")
                    
                    # 显示前几个不同的位置
                    diff_indices = np.where(d1 != d2)[0][:5]
                    for idx in diff_indices:
                        print(f"    位置{idx}: {d1[idx]} vs {d2[idx]}")
            else:
                print(f"  ❌ 数据长度不同")

def main():
    """主函数"""
    print(f"🔍 H5文件修复前后详细对比分析")
    
    # 文件路径
    current_file = "/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego/events_h5/lego_sequence.h5"
    backup_file = "/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego/events_h5/lego_sequence_backup_wrong.h5"
    
    # 加载原始DVS数据作为基准
    dvs_data = load_dvs_reference()
    
    # 分析修复后的文件
    current_data = analyze_h5_file(current_file, "修复后的H5文件")
    
    # 分析修复前的文件
    backup_data = analyze_h5_file(backup_file, "修复前的H5文件（错误版本）")
    
    # 对比修复前后
    if current_data and backup_data:
        compare_datasets(current_data, backup_data, "修复后", "修复前")
    
    # 验证修复后的数据与原始DVS数据是否匹配
    if current_data and dvs_data:
        print(f"\n🎯 修复后H5数据 vs 原始DVS数据验证")
        print(f"{'='*60}")
        
        for field in ['t', 'x', 'y', 'p']:
            h5_data = current_data[field]
            dvs_ref = dvs_data[field]
            
            print(f"\n{field}字段验证:")
            print(f"  H5数据: 范围[{h5_data.min()}, {h5_data.max()}], 数量{len(h5_data):,}")
            print(f"  DVS原始: 范围[{dvs_ref.min()}, {dvs_ref.max()}], 数量{len(dvs_ref):,}")
            
            if len(h5_data) == len(dvs_ref):
                if np.allclose(h5_data, dvs_ref):
                    print(f"  ✅ 与原始DVS数据完全匹配")
                else:
                    diff_count = np.sum(~np.isclose(h5_data, dvs_ref))
                    print(f"  ❌ 与原始数据不匹配: {diff_count:,} 个差异")
            else:
                print(f"  ❌ 数据长度不匹配")
    
    print(f"\n{'📋 验证总结'}")
    print(f"{'='*60}")
    if current_data:
        # 验证数据范围
        ranges_ok = (
            225 <= current_data['t'].min() <= current_data['t'].max() <= 199000 and
            20 <= current_data['x'].min() <= current_data['x'].max() <= 639 and
            0 <= current_data['y'].min() <= current_data['y'].max() <= 479 and
            set(np.unique(current_data['p'])) == {0.0, 1.0}
        )
        
        print(f"修复后H5文件数据范围验证: {'✅ 通过' if ranges_ok else '❌ 失败'}")
        print(f"数据量: {len(current_data['t']):,} 个事件")
        print(f"时间范围: {current_data['t'].min():.0f} - {current_data['t'].max():.0f} μs")
        print(f"空间范围: X[{current_data['x'].min():.0f}, {current_data['x'].max():.0f}], Y[{current_data['y'].min():.0f}, {current_data['y'].max():.0f}]")
        
        # 极性分布
        unique_p, counts_p = np.unique(current_data['p'], return_counts=True)
        p_dist = dict(zip(unique_p, counts_p))
        print(f"极性分布: ON={p_dist.get(1.0, 0):,}, OFF={p_dist.get(0.0, 0):,}")

if __name__ == "__main__":
    main()