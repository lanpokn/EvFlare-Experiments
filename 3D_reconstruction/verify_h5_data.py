#!/usr/bin/env python3
"""
验证H5文件数据是否正确
检查修复前后的数据范围和前10个值
"""

import h5py
import numpy as np
import os

def verify_h5_file(filepath, label):
    """验证H5文件数据范围和显示前10个值"""
    print(f"\n{'='*60}")
    print(f"验证 {label}")
    print(f"文件: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
            print(f"HDF5结构:")
            
            def print_structure(name, obj):
                print(f"  {name}: {type(obj).__name__}")
                if isinstance(obj, h5py.Dataset):
                    print(f"    shape: {obj.shape}, dtype: {obj.dtype}")
            
            f.visititems(print_structure)
            
            # 检查events组
            if 'events' in f:
                events = f['events']
                print(f"\n📊 Events数据分析:")
                
                # 检查每个字段
                for field in ['t', 'x', 'y', 'p']:
                    if field in events:
                        data = events[field][:]
                        print(f"\n{field}字段:")
                        print(f"  数据量: {len(data):,}")
                        print(f"  数据类型: {data.dtype}")
                        print(f"  范围: [{data.min()}, {data.max()}]")
                        print(f"  前10个值: {data[:10]}")
                        
                        # 验证范围
                        if field == 't':
                            expected_range = (225, 199000)
                            in_range = (data.min() >= expected_range[0] and 
                                      data.max() <= expected_range[1])
                            status = "✅" if in_range else "❌"
                            print(f"  预期范围 {expected_range}: {status}")
                            
                        elif field == 'x':
                            expected_range = (20, 639)
                            in_range = (data.min() >= expected_range[0] and 
                                      data.max() <= expected_range[1])
                            status = "✅" if in_range else "❌"
                            print(f"  预期范围 {expected_range}: {status}")
                            
                        elif field == 'y':
                            expected_range = (0, 479)
                            in_range = (data.min() >= expected_range[0] and 
                                      data.max() <= expected_range[1])
                            status = "✅" if in_range else "❌"
                            print(f"  预期范围 {expected_range}: {status}")
                            
                        elif field == 'p':
                            unique_values = np.unique(data)
                            expected_values = [0, 1]
                            is_valid = all(v in expected_values for v in unique_values)
                            status = "✅" if is_valid else "❌"
                            print(f"  预期值 {expected_values}: {status}")
                            print(f"  实际唯一值: {unique_values}")
                    else:
                        print(f"\n❌ 缺失字段: {field}")
            else:
                print(f"\n❌ 缺失'events'组")
                
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")

def compare_files():
    """对比修复前后的H5文件"""
    print(f"\n{'🔍 H5文件数据验证报告'}")
    
    # 验证修复后的文件
    current_file = "/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego/events_h5/lego_sequence.h5"
    verify_h5_file(current_file, "修复后的H5文件")
    
    # 查找backup文件
    backup_file = "/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego/events_h5/lego_sequence_backup.h5"
    if os.path.exists(backup_file):
        verify_h5_file(backup_file, "修复前的H5文件 (备份)")
    else:
        print(f"\n📝 备份文件不存在: {backup_file}")
        print("这可能意味着之前没有创建备份，或备份在其他位置")

if __name__ == "__main__":
    compare_files()