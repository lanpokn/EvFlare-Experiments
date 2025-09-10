#!/usr/bin/env python3
"""
尝试通过Python访问EVK4数据，绕过ECF解码
"""
import h5py
import numpy as np
import os

def examine_evk4_structure():
    """检查EVK4文件的内部结构"""
    hdf5_path = "Datasets/EVK4/full_noFlare_part_defocus.hdf5"
    
    with h5py.File(hdf5_path, 'r') as f:
        print("=== File Structure ===")
        print(f"Root keys: {list(f.keys())}")
        
        cd_group = f['CD']
        print(f"CD group keys: {list(cd_group.keys())}")
        
        events = cd_group['events']
        print(f"Events dataset:")
        print(f"  Shape: {events.shape}")
        print(f"  Dtype: {events.dtype}")
        print(f"  Chunks: {events.chunks}")
        print(f"  Compression: {events.compression}")
        print(f"  Compression opts: {events.compression_opts}")
        print(f"  Filters: {events.get_create_plist().get_nfilters()} filters")
        
        # 检查过滤器信息
        plist = events.get_create_plist()
        for i in range(plist.get_nfilters()):
            filter_info = plist.get_filter(i)
            print(f"    Filter {i}: {filter_info}")
        
        # 尝试读取原始压缩数据（如果可能）
        try:
            print("Attempting to access raw data...")
            # 这可能不工作，但值得一试
            raw_data = events[()]
            print(f"Success! Raw data shape: {raw_data.shape}")
        except Exception as e:
            print(f"Raw data access failed: {e}")
            
        # 检查indexes
        if 'indexes' in cd_group:
            indexes = cd_group['indexes']
            print(f"Indexes dataset:")
            print(f"  Shape: {indexes.shape}")
            print(f"  Dtype: {indexes.dtype}")

if __name__ == "__main__":
    examine_evk4_structure()