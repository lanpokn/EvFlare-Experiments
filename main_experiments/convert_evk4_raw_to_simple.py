#!/usr/bin/env python3
"""
尝试从RAW文件读取EVK4数据并转换为简单格式
先忽略复杂的HDF5 ECF格式，从RAW文件入手
"""
import numpy as np
import struct
import h5py
import os

def read_prophesee_raw_events(filename, max_events=None):
    """
    尝试读取Prophesee RAW格式文件
    """
    print(f"Reading RAW file: {filename}")
    print(f"File size: {os.path.getsize(filename)} bytes")
    
    events = []
    
    with open(filename, 'rb') as f:
        # 尝试跳过文件头
        header_size = 0
        data = f.read()
        
        print(f"First 100 bytes (as hex): {data[:100].hex()}")
        print(f"First 100 bytes (as ascii, ignore errors): {data[:100].decode('ascii', errors='ignore')}")
        
        # 查找可能的事件数据模式
        # Prophesee事件通常是连续的二进制数据
        # 每个事件可能是12-16字节，包含 x, y, p, t
        
        event_size = 16  # 假设每个事件16字节，基于HDF5中看到的itemsize
        num_events = (len(data) - header_size) // event_size
        
        print(f"Estimated events: {num_events}")
        
        if max_events:
            num_events = min(num_events, max_events)
        
        # 尝试解析为结构化数据
        try:
            # 跳过可能的文件头
            events_data = data[header_size:header_size + num_events * event_size]
            
            # 基于HDF5格式定义数据类型
            dt = np.dtype([('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')])
            events_array = np.frombuffer(events_data, dtype=dt)
            
            print(f"Successfully parsed {len(events_array)} events")
            print(f"X range: {events_array['x'].min()} - {events_array['x'].max()}")
            print(f"Y range: {events_array['y'].min()} - {events_array['y'].max()}")
            print(f"P values: {np.unique(events_array['p'])}")
            print(f"T range: {events_array['t'].min()} - {events_array['t'].max()}")
            
            return {
                'x': events_array['x'].astype(np.int32),
                'y': events_array['y'].astype(np.int32),
                'p': events_array['p'].astype(np.int8),
                't': events_array['t'].astype(np.float64) / 1e6  # 转换为秒
            }
            
        except Exception as e:
            print(f"Failed to parse as structured data: {e}")
            return None

def save_as_simple_h5(events_dict, output_path):
    """
    将事件数据保存为简单的H5格式，无压缩
    """
    with h5py.File(output_path, 'w') as f:
        events_group = f.create_group('events')
        events_group.create_dataset('x', data=events_dict['x'], compression=None)
        events_group.create_dataset('y', data=events_dict['y'], compression=None)
        events_group.create_dataset('p', data=events_dict['p'], compression=None)
        events_group.create_dataset('t', data=events_dict['t'], compression=None)
        
    print(f"Saved simplified H5 file: {output_path}")

def main():
    # 尝试转换RAW文件
    raw_files = [
        "Datasets/EVK4/full_noflare_defocus_part.raw",
        "Datasets/EVK4/full_sixflare_defocus_part.raw"
    ]
    
    for raw_file in raw_files:
        if os.path.exists(raw_file):
            print(f"\n=== Processing {raw_file} ===")
            events = read_prophesee_raw_events(raw_file, max_events=100000)  # 限制事件数量先测试
            
            if events:
                # 保存为简单H5格式
                output_name = os.path.basename(raw_file).replace('.raw', '_simple.h5')
                output_path = f"Datasets/EVK4/{output_name}"
                save_as_simple_h5(events, output_path)
            else:
                print("Failed to extract events from RAW file")
        else:
            print(f"File not found: {raw_file}")

if __name__ == "__main__":
    main()