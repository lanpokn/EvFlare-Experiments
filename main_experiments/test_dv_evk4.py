#!/usr/bin/env python3
"""
使用dv库测试读取EVK4 HDF5文件
"""
import dv
import numpy as np
import os

def test_dv_evk4_reading():
    """使用dv库测试EVK4文件读取"""
    
    hdf5_files = [
        "Datasets/EVK4/full_noFlare_part_defocus.hdf5",
        "Datasets/EVK4/full_sixFlare_part_defocus.hdf5"
    ]
    
    for hdf5_path in hdf5_files:
        if not os.path.exists(hdf5_path):
            print(f"File not found: {hdf5_path}")
            continue
            
        print(f"\n=== Testing {hdf5_path} ===")
        
        try:
            # 尝试使用dv库读取
            print("Attempting to read with dv library...")
            
            # 方法1: 尝试直接读取HDF5文件
            reader = dv.io.MonoCameraReader(hdf5_path)
            
            print(f"Successfully opened file with dv!")
            print(f"Camera name: {reader.getCameraName()}")
            print(f"Resolution: {reader.getEventResolution()}")
            
            # 读取事件数据
            events = reader.getNextEventBatch()
            
            if events is not None:
                print(f"Read {len(events)} events")
                print(f"Timestamp range: {events.timestamps[0]} - {events.timestamps[-1]}")
                print(f"X range: {events.x.min()} - {events.x.max()}")  
                print(f"Y range: {events.y.min()} - {events.y.max()}")
                print(f"Polarity values: {np.unique(events.polarity)}")
                
                # 转换为标准格式
                events_dict = {
                    'x': events.x.astype(np.int32),
                    'y': events.y.astype(np.int32),
                    'p': events.polarity.astype(np.int8), 
                    't': events.timestamps.astype(np.float64) / 1e6  # 转换为秒
                }
                
                print("✓ Successfully converted to standard format")
                return events_dict
            else:
                print("No events found in file")
                
        except Exception as e:
            print(f"Failed to read with dv library: {e}")
            
            # 尝试其他dv读取方法
            try:
                print("Trying alternative dv reading method...")
                # 可能需要尝试其他dv的读取函数
                
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
    
    return None

if __name__ == "__main__":
    result = test_dv_evk4_reading()
    if result:
        print("\n✓ DV library successfully read EVK4 data!")
    else:
        print("\n✗ Failed to read EVK4 data with dv library")