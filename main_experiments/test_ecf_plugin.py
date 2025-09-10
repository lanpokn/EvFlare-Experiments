#!/usr/bin/env python3
"""
测试ECF插件是否正常工作
"""
import os
import sys

# 在导入h5py之前设置环境变量
os.environ['HDF5_PLUGIN_PATH'] = '/home/lanpoknlanpokn/miniconda3/envs/Umain/lib/hdf5/plugin'

print(f"HDF5_PLUGIN_PATH: {os.environ.get('HDF5_PLUGIN_PATH', 'Not set')}")

try:
    import h5py
    print(f"h5py version: {h5py.__version__}")
    
    # 尝试读取EVK4文件
    print("Attempting to open EVK4 file...")
    hdf5_path = "Datasets/EVK4/full_noFlare_part_defocus.hdf5"
    
    with h5py.File(hdf5_path, 'r') as f:
        print("File opened successfully!")
        print(f"Root keys: {list(f.keys())}")
        
        grp = f.get('CD', f)
        print(f"CD group keys: {list(grp.keys())}")
        
        ev = grp['events']
        print(f"Events shape: {ev.shape}")
        print(f"Events dtype: {ev.dtype}")
        
        # 尝试读取小样本
        print("Attempting to read small sample (10 events)...")
        try:
            sample = ev[:10]
            print("SUCCESS: Read 10 events!")
            print("Sample data:")
            print(f"  X: {sample['x']}")
            print(f"  Y: {sample['y']}")
            print(f"  P: {sample['p']}")
            print(f"  T: {sample['t']}")
            
        except Exception as e:
            print(f"FAILED to read events: {e}")
            
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("Test completed!")