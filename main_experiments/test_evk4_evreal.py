#!/usr/bin/env python3
"""
在evreal-tools环境中测试读取EVK4 HDF5文件
"""
import os
# 设置多个可能的插件路径
os.environ['HDF5_PLUGIN_PATH'] = '/home/lanpoknlanpokn/miniconda3/envs/Umain/lib/hdf5/plugin:/usr/local/hdf5/lib/plugin'

import h5py
import numpy as np

def test_evk4_reading_evreal():
    """使用标准h5py方式在evreal-tools环境中测试EVK4读取"""
    
    hdf5_path = "Datasets/EVK4/full_noFlare_part_defocus.hdf5"
    
    print(f"Testing EVK4 reading in evreal-tools environment")
    print(f"File: {hdf5_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            grp = f.get('CD', f)  # EVK4 HDF5 存储在 group 'CD'
            ev = grp['events']
            
            print(f"Events shape: {ev.shape}")
            print(f"Events dtype: {ev.dtype}")
            
            # 尝试按用户提供的方式读取
            print("Attempting to read events...")
            x = ev['x'][:].astype(np.int32)
            y = ev['y'][:].astype(np.int32)
            p = ev['p'][:].astype(np.int8)
            ts = ev['t'][:].astype(np.float64)
            
            print(f"SUCCESS! Read {len(x)} events")
            print(f"X range: {x.min()} - {x.max()}")
            print(f"Y range: {y.min()} - {y.max()}")
            print(f"P values: {np.unique(p)}")
            print(f"T range: {ts.min()} - {ts.max()}")
            print(f"Duration: {(ts.max() - ts.min())/1e6:.3f} seconds")
            
            return {
                'x': x,
                'y': y,
                'p': p,
                't': ts
            }
            
    except Exception as e:
        print(f"FAILED: {e}")
        return None

if __name__ == "__main__":
    result = test_evk4_reading_evreal()
    if result:
        print("\n✓ EVK4 reading successful in evreal-tools environment!")
    else:
        print("\n✗ EVK4 reading failed in evreal-tools environment")