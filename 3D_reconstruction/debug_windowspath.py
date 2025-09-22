#!/usr/bin/env python3
"""
Debug script to understand and fix the WindowsPath error in EVREAL
"""

import sys
import os
import torch
import platform

# Add EVREAL path
sys.path.append('/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main')

def debug_model_loading():
    """Debug model loading issues"""
    
    # 失败的方法
    failed_methods = ['E2VID+', 'FireNet+', 'ET-Net', 'HyperE2VID']
    # 成功的方法
    success_methods = ['E2VID', 'FireNet', 'SPADE-E2VID', 'SSL-E2VID']
    
    print(f"Current platform: {platform.system()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)
    
    print("=== 分析失败原因 ===")
    
    for method in failed_methods:
        model_path = f'/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/pretrained/{method}/model.pth'
        print(f"\n--- {method} ---")
        
        try:
            # 加载checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['config']
            
            print(f"Config type: {type(config)}")
            
            # 检查Path对象
            path_attrs = []
            for attr_name in dir(config):
                if attr_name.startswith('_'):
                    attr_value = getattr(config, attr_name)
                    if 'Path' in str(type(attr_value)):
                        path_attrs.append(f"{attr_name}: {type(attr_value)}")
            
            if path_attrs:
                print(f"Path attributes found: {path_attrs}")
                
                # 尝试调用init_obj - 这里会出错
                try:
                    print("Attempting to call config.init_obj...")
                    # 这行代码会失败，因为ConfigParser在pickle时保存了Path对象
                    # 但在WSL上无法实例化WindowsPath
                    model = config.init_obj('arch', None)  # 简化测试
                except Exception as e:
                    print(f"ERROR in init_obj: {e}")
                    print(f"Error type: {type(e)}")
            
        except Exception as e:
            print(f"ERROR loading checkpoint: {e}")
    
    print("\n=== 分析成功方法 ===")
    
    for method in success_methods:
        model_path = f'/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/pretrained/{method}/model.pth'
        print(f"\n--- {method} ---")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"Config type: {type(config)}")
                
                if hasattr(config, 'init_obj'):
                    print("Has init_obj method - ConfigParser")
                else:
                    print("Regular dict config - no Path objects")
            else:
                print("No config in checkpoint")
                
        except Exception as e:
            print(f"ERROR: {e}")

def propose_solution():
    """提出解决方案"""
    
    print("\n" + "=" * 60)
    print("=== 根本原因分析 ===")
    print("""
1. 失败的方法(E2VID+, FireNet+, ET-Net, HyperE2VID)：
   - 使用 ConfigParser 对象存储配置
   - ConfigParser 对象包含 pathlib.PosixPath 属性 (_save_dir, _log_dir, resume)
   - 这些 Path 对象在 Windows 上训练时被序列化进 .pth 文件
   - 在 WSL/Linux 上加载时，无法反序列化 WindowsPath 对象

2. 成功的方法(E2VID, FireNet, SPADE-E2VID, SSL-E2VID)：
   - E2VID: 没有 config 字段
   - FireNet: config 是普通 dict
   - SPADE-E2VID/SSL-E2VID: 直接存储 state_dict，没有 ConfigParser
    """)
    
    print("\n=== 解决方案建议 ===")
    print("""
解决方案1: 修改 torch.load 调用 (推荐)
- 在 eval.py:132 添加自定义 pickle 模块
- 替换 WindowsPath 为 PosixPath

解决方案2: 预处理模型文件
- 重新保存有问题的模型文件
- 移除或转换 Path 对象

解决方案3: 修改 ConfigParser 类
- 添加 __setstate__ 方法处理反序列化
- 自动转换 Path 对象
    """)

if __name__ == "__main__":
    debug_model_loading()
    propose_solution()