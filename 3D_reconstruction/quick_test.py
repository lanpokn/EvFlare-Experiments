#!/usr/bin/env python3
"""
快速测试WindowsPath修复是否有效
"""

import sys
sys.path.append('/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main')

import torch

# 在eval.py中应用的monkeypatch
import pathlib
from pathlib import PosixPath

original_new = pathlib.Path.__new__

def patched_new(cls, *args, **kwargs):
    if cls.__name__ == 'WindowsPath':
        return PosixPath(*args, **kwargs)
    return original_new(cls, *args, **kwargs)

pathlib.Path.__new__ = staticmethod(patched_new)
pathlib.WindowsPath = PosixPath

# 测试模型加载
failed_methods = ['E2VID+', 'FireNet+', 'ET-Net', 'HyperE2VID']

print("测试WindowsPath修复效果")
print("="*40)

for method in failed_methods:
    model_path = f'/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/pretrained/{method}/model.pth'
    print(f"\n测试 {method}...")
    
    try:
        # 尝试加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ {method}: 成功加载checkpoint")
        
        # 检查config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"   Config类型: {type(config)}")
            
            # 检查是否还有路径对象
            path_attrs = []
            for attr_name in dir(config):
                if attr_name.startswith('_'):
                    attr_value = getattr(config, attr_name)
                    if 'Path' in str(type(attr_value)):
                        path_attrs.append(f"{attr_name}:{type(attr_value)}")
            
            if path_attrs:
                print(f"   路径属性: {path_attrs}")
            else:
                print(f"   ✅ 无路径对象问题")
                
    except Exception as e:
        print(f"❌ {method}: 加载失败 - {e}")

print(f"\n测试完成！")
print(f"如果所有方法都显示'成功加载checkpoint'，说明WindowsPath问题已解决。")