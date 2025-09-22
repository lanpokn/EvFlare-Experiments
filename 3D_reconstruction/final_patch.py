#!/usr/bin/env python3
"""
最终的WindowsPath修复方案
"""

import os
import shutil

def apply_final_patch():
    """应用最终修复补丁"""
    
    eval_path = '/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/eval.py'
    backup_path = eval_path + '.backup'
    
    # 备份
    if not os.path.exists(backup_path):
        shutil.copy2(eval_path, backup_path)
        print("已备份原始文件")
    
    # 读取文件
    with open(eval_path, 'r') as f:
        content = f.read()
    
    # 添加修复函数
    fix_function = '''
import pickle
import pathlib
from pathlib import PosixPath

class CompatibleUnpickler(pickle.Unpickler):
    """兼容Windows和Linux路径的unpickler"""
    
    def find_class(self, module, name):
        # 重定向WindowsPath到PosixPath
        if module == 'pathlib':
            if name == 'WindowsPath':
                return PosixPath
            elif name == 'PosixPath':
                return PosixPath
        return super().find_class(module, name)

def load_checkpoint_safely(checkpoint_path, device):
    """安全加载checkpoint，处理路径兼容性问题"""
    try:
        # 首先尝试标准加载
        return torch.load(checkpoint_path, device)
    except (NotImplementedError, OSError) as e:
        if 'WindowsPath' in str(e) or 'cannot instantiate' in str(e):
            print(f"检测到路径兼容性问题，使用兼容模式加载...")
            # 使用自定义unpickler
            with open(checkpoint_path, 'rb') as f:
                unpickler = CompatibleUnpickler(f)
                checkpoint = unpickler.load()
            return checkpoint
        else:
            raise e

'''
    
    # 在import后添加修复函数
    import_section = content.find('import model as model_arch')
    if import_section != -1:
        content = content[:import_section] + fix_function + content[import_section:]
    
    # 替换torch.load调用
    old_load = '    checkpoint = torch.load(checkpoint_path, device)'
    new_load = '    checkpoint = load_checkpoint_safely(checkpoint_path, device)'
    
    content = content.replace(old_load, new_load)
    
    # 写回文件
    with open(eval_path, 'w') as f:
        f.write(content)
    
    print("最终补丁应用成功！")

if __name__ == "__main__":
    apply_final_patch()