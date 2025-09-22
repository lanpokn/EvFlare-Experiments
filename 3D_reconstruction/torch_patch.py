#!/usr/bin/env python3
"""
使用torch.load的map_location修复WindowsPath问题
"""

import os
import shutil

def apply_torch_patch():
    """应用基于torch.load的补丁"""
    
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
from pathlib import PosixPath

def fix_path_in_obj(obj):
    """递归修复对象中的WindowsPath"""
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if 'WindowsPath' in str(type(attr_value)) or 'PosixPath' in str(type(attr_value)):
                # 转换为字符串，避免路径对象
                setattr(obj, attr_name, str(attr_value))
    return obj

'''
    
    # 在import后添加修复函数
    import_section = content.find('import model as model_arch')
    if import_section != -1:
        content = content[:import_section] + fix_function + content[import_section:]
    
    # 替换torch.load调用
    old_load = 'checkpoint = torch.load(checkpoint_path, device)'
    new_load = '''try:
        checkpoint = torch.load(checkpoint_path, device)
    except NotImplementedError as e:
        if "cannot instantiate 'WindowsPath'" in str(e):
            # 使用cpu加载避免设备问题，然后修复路径
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'config' in checkpoint:
                fix_path_in_obj(checkpoint['config'])
        else:
            raise e'''
    
    content = content.replace(old_load, new_load)
    
    # 写回文件
    with open(eval_path, 'w') as f:
        f.write(content)
    
    print("Torch补丁应用成功！")

if __name__ == "__main__":
    apply_torch_patch()