#!/usr/bin/env python3
"""
终极WindowsPath修复方案 - 使用monkeypatch
"""

import os
import shutil

def apply_ultimate_patch():
    """应用终极修复补丁"""
    
    eval_path = '/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/eval.py'
    backup_path = eval_path + '.backup'
    
    # 备份
    if not os.path.exists(backup_path):
        shutil.copy2(eval_path, backup_path)
        print("已备份原始文件")
    
    # 读取文件
    with open(eval_path, 'r') as f:
        content = f.read()
    
    # 添加monkeypatch修复
    fix_code = '''
import pathlib
from pathlib import PosixPath

# Monkeypatch WindowsPath to PosixPath
original_new = pathlib.Path.__new__

def patched_new(cls, *args, **kwargs):
    if cls.__name__ == 'WindowsPath':
        return PosixPath(*args, **kwargs)
    return original_new(cls, *args, **kwargs)

pathlib.Path.__new__ = staticmethod(patched_new)
pathlib.WindowsPath = PosixPath

'''
    
    # 在第一个import之前添加修复代码
    first_import = content.find('import argparse')
    if first_import != -1:
        content = content[:first_import] + fix_code + content[first_import:]
    
    # 写回文件
    with open(eval_path, 'w') as f:
        f.write(content)
    
    print("终极补丁应用成功！")

if __name__ == "__main__":
    apply_ultimate_patch()