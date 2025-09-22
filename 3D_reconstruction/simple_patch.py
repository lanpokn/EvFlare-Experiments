#!/usr/bin/env python3
"""
简单的EVREAL WindowsPath修复补丁
"""

import os
import shutil

def apply_simple_patch():
    """应用简单补丁到eval.py"""
    
    eval_path = '/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/eval.py'
    backup_path = eval_path + '.backup'
    
    # 备份
    if not os.path.exists(backup_path):
        shutil.copy2(eval_path, backup_path)
        print("已备份原始文件")
    
    # 读取文件
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    
    # 找到torch.load那一行
    for i, line in enumerate(lines):
        if 'checkpoint = torch.load(checkpoint_path, device)' in line:
            # 在前面插入import和函数定义
            import_lines = [
                'import pickle\n',
                'from pathlib import PosixPath\n',
                '\n',
                'class WindowsPathFix(pickle.Unpickler):\n',
                '    def find_class(self, module, name):\n',
                '        if module == "pathlib" and name == "WindowsPath":\n',
                '            return PosixPath\n',
                '        return super().find_class(module, name)\n',
                '\n'
            ]
            
            # 替换torch.load调用
            new_load_lines = [
                '    # 使用修复的加载器避免WindowsPath错误\n',
                '    with open(checkpoint_path, "rb") as f:\n',
                '        checkpoint = WindowsPathFix(f).load()\n'
            ]
            
            # 插入修复代码
            lines = lines[:10] + import_lines + lines[10:]  # 在import后插入
            
            # 找到新的torch.load位置并替换
            for j, line in enumerate(lines):
                if 'checkpoint = torch.load(checkpoint_path, device)' in line:
                    lines = lines[:j] + new_load_lines + lines[j+1:]
                    break
            break
    
    # 写回文件
    with open(eval_path, 'w') as f:
        f.writelines(lines)
    
    print("补丁应用成功！")

if __name__ == "__main__":
    apply_simple_patch()