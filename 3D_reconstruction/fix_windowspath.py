#!/usr/bin/env python3
"""
修复 EVREAL 中的 WindowsPath 兼容性问题

根本原因：
失败的方法 (E2VID+, FireNet+, ET-Net, HyperE2VID) 在模型checkpoint中
保存了带有Windows路径的ConfigParser对象，这些对象在WSL/Linux上
无法正确反序列化。

解决方案：
方案1：修改torch.load时的unpickle行为，将WindowsPath转换为PosixPath
方案2：预处理模型文件，移除有问题的Path对象
"""

import sys
import os
import torch
import pickle
import pathlib
from pathlib import PosixPath
import shutil

def create_custom_unpickler():
    """创建自定义unpickler，处理WindowsPath问题"""
    
    class PathUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # 将WindowsPath重定向到PosixPath
            if module == 'pathlib' and name == 'WindowsPath':
                return PosixPath
            return super().find_class(module, name)
    
    return PathUnpickler

def fix_model_checkpoint(model_path, output_path=None):
    """修复单个模型文件"""
    
    if output_path is None:
        output_path = model_path + '.fixed'
        
    print(f"修复模型文件: {model_path}")
    
    try:
        # 使用自定义unpickler加载
        with open(model_path, 'rb') as f:
            unpickler = create_custom_unpickler()(f)
            checkpoint = unpickler.load()
            
        print("成功加载checkpoint")
        
        # 检查和修复config中的Path对象
        if 'config' in checkpoint:
            config = checkpoint['config']
            if hasattr(config, '_save_dir') and isinstance(config._save_dir, pathlib.Path):
                # 转换为字符串，避免保存Path对象
                config._save_dir = str(config._save_dir)
                print(f"转换 _save_dir: {config._save_dir}")
                
            if hasattr(config, '_log_dir') and isinstance(config._log_dir, pathlib.Path):
                config._log_dir = str(config._log_dir)
                print(f"转换 _log_dir: {config._log_dir}")
                
            if hasattr(config, 'resume') and isinstance(config.resume, pathlib.Path):
                config.resume = str(config.resume)
                print(f"转换 resume: {config.resume}")
        
        # 保存修复后的checkpoint
        torch.save(checkpoint, output_path)
        print(f"已保存修复后的模型: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"修复失败: {e}")
        return False

def patch_eval_py():
    """为eval.py创建临时补丁"""
    
    eval_path = '/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/eval.py'
    backup_path = eval_path + '.backup'
    
    # 备份原文件
    if not os.path.exists(backup_path):
        shutil.copy2(eval_path, backup_path)
        print(f"已备份 eval.py -> {backup_path}")
    
    # 读取原文件
    with open(eval_path, 'r') as f:
        content = f.read()
    
    # 创建补丁
    patch_code = '''
import pathlib
from pathlib import PosixPath

class PathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'WindowsPath':
            return PosixPath
        return super().find_class(module, name)

def load_checkpoint_with_path_fix(checkpoint_path, device):
    """使用自定义unpickler加载checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        unpickler = PathUnpickler(f)
        checkpoint = unpickler.load()
    return checkpoint
'''
    
    # 替换torch.load调用
    old_line = "    checkpoint = torch.load(checkpoint_path, device)"
    new_line = "    checkpoint = load_checkpoint_with_path_fix(checkpoint_path, device)"
    
    if old_line in content:
        # 添加import
        if 'import pickle' not in content:
            content = content.replace('import torch', 'import torch\nimport pickle')
        
        # 添加自定义函数
        content = content.replace('import torch', f'import torch{patch_code}')
        
        # 替换函数调用
        content = content.replace(old_line, new_line)
        
        # 写回文件
        with open(eval_path, 'w') as f:
            f.write(content)
            
        print("已应用eval.py补丁")
        return True
    else:
        print("未找到需要替换的代码行")
        return False

def restore_eval_py():
    """恢复原始eval.py"""
    
    eval_path = '/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/eval.py'
    backup_path = eval_path + '.backup'
    
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, eval_path)
        print("已恢复原始 eval.py")
        return True
    else:
        print("未找到备份文件")
        return False

def fix_all_failed_models():
    """修复所有失败的模型"""
    
    failed_methods = ['E2VID+', 'FireNet+', 'ET-Net', 'HyperE2VID']
    base_path = '/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/pretrained'
    
    for method in failed_methods:
        model_path = os.path.join(base_path, method, 'model.pth')
        fixed_path = os.path.join(base_path, method, 'model_fixed.pth')
        
        if os.path.exists(model_path):
            success = fix_model_checkpoint(model_path, fixed_path)
            if success:
                print(f"✅ {method} 修复成功")
            else:
                print(f"❌ {method} 修复失败")
        else:
            print(f"⚠️  {method} 模型文件不存在")

def main():
    """主函数"""
    
    print("EVREAL WindowsPath 兼容性修复工具")
    print("="*50)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['patch', 'fix-models', 'restore'], 
                       default='patch', help='修复方法')
    
    args = parser.parse_args()
    
    if args.method == 'patch':
        print("应用 eval.py 补丁...")
        patch_eval_py()
        
    elif args.method == 'fix-models':
        print("修复模型文件...")
        fix_all_failed_models()
        
    elif args.method == 'restore':
        print("恢复原始文件...")
        restore_eval_py()

if __name__ == "__main__":
    main()