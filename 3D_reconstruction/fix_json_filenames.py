#!/usr/bin/env python3
"""
修复transforms JSON文件中的文件名格式问题
将00001格式改为0001格式以匹配实际文件名
"""

import json
import os
from pathlib import Path

def fix_json_file(json_path):
    """修复单个JSON文件"""
    print(f"修复文件: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 修复所有frame的file_path
    for frame in data['frames']:
        old_path = frame['file_path']
        
        # 如果是train/00001格式，改为train/0001
        if '/00' in old_path and len(old_path.split('/')[-1]) == 5:
            # 提取目录和文件名
            parts = old_path.split('/')
            directory = parts[0]  # train 或 test
            filename = parts[1]   # 00001
            
            # 去掉前导0，保持4位数格式
            new_filename = filename[1:]  # 0001
            new_path = f"{directory}/{new_filename}"
            
            print(f"  {old_path} -> {new_path}")
            frame['file_path'] = new_path
    
    # 保存修复后的文件
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ {json_path} 修复完成")

def main():
    """修复所有transforms JSON文件"""
    lego_dir = Path("datasets/lego")
    
    json_files = [
        lego_dir / "transforms_train.json",
        lego_dir / "transforms_test.json"
    ]
    
    for json_file in json_files:
        if json_file.exists():
            fix_json_file(json_file)
        else:
            print(f"⚠️  文件不存在: {json_file}")
    
    print("\n🎉 所有JSON文件修复完成！")
    print("现在可以正常运行3DGS训练了")

if __name__ == "__main__":
    main()