#!/usr/bin/env python3
"""
创建一个小的测试数据集来快速验证灰度图功能
"""

import json
import shutil
from pathlib import Path

def create_small_dataset():
    """创建包含前10张图像的小数据集"""
    source_dir = Path("datasets/lego")
    target_dir = Path("datasets/lego_small")
    
    # 创建目标目录
    target_dir.mkdir(exist_ok=True)
    (target_dir / "train").mkdir(exist_ok=True)
    (target_dir / "test").mkdir(exist_ok=True)
    
    # 复制前10张训练图像
    print("复制前10张训练图像...")
    for i in range(1, 11):
        src_file = source_dir / "train" / f"{i:04d}.png"
        dst_file = target_dir / "train" / f"{i:04d}.png"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"  复制: {src_file.name}")
    
    # 复制前10张测试图像
    print("复制前10张测试图像...")
    for i in range(1, 11):
        src_file = source_dir / "test" / f"{i:04d}.png"
        dst_file = target_dir / "test" / f"{i:04d}.png"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"  复制: {src_file.name}")
    
    # 创建精简版的transforms文件
    print("创建精简版transforms文件...")
    
    # 读取原始训练transforms
    with open(source_dir / "transforms_train.json") as f:
        train_data = json.load(f)
    
    # 只保留前10个frames
    train_data["frames"] = train_data["frames"][:10]
    
    # 保存精简版训练transforms
    with open(target_dir / "transforms_train.json", 'w') as f:
        json.dump(train_data, f, indent=4)
    
    # 读取原始测试transforms
    with open(source_dir / "transforms_test.json") as f:
        test_data = json.load(f)
    
    # 只保留前10个frames
    test_data["frames"] = test_data["frames"][:10]
    
    # 保存精简版测试transforms
    with open(target_dir / "transforms_test.json", 'w') as f:
        json.dump(test_data, f, indent=4)
    
    # 复制点云文件
    shutil.copy2(source_dir / "points3d.ply", target_dir / "points3d.ply")
    
    print(f"\n✅ 小数据集创建完成: {target_dir}")
    print("包含:")
    print("  - 10张训练图像")
    print("  - 10张测试图像") 
    print("  - 对应的transforms文件")
    print("  - 点云文件")
    
    return target_dir

if __name__ == "__main__":
    dataset_path = create_small_dataset()
    print(f"\n🚀 现在可以用小数据集测试:")
    print(f"python train.py -s ../{dataset_path} -m output/lego_small_test --iterations 1000 --grayscale")