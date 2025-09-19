#!/usr/bin/env python3
"""
数据集合并脚本
将 xx_flare 和 xx_normal 数据集合并为完整的 xx 数据集
训练集使用 flare 版本，测试集使用 normal 版本
"""

import os
import json
import shutil
from pathlib import Path

def merge_datasets(base_dir="datasets"):
    """合并数据集"""
    base_path = Path(base_dir)
    
    # 找到所有 _flare 和 _normal 数据集
    flare_datasets = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith('_flare')]
    
    for flare_dir in flare_datasets:
        dataset_name = flare_dir.name[:-6]  # 移除 '_flare' 后缀
        normal_dir = base_path / f"{dataset_name}_normal"
        merged_dir = base_path / dataset_name
        
        if not normal_dir.exists():
            print(f"警告: 找不到对应的 normal 数据集: {normal_dir}")
            continue
            
        print(f"处理数据集: {dataset_name}")
        
        # 创建合并后的目录结构
        merged_dir.mkdir(exist_ok=True)
        (merged_dir / "train").mkdir(exist_ok=True)
        (merged_dir / "test").mkdir(exist_ok=True)
        
        # 复制训练集图片 (from flare)
        train_src = flare_dir / "train"
        train_dst = merged_dir / "train"
        if train_src.exists():
            print(f"  复制训练集图片: {train_src} -> {train_dst}")
            for img_file in train_src.glob("*.png"):
                shutil.copy2(img_file, train_dst / img_file.name)
        
        # 复制测试集图片 (from normal)
        test_src = normal_dir / "train"  # normal数据集的train作为测试集
        test_dst = merged_dir / "test"
        if test_src.exists():
            print(f"  复制测试集图片: {test_src} -> {test_dst}")
            for img_file in test_src.glob("*.png"):
                shutil.copy2(img_file, test_dst / img_file.name)
        
        # 处理 transforms 文件
        process_transforms(flare_dir, normal_dir, merged_dir)
        
        # 复制其他文件 (如 points3d.ply)
        for extra_file in flare_dir.glob("*.ply"):
            shutil.copy2(extra_file, merged_dir / extra_file.name)
            
        print(f"  完成数据集合并: {merged_dir}")

def process_transforms(flare_dir, normal_dir, merged_dir):
    """处理和修正transforms文件"""
    
    # 加载原始transforms文件
    flare_train_file = flare_dir / "transforms_train.json"
    flare_test_file = flare_dir / "transforms_test.json"
    normal_train_file = normal_dir / "transforms_train.json"
    normal_test_file = normal_dir / "transforms_test.json"
    
    # 创建训练集transforms (使用flare数据)
    if flare_train_file.exists():
        with open(flare_train_file, 'r') as f:
            train_data = json.load(f)
        
        # 修正文件路径格式
        for frame in train_data["frames"]:
            old_path = frame["file_path"]
            # 将 "train\\r_00001" 转换为 "train/0001"
            if "train\\r_" in old_path:
                frame_num = old_path.split("r_")[-1]
                frame["file_path"] = f"train/{frame_num}"
        
        # 保存训练集transforms
        with open(merged_dir / "transforms_train.json", 'w') as f:
            json.dump(train_data, f, indent=4)
        print(f"  创建训练集transforms: transforms_train.json")
    
    # 创建测试集transforms (使用normal数据的train作为test)
    if normal_train_file.exists():
        with open(normal_train_file, 'r') as f:
            test_data = json.load(f)
        
        # 修正文件路径格式，将train路径改为test路径
        for frame in test_data["frames"]:
            old_path = frame["file_path"]
            if "train\\r_" in old_path:
                frame_num = old_path.split("r_")[-1]
                frame["file_path"] = f"test/{frame_num}"
        
        # 保存测试集transforms
        with open(merged_dir / "transforms_test.json", 'w') as f:
            json.dump(test_data, f, indent=4)
        print(f"  创建测试集transforms: transforms_test.json")

def verify_dataset(dataset_dir):
    """验证合并后的数据集"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"错误: 数据集不存在 {dataset_path}")
        return False
    
    print(f"\n验证数据集: {dataset_path.name}")
    
    # 检查目录结构
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    
    if not train_dir.exists():
        print("  错误: 缺少train目录")
        return False
    if not test_dir.exists():
        print("  错误: 缺少test目录")
        return False
    
    # 统计图片数量
    train_imgs = list(train_dir.glob("*.png"))
    test_imgs = list(test_dir.glob("*.png"))
    
    print(f"  训练集图片数量: {len(train_imgs)}")
    print(f"  测试集图片数量: {len(test_imgs)}")
    
    # 检查transforms文件
    train_transforms = dataset_path / "transforms_train.json"
    test_transforms = dataset_path / "transforms_test.json"
    
    if train_transforms.exists():
        with open(train_transforms, 'r') as f:
            train_data = json.load(f)
        print(f"  训练集transforms帧数: {len(train_data['frames'])}")
    else:
        print("  错误: 缺少transforms_train.json")
        return False
        
    if test_transforms.exists():
        with open(test_transforms, 'r') as f:
            test_data = json.load(f)
        print(f"  测试集transforms帧数: {len(test_data['frames'])}")
    else:
        print("  错误: 缺少transforms_test.json")
        return False
    
    # 验证图片和transforms对应关系
    expected_train = len(train_data['frames'])
    expected_test = len(test_data['frames'])
    
    if len(train_imgs) != expected_train:
        print(f"  警告: 训练集图片数量不匹配 (图片:{len(train_imgs)}, transforms:{expected_train})")
    if len(test_imgs) != expected_test:
        print(f"  警告: 测试集图片数量不匹配 (图片:{len(test_imgs)}, transforms:{expected_test})")
    
    print("  验证完成")
    return True

if __name__ == "__main__":
    print("开始合并数据集...")
    merge_datasets()
    
    print("\n验证合并结果...")
    # 验证所有生成的数据集
    base_path = Path("datasets")
    for merged_dataset in base_path.iterdir():
        if merged_dataset.is_dir() and not merged_dataset.name.endswith(('_flare', '_normal')):
            verify_dataset(merged_dataset)
    
    print("\n数据集合并完成!")