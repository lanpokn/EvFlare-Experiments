#!/usr/bin/env python3
"""
为不同重建方法和H5数据源生成3DGS训练用的JSON配置文件
支持: reconstruction_original, reconstruction_Unet, reconstruction_Unetsimple
用法: 
- python generate_json_configs.py lego2                      # 为所有方法生成JSON
- python generate_json_configs.py lego2 spade-e2vid          # 只为spade-e2vid方法生成JSON (三种H5数据源)
"""

import json
import os
from pathlib import Path

def create_json_config(dataset_name, method_name, h5_type, images_subdir):
    """
    创建单个重建方法的JSON配置文件
    
    Args:
        dataset_name: 数据集名称 (如 lego2)
        method_name: 重建方法名 (如 spade-e2vid, firenet, original)
        h5_type: H5数据源类型 (original, Unet, Unetsimple)
        images_subdir: 图像子目录路径
    """
    dataset_dir = Path(f"datasets/{dataset_name}")
    
    # 读取原始的transforms_train.json作为模板
    template_path = dataset_dir / "transforms_train.json"
    if not template_path.exists():
        print(f"❌ 模板文件不存在: {template_path}")
        return None
    
    with open(template_path, 'r') as f:
        config = json.load(f)
    
    # 检查图像目录是否存在
    images_path = dataset_dir / images_subdir
    if not images_path.exists():
        print(f"⚠️  图像目录不存在: {images_path}")
        return None
    
    # 更新所有frame的file_path
    for frame in config['frames']:
        old_path = frame['file_path']  # 如 "train/0001"
        filename = Path(old_path).name  # 提取 "0001"
        
        # 设置新的路径
        new_path = f"{images_subdir}/{filename}"
        frame['file_path'] = new_path
    
    # 生成配置文件名
    if method_name == "original":
        output_filename = f"transforms_train_{method_name}.json"
    else:
        output_filename = f"transforms_train_{method_name}_{h5_type}.json"
    
    output_path = dataset_dir / output_filename
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ 已生成: {output_filename}")
    return output_path

def get_available_h5_sources(dataset_name):
    """获取可用的H5数据源列表"""
    dataset_dir = Path(f"datasets/{dataset_name}")
    h5_sources = []
    
    # 扫描所有reconstruction_*目录
    for item in dataset_dir.iterdir():
        if item.is_dir() and item.name.startswith('reconstruction'):
            if item.name == 'reconstruction':
                continue  # 跳过旧的reconstruction目录
            
            # 提取H5类型
            if item.name == 'reconstruction_original':
                h5_type = 'original'
            else:
                h5_type = item.name.replace('reconstruction_', '')
            
            h5_sources.append((h5_type, item.name))
    
    return h5_sources

def get_available_methods(dataset_name, h5_sources):
    """获取所有H5数据源中的可用重建方法"""
    dataset_dir = Path(f"datasets/{dataset_name}")
    all_methods = set()
    
    # 扫描每个H5数据源目录找到通用方法
    for h5_type, h5_dir_name in h5_sources:
        h5_dir = dataset_dir / h5_dir_name
        if not h5_dir.exists():
            continue
            
        for method_dir in h5_dir.iterdir():
            if method_dir.is_dir() and method_dir.name.startswith('evreal_'):
                method_name = method_dir.name.replace('evreal_', '').replace('-', '_')
                all_methods.add(method_name)
    
    return sorted(list(all_methods))

def main():
    """主函数"""
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "lego2"
    
    # 支持指定单个方法
    if len(sys.argv) > 2:
        target_method = sys.argv[2].replace('-', '_')  # 统一使用下划线
    else:
        target_method = None
    
    dataset_dir = Path(f"datasets/{dataset_name}")
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return
    
    print(f"🚀 为数据集 {dataset_name} 生成JSON配置文件...")
    if target_method:
        print(f"🎯 指定重建方法: {target_method}")
    
    # 1. 获取所有H5数据源
    h5_sources = get_available_h5_sources(dataset_name)
    if not h5_sources:
        print(f"⚠️  未找到H5重建数据源目录")
        return
    
    print(f"📁 找到 {len(h5_sources)} 个H5数据源: {[h5_type for h5_type, _ in h5_sources]}")
    
    # 2. 获取所有可用方法
    available_methods = get_available_methods(dataset_name, h5_sources)
    if not available_methods:
        print(f"⚠️  未找到可用的重建方法")
        return
    
    print(f"🔧 找到 {len(available_methods)} 个重建方法: {available_methods}")
    
    # 3. 生成原始训练数据的配置 (只生成一次)
    generated_configs = []
    
    if (dataset_dir / "train").exists():
        config_path = create_json_config(dataset_name, "original", "original", "train")
        if config_path:
            generated_configs.append(("original", "original"))
    
    # 4. 为每个指定方法的每个H5数据源生成配置
    for method_name in available_methods:
        # 如果指定了特定方法，只处理该方法
        if target_method and target_method != method_name:
            continue
        
        for h5_type, h5_dir_name in h5_sources:
            # 构建图像路径
            evreal_method_name = method_name.replace('_', '-')
            images_subdir = f"{h5_dir_name}/evreal_{evreal_method_name}"
            
            # 检查方法目录是否存在
            method_path = dataset_dir / images_subdir
            if not method_path.exists():
                print(f"⚠️  跳过不存在的方法: {images_subdir}")
                continue
            
            # 生成配置文件
            config_path = create_json_config(
                dataset_name,
                method_name,
                h5_type,
                images_subdir
            )
            
            if config_path:
                config_key = f"{method_name}_{h5_type}"
                generated_configs.append((config_key, h5_type))
    
    if not generated_configs:
        print(f"⚠️  没有生成任何配置文件")
        return
    
    print(f"\n🎉 配置生成完成！共生成 {len(generated_configs)} 个配置文件")
    
    # 5. 生成训练配置组合列表
    print(f"\n📋 生成的训练配置:")
    print(f"1. 原始训练: transforms_train_original.json")
    
    # 按方法分组显示
    if target_method:
        print(f"2. {target_method} 方法的三种H5数据源:")
        for config_key, h5_type in generated_configs:
            if config_key.startswith(target_method) and h5_type != 'original':
                print(f"   - transforms_train_{config_key}.json ({h5_type})")
    else:
        method_groups = {}
        for config_key, h5_type in generated_configs:
            if h5_type != 'original':
                method = config_key.split('_')[0] + '_' + config_key.split('_')[1] if '_' in config_key.split('_', 1)[1] else config_key.split('_')[0]
                if method not in method_groups:
                    method_groups[method] = []
                method_groups[method].append((config_key, h5_type))
        
        for i, (method, configs) in enumerate(method_groups.items(), 2):
            print(f"{i}. {method} 方法:")
            for config_key, h5_type in configs:
                print(f"   - transforms_train_{config_key}.json ({h5_type})")
    
    # 6. 生成Windows批处理脚本可用的方法列表
    if target_method:
        # 如果指定了方法，生成该方法的H5数据源列表
        training_combinations = ["original"]  # 总是包含原始训练
        for config_key, h5_type in generated_configs:
            if config_key.startswith(target_method):
                training_combinations.append(config_key)
        
        methods_file = dataset_dir / f"training_methods_{target_method}.txt"
        with open(methods_file, 'w') as f:
            for combo in training_combinations:
                f.write(f"{combo}\n")
        
        print(f"\n📝 已生成方法列表: training_methods_{target_method}.txt")
        print(f"🎯 {target_method} 训练组合: {training_combinations}")
    else:
        print(f"\n💡 使用方式:")
        print(f"   指定方法重新生成: python generate_json_configs.py {dataset_name} spade-e2vid")
        print(f"   然后运行批处理: auto_train_3dgs.bat {dataset_name} spade_e2vid")

if __name__ == "__main__":
    main()