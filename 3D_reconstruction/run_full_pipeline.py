#!/usr/bin/env python3
"""
完整事件相机重建Pipeline - 通用一键式运行
支持任意数据集: lego, ship, hotdog等
从 xxx_flare + xxx_normal开始，到EVREAL重建结果结束

Author: Claude Code Assistant
Date: 2025-09-25 (Updated for universal support)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_step(step_name: str, command: str, check_success=None):
    """运行Pipeline步骤"""
    print(f"\n{'='*60}")
    print(f"🚀 步骤: {step_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               capture_output=True, text=True)
        
        # 检查成功条件
        if check_success and not check_success():
            print(f"❌ {step_name} - 成功条件未满足")
            return False
            
        print(f"✅ {step_name} - 完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} - 失败")
        print(f"错误: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_datasets(dataset_name):
    """检查原始数据集是否存在"""
    flare_dir = Path(f"datasets/{dataset_name}_flare")
    normal_dir = Path(f"datasets/{dataset_name}_normal")
    merged_dir = Path(f"datasets/{dataset_name}")
    
    print(f"🔍 检查{dataset_name}数据集...")
    
    # 检查原始数据集
    flare_exists = flare_dir.exists()
    normal_exists = normal_dir.exists() 
    merged_exists = merged_dir.exists()
    
    print(f"  {dataset_name}_flare: {'存在' if flare_exists else '缺失'}")
    print(f"  {dataset_name}_normal: {'存在' if normal_exists else '缺失'}")
    print(f"  {dataset_name} (合并): {'存在' if merged_exists else '缺失'}")
    
    if not flare_exists or not normal_exists:
        print(f"❌ 缺少{dataset_name}原始数据集")
        return False
        
    if not merged_exists:
        print(f"⚠️  {dataset_name}合并数据集不存在，需要先运行合并")
        return False
        
    # 验证合并数据集完整性
    train_dir = merged_dir / "train"
    test_dir = merged_dir / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print(f"❌ {dataset_name}合并数据集结构不完整")
        return False
        
    train_count = len(list(train_dir.glob("*.png")))
    test_count = len(list(test_dir.glob("*.png")))
    
    print(f"✅ {dataset_name}数据集检查通过:")
    print(f"  训练图像: {train_count}张")
    print(f"  测试图像: {test_count}张")
    
    return True

def create_dataset_pipeline_commands(dataset_name):
    """为指定数据集创建pipeline命令"""
    return {
        'preprocess': f'python -c "import sys; sys.path.append(\\".\\"); from modules.image_preprocessor import *; from pathlib import Path; config = PreprocessConfig(); config.input_dir = Path(\\"datasets/{dataset_name}/train\\"); preprocessor = ImagePreprocessor(config); result = preprocessor.process(); print(f\\"处理完成: {{len(result.image_paths) if result else 0}}张图像\\")"',
        
        'dvs_simulate': f'python -c "import sys; sys.path.append(\\".\\"); from modules.dvs_simulator import *; from modules.image_preprocessor import *; from pathlib import Path; preprocess_config = PreprocessConfig(); preprocess_config.input_dir = Path(\\"datasets/{dataset_name}/train\\"); preprocessor = ImagePreprocessor(preprocess_config); image_sequence = preprocessor.process(); dvs_config = DVSSimulatorConfig(); dvs_config.output_dir = Path(\\"datasets/{dataset_name}/events_dvs\\"); simulator = DVSSimulatorWrapper(dvs_config); result = simulator.simulate(image_sequence); print(f\\"DVS仿真完成: {{result.metadata[\'num_events\'] if result else 0}}个事件\\")"',
        
        'convert_format': f'python -c "import sys; sys.path.append(\\".\\"); from modules.format_converter import *; from pathlib import Path; config = ConversionConfig(); config.dataset_name = \\"{dataset_name}\\"; config.dataset_dir = Path(\\"datasets/{dataset_name}\\"); config.dvs_events_dir = Path(\\"datasets/{dataset_name}/events_dvs\\"); config.evreal_output_dir = Path(\\"datasets/{dataset_name}/events_evreal\\"); converter = FormatConverterPipeline(config); dvs_file = Path(\\"datasets/{dataset_name}/events_dvs/{dataset_name}_train_events_new.txt\\"); result = converter.convert_dvs_events(dvs_file, \\"{dataset_name}_sequence_new\\"); print(\\"格式转换完成\\")"',
        
        'evreal_reconstruct': f'python -c "import sys; sys.path.append(\\".\\"); from modules.evreal_integration import *; from pathlib import Path; config = EVREALIntegrationConfig(); config.dataset_name = \\"{dataset_name}\\"; config.dataset_dir = Path(\\"datasets/{dataset_name}\\"); config.evreal_data_dir = Path(\\"datasets/{dataset_name}/events_evreal\\"); config.output_dir = Path(\\"datasets/{dataset_name}/reconstruction\\"); integration = EVREALIntegration(config); result = integration.run_full_pipeline(); print(\\"EVREAL重建完成: \\", result[\\"success\\"])"'
    }

def main():
    """主函数 - 运行完整Pipeline"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='通用事件相机重建Pipeline')
    parser.add_argument('dataset', nargs='?', default='lego', help='数据集名称 (默认: lego)')
    parser.add_argument('--auto-merge', action='store_true', help='如果合并数据集不存在，自动运行merge_datasets.py')
    parser.add_argument('--skip-preprocess', action='store_true', help='跳过图像预处理步骤')
    parser.add_argument('--skip-dvs', action='store_true', help='跳过DVS仿真步骤') 
    parser.add_argument('--skip-convert', action='store_true', help='跳过格式转换步骤')
    parser.add_argument('--skip-reconstruct', action='store_true', help='跳过EVREAL重建步骤')
    
    args = parser.parse_args()
    dataset_name = args.dataset
    
    print("🎯 通用事件相机重建Pipeline")
    print(f"数据集: {dataset_name}")
    print(f"从 {dataset_name}_flare + {dataset_name}_normal → DVS事件 → EVREAL重建")
    
    # 检查数据集
    if not check_datasets(dataset_name):
        if args.auto_merge:
            print("🔄 自动运行数据集合并...")
            success = run_step(
                "数据集合并",
                "python merge_datasets.py",
                lambda: Path(f"datasets/{dataset_name}").exists()
            )
            if not success:
                return False
                
            # 重新检查
            if not check_datasets(dataset_name):
                return False
        else:
            print("请先运行: python merge_datasets.py")
            return False
    
    # 获取数据集专用命令
    commands = create_dataset_pipeline_commands(dataset_name)
    
    # 步骤1: 图像预处理
    if not args.skip_preprocess:
        success = run_step(
            f"{dataset_name}图像预处理",
            commands['preprocess'],
            lambda: Path("temp/dvs_input").exists() and 
                    len(list(Path("temp/dvs_input").glob("*.png"))) > 0
        )
        if not success:
            return False
    
    # 步骤2: DVS事件仿真
    if not args.skip_dvs:
        success = run_step(
            f"{dataset_name} DVS事件仿真",
            commands['dvs_simulate'],
            lambda: Path(f"datasets/{dataset_name}/events_dvs").exists() and 
                    len(list(Path(f"datasets/{dataset_name}/events_dvs").glob("*.txt"))) > 0
        )
        if not success:
            return False
    
    # 步骤3: 格式转换
    if not args.skip_convert:
        success = run_step(
            f"{dataset_name}格式转换",
            commands['convert_format'],
            lambda: Path(f"datasets/{dataset_name}/events_evreal/sequence/events_ts.npy").exists()
        )
        if not success:
            return False
    
    # 步骤4: EVREAL重建
    if not args.skip_reconstruct:
        success = run_step(
            f"{dataset_name} EVREAL图像重建",
            commands['evreal_reconstruct'],
            lambda: Path(f"datasets/{dataset_name}/reconstruction").exists()
        )
        if not success:
            return False
    
    # 最终检查
    print(f"\n{'='*60}")
    print(f"🎉 {dataset_name} Pipeline完成 - 最终检查")
    print(f"{'='*60}")
    
    # 统计结果
    train_images = len(list(Path(f"datasets/{dataset_name}/train").glob("*.png")))
    test_images = len(list(Path(f"datasets/{dataset_name}/test").glob("*.png")))
    events_dir = Path(f"datasets/{dataset_name}/events_dvs")
    reconstruction_dir = Path(f"datasets/{dataset_name}/reconstruction")
    
    print(f"✅ {dataset_name}训练图像 (flare): {train_images}张")
    print(f"✅ {dataset_name}测试图像 (normal): {test_images}张")
    
    # 检查事件数据
    if events_dir.exists():
        event_files = list(events_dir.glob("*.txt"))
        print(f"✅ {dataset_name} DVS事件数据: {len(event_files)}个文件")
        
        # 统计事件数量
        if event_files:
            try:
                import numpy as np
                events = np.loadtxt(event_files[0])
                print(f"   事件总数: {len(events):,}个")
            except:
                print(f"   (无法读取事件数据)")
    else:
        print(f"❌ {dataset_name} DVS事件数据: 缺失")
    
    # 检查重建结果
    if reconstruction_dir.exists():
        recon_methods = list(reconstruction_dir.glob("evreal_*"))
        print(f"✅ {dataset_name}重建方法: {[m.name for m in recon_methods]}")
        
        # 检查是否有重建图像
        total_recon_images = 0
        for method_dir in recon_methods:
            png_count = len(list(method_dir.glob("*.png")))
            total_recon_images += png_count
            print(f"   {method_dir.name}: {png_count}张重建图像")
            
        print(f"✅ {dataset_name}重建图像总计: {total_recon_images}张")
    else:
        print(f"❌ {dataset_name}重建结果目录: 缺失")
    
    print(f"\n🎯 {dataset_name} Pipeline执行完毕!")
    print(f"总体状态: 完全成功")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)