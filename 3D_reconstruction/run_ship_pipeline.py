#!/usr/bin/env python3
"""
Ship数据集完整事件相机重建Pipeline - 直接处理ship数据
不改动lego数据，直接为ship数据集创建完整pipeline

Author: Claude Code Assistant  
Date: 2025-09-25
"""

import os
import sys
import subprocess
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
        if result.stdout.strip():
            print("输出:", result.stdout.strip())
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} - 失败")
        print(f"错误: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_ship_datasets():
    """检查ship数据集是否存在"""
    ship_dir = Path("datasets/ship")
    train_dir = ship_dir / "train"
    test_dir = ship_dir / "test"
    
    if not ship_dir.exists():
        print("❌ ship数据集不存在，请先运行 python merge_datasets.py")
        return False
        
    if not train_dir.exists() or not test_dir.exists():
        print(f"❌ ship数据集结构不完整:")
        print(f"  train目录: {'存在' if train_dir.exists() else '缺失'}")
        print(f"  test目录: {'存在' if test_dir.exists() else '缺失'}")
        return False
    
    train_count = len(list(train_dir.glob("*.png")))
    test_count = len(list(test_dir.glob("*.png")))
    
    print(f"✅ Ship数据集检查通过:")
    print(f"  训练图像: {train_count}张")
    print(f"  测试图像: {test_count}张")
    
    return True

def main():
    """主函数 - 直接处理ship数据集"""
    print("🎯 Ship数据集事件相机完整重建Pipeline")
    print("直接处理ship数据，不影响lego数据")
    
    # 检查ship数据集
    if not check_ship_datasets():
        return False
    
    # 步骤1: Ship图像预处理
    success = run_step(
        "Ship图像预处理",
        'python -c "import sys; sys.path.append(\\".\\"); from modules.image_preprocessor import *; from pathlib import Path; config = PreprocessConfig(); config.input_dir = Path(\\"datasets/ship/train\\"); preprocessor = ImagePreprocessor(config); result = preprocessor.process(); print(f\\"处理完成: {len(result.image_paths) if result else 0}张图像\\")"',
        lambda: Path("temp/dvs_input").exists() and len(list(Path("temp/dvs_input").glob("*.png"))) > 0
    )
    if not success:
        return False
    
    # 步骤2: Ship DVS事件仿真
    success = run_step(
        "Ship DVS事件仿真",
        'python -c "import sys; sys.path.append(\\".\\"); from modules.dvs_simulator import *; from modules.image_preprocessor import *; from pathlib import Path; preprocess_config = PreprocessConfig(); preprocess_config.input_dir = Path(\\"datasets/ship/train\\"); preprocessor = ImagePreprocessor(preprocess_config); image_sequence = preprocessor.process(); dvs_config = DVSSimulatorConfig(); dvs_config.output_dir = Path(\\"datasets/ship/events_dvs\\"); simulator = DVSSimulatorWrapper(dvs_config); result = simulator.simulate(image_sequence); print(f\\"DVS仿真完成: {result.metadata[\'num_events\'] if result else 0}个事件\\")"',
        lambda: Path("datasets/ship/events_dvs").exists() and len(list(Path("datasets/ship/events_dvs").glob("*.txt"))) > 0
    )
    if not success:
        return False
    
    # 步骤3: Ship格式转换
    success = run_step(
        "Ship格式转换",
        'python -c "import sys; sys.path.append(\\".\\"); from modules.format_converter import *; from pathlib import Path; config = FormatConverterConfig(); config.dataset_name = \\"ship\\"; config.dataset_dir = Path(\\"datasets/ship\\"); config.dvs_events_dir = Path(\\"datasets/ship/events_dvs\\"); config.evreal_output_dir = Path(\\"datasets/ship/events_evreal\\"); converter = FormatConverter(config); result = converter.convert_all_formats(); print(\\"格式转换完成\\")"',
        lambda: Path("datasets/ship/events_evreal/sequence/events_ts.npy").exists()
    )
    if not success:
        return False
    
    # 步骤4: Ship EVREAL重建
    success = run_step(
        "Ship EVREAL图像重建",
        'python -c "import sys; sys.path.append(\\".\\"); from modules.evreal_integration import *; from pathlib import Path; config = EVREALIntegrationConfig(); config.dataset_name = \\"ship\\"; config.dataset_dir = Path(\\"datasets/ship\\"); config.evreal_data_dir = Path(\\"datasets/ship/events_evreal\\"); config.output_dir = Path(\\"datasets/ship/reconstruction\\"); integration = EVREALIntegration(config); result = integration.run_reconstruction(); print(\\"EVREAL重建完成\\")"',
        lambda: Path("datasets/ship/reconstruction").exists()
    )
    if not success:
        return False
    
    # 最终检查
    print(f"\n{'='*60}")
    print("🎉 Ship Pipeline完成 - 最终检查")
    print(f"{'='*60}")
    
    # 统计结果
    train_images = len(list(Path("datasets/ship/train").glob("*.png")))
    test_images = len(list(Path("datasets/ship/test").glob("*.png")))
    
    print(f"✅ Ship训练图像 (flare): {train_images}张")
    print(f"✅ Ship测试图像 (normal): {test_images}张")
    
    # 检查事件数据
    events_dir = Path("datasets/ship/events_dvs")
    if events_dir.exists():
        event_files = list(events_dir.glob("*.txt"))
        print(f"✅ Ship事件数据: {len(event_files)}个文件")
        
        # 统计事件数量
        if event_files:
            try:
                import numpy as np
                events = np.loadtxt(event_files[0])
                print(f"   事件总数: {len(events):,}个")
            except:
                pass
    
    # 检查EVREAL数据
    evreal_dir = Path("datasets/ship/events_evreal")
    if evreal_dir.exists():
        print(f"✅ Ship EVREAL数据: 存在")
        if (evreal_dir / "sequence").exists():
            print(f"   序列数据: 完整")
    
    # 检查重建结果
    recon_dir = Path("datasets/ship/reconstruction")
    if recon_dir.exists():
        recon_methods = list(recon_dir.glob("evreal_*"))
        print(f"✅ Ship重建方法: {[m.name for m in recon_methods]}")
        
        total_images = 0
        for method_dir in recon_methods:
            count = len(list(method_dir.glob("*.png")))
            total_images += count
            print(f"   {method_dir.name}: {count}张重建图像")
        
        print(f"✅ Ship重建图像总计: {total_images}张")
    
    print(f"\n🎯 Ship数据集Pipeline执行完毕!")
    print(f"总体状态: 完全成功")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)