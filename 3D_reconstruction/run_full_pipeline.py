#!/usr/bin/env python3
"""
完整事件相机重建Pipeline - 一键式运行
从lego_flare + lego_normal开始，到EVREAL重建结果结束

Author: Claude Code Assistant
Date: 2025-09-19
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
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} - 失败")
        print(f"错误: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_datasets():
    """检查原始数据集是否存在"""
    flare_dir = Path("datasets/lego_flare")
    normal_dir = Path("datasets/lego_normal")
    
    if not flare_dir.exists() or not normal_dir.exists():
        print(f"❌ 缺少原始数据集:")
        print(f"  lego_flare: {'存在' if flare_dir.exists() else '缺失'}")
        print(f"  lego_normal: {'存在' if normal_dir.exists() else '缺失'}")
        return False
    return True

def main():
    """主函数 - 运行完整Pipeline"""
    print("🎯 事件相机完整重建Pipeline")
    print("从 lego_flare + lego_normal → DVS事件 → EVREAL重建")
    
    # 检查原始数据集
    if not check_datasets():
        return False
    
    # 步骤1: 数据集合并
    success = run_step(
        "数据集合并",
        "python merge_datasets.py",
        lambda: Path("datasets/lego/train").exists() and 
                Path("datasets/lego/test").exists()
    )
    if not success:
        return False
    
    # 步骤2: 图像预处理
    success = run_step(
        "图像预处理",
        "python modules/image_preprocessor.py",
        lambda: Path("temp/dvs_input").exists() and 
                len(list(Path("temp/dvs_input").glob("*.png"))) > 0
    )
    if not success:
        return False
    
    # 步骤3: DVS事件仿真
    success = run_step(
        "DVS事件仿真",
        "python modules/dvs_simulator.py",
        lambda: Path("datasets/lego/events_dvs/lego_train_events.txt").exists()
    )
    if not success:
        return False
    
    # 步骤4: 格式转换
    success = run_step(
        "格式转换",
        "python modules/format_converter.py",
        lambda: Path("datasets/lego/events_evreal/sequence/events_ts.npy").exists()
    )
    if not success:
        return False
    
    # 步骤5: EVREAL重建
    success = run_step(
        "EVREAL图像重建",
        "python modules/evreal_integration.py",
        lambda: Path("datasets/lego/reconstruction").exists()
    )
    if not success:
        return False
    
    # 最终检查
    print(f"\n{'='*60}")
    print("🎉 Pipeline完成 - 最终检查")
    print(f"{'='*60}")
    
    # 统计结果
    train_images = len(list(Path("datasets/lego/train").glob("*.png")))
    test_images = len(list(Path("datasets/lego/test").glob("*.png"))) 
    event_file = Path("datasets/lego/events_dvs/lego_train_events.txt")
    reconstruction_dir = Path("datasets/lego/reconstruction")
    
    print(f"✅ 训练图像 (flare): {train_images}张")
    print(f"✅ 测试图像 (normal): {test_images}张")
    print(f"✅ DVS事件数据: {'存在' if event_file.exists() else '缺失'}")
    print(f"✅ 重建结果目录: {'存在' if reconstruction_dir.exists() else '缺失'}")
    
    if reconstruction_dir.exists():
        recon_methods = list(reconstruction_dir.glob("evreal_*"))
        print(f"✅ 重建方法: {[m.name for m in recon_methods]}")
        
        # 检查是否有重建图像
        total_recon_images = 0
        for method_dir in recon_methods:
            png_count = len(list(method_dir.glob("*.png")))
            total_recon_images += png_count
            print(f"   {method_dir.name}: {png_count}张重建图像")
    
    print(f"\n🎯 Pipeline执行完毕!")
    print(f"总体状态: {'完全成功' if success else '部分失败'}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)