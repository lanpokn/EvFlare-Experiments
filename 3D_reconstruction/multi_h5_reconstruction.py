#!/usr/bin/env python3
"""
通用多H5文件重建脚本
支持对任意数据集中的多个H5文件进行EVREAL重建
每个H5文件生成独立的重建目录

使用方法:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && python multi_h5_reconstruction.py [dataset_name]

示例:
python multi_h5_reconstruction.py lego2
python multi_h5_reconstruction.py ship  

Author: Claude Code Assistant
Date: 2025-09-25
"""

import os
import sys
import h5py
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time

# 添加项目路径
sys.path.append('.')
sys.path.append('..')

# 导入必要的模块
try:
    from pipeline_architecture import DataFormat, FormatConverter
    from modules.format_converter import ConversionConfig, DVSToEVREALConverter
    from modules.evreal_integration import EVREALIntegrationConfig, EVREALIntegration
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确的项目根目录下运行此脚本")
    sys.exit(1)

class MultiH5ReconstructionManager:
    """通用多H5文件重建管理器"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_dir = Path("datasets") / dataset_name
        self.h5_dir = self.dataset_dir / "events_h5"
        self.original_evreal_dir = self.dataset_dir / "events_evreal"
        
        # 验证必要目录存在
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")
        if not self.h5_dir.exists():
            raise FileNotFoundError(f"H5目录不存在: {self.h5_dir}")
        if not self.original_evreal_dir.exists():
            raise FileNotFoundError(f"原始EVREAL目录不存在: {self.original_evreal_dir}")
    
    def get_h5_files(self) -> List[Path]:
        """获取所有需要处理的H5文件"""
        h5_files = []
        for h5_file in self.h5_dir.glob("*.h5"):
            # 跳过备份文件
            if "backup" in h5_file.name or "wrong" in h5_file.name:
                continue
            h5_files.append(h5_file)
        
        h5_files.sort()  # 按名称排序
        print(f"找到 {len(h5_files)} 个H5文件待处理:")
        for h5_file in h5_files:
            print(f"  - {h5_file.name}")
        return h5_files
    
    def get_suffix_from_filename(self, h5_file: Path) -> str:
        """从H5文件名提取后缀，支持多种命名模式"""
        stem = h5_file.stem  # 去掉.h5扩展名
        
        # 尝试多种命名模式
        base_patterns = [
            f"{self.dataset_name}_sequence_new",
            f"{self.dataset_name}_sequence", 
            f"{self.dataset_name}_train_events",
            f"{self.dataset_name}"
        ]
        
        for base_pattern in base_patterns:
            if stem.startswith(base_pattern):
                suffix = stem.replace(base_pattern, "")
                if suffix.startswith("_"):
                    return suffix[1:]  # 移除前导下划线
                elif suffix == "":
                    return "original"  # 原始文件
                else:
                    return suffix
        
        # 如果没有匹配的模式，使用完整文件名作为后缀
        return stem.replace(".", "_")
    
    def h5_to_evreal_conversion(self, h5_file: Path, suffix: str) -> Path:
        """将H5文件转换为EVREAL格式"""
        print(f"\n=== 转换H5文件: {h5_file.name} (后缀: {suffix}) ===")
        
        # 创建临时EVREAL目录
        temp_evreal_dir = self.dataset_dir / f"events_evreal_{suffix}"
        sequence_dir = temp_evreal_dir / "sequence"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 读取H5文件
            print(f"读取H5文件: {h5_file}")
            with h5py.File(h5_file, 'r') as f:
                # 检查H5文件结构
                if 'events' in f and isinstance(f['events'], h5py.Group):
                    # 格式: events/t, events/x, events/y, events/p
                    events_t = f['events/t'][:]
                    events_x = f['events/x'][:]
                    events_y = f['events/y'][:]
                    events_p = f['events/p'][:]
                    print(f"H5事件数量: {len(events_t)} (分组格式)")
                elif 'events' in f and isinstance(f['events'], h5py.Dataset):
                    # 格式: events[:, [t,x,y,p]]
                    events_data = f['events'][:]
                    events_t = events_data[:, 0]
                    events_x = events_data[:, 1] 
                    events_y = events_data[:, 2]
                    events_p = events_data[:, 3]
                    print(f"H5事件数量: {len(events_data)} (数组格式)")
                else:
                    raise ValueError("未识别的H5文件格式")
                
                # 转换为EVREAL格式
                events_ts = events_t.astype(np.float64) / 1000000.0  # 微秒转秒
                events_xy = np.column_stack([events_x.astype(np.int16), events_y.astype(np.int16)])  # [x, y]
                events_p = events_p.astype(np.int8)  # polarity
                
                # 保存EVREAL事件文件
                np.save(sequence_dir / "events_ts.npy", events_ts)
                np.save(sequence_dir / "events_xy.npy", events_xy)
                np.save(sequence_dir / "events_p.npy", events_p)
                
                print(f"转换完成:")
                print(f"  - 时间范围: {events_ts.min():.6f}s - {events_ts.max():.6f}s")
                print(f"  - 空间范围: X[{events_xy[:, 0].min()}, {events_xy[:, 0].max()}], Y[{events_xy[:, 1].min()}, {events_xy[:, 1].max()}]")
                print(f"  - 极性分布: +{(events_p == 1).sum()}, -{(events_p == 0).sum()}")
            
            # 复制图像相关文件（从原始EVREAL目录）
            print("复制图像相关文件...")
            original_sequence_dir = self.original_evreal_dir / "sequence"
            
            files_to_copy = [
                "images.npy",
                "images_ts.npy", 
                "image_event_indices.npy",
                "metadata.json"
            ]
            
            for file_name in files_to_copy:
                src_file = original_sequence_dir / file_name
                dst_file = sequence_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✓ {file_name}")
                else:
                    print(f"  ⚠ 未找到: {file_name}")
            
            # 复制images目录
            src_images_dir = original_sequence_dir / "images"
            dst_images_dir = sequence_dir / "images"
            if src_images_dir.exists():
                if dst_images_dir.exists():
                    shutil.rmtree(dst_images_dir)
                shutil.copytree(src_images_dir, dst_images_dir)
                print(f"  ✓ images/ 目录 ({len(list(dst_images_dir.glob('*.png')))} 张图像)")
            
            # 更新元数据
            metadata_file = sequence_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # 更新事件相关信息
                metadata["num_events"] = len(events_ts)
                metadata["time_range_s"] = [float(events_ts.min()), float(events_ts.max())]
                metadata["spatial_range"] = {
                    "x_range": [int(events_xy[:, 0].min()), int(events_xy[:, 0].max())],
                    "y_range": [int(events_xy[:, 1].min()), int(events_xy[:, 1].max())]
                }
                metadata["source_h5_file"] = str(h5_file)
                metadata["conversion_suffix"] = suffix
                metadata["conversion_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"  ✓ 更新metadata.json")
            
            print(f"✅ H5→EVREAL转换完成: {temp_evreal_dir}")
            return temp_evreal_dir
            
        except Exception as e:
            print(f"❌ H5→EVREAL转换失败: {e}")
            if temp_evreal_dir.exists():
                shutil.rmtree(temp_evreal_dir)
            raise e
    
    def run_evreal_reconstruction(self, evreal_dir: Path, suffix: str) -> Path:
        """运行EVREAL重建"""
        print(f"\n=== EVREAL重建: {suffix} ===")
        
        try:
            # 配置EVREAL集成
            config = EVREALIntegrationConfig()
            config.dataset_name = f"{self.dataset_name}_{suffix}"
            config.dataset_dir = self.dataset_dir
            config.evreal_data_dir = evreal_dir  # 使用临时EVREAL目录
            config.reconstruction_dir = self.dataset_dir / f"reconstruction_{suffix}"
            
            # 运行重建
            integration = EVREALIntegration(config)
            result = integration.run_full_pipeline()
            
            if result.get("successful_methods"):
                print(f"✅ 重建成功: {result['successful_methods']}")
                print(f"   输出目录: {config.output_base_dir}")
                return config.output_base_dir
            else:
                print(f"❌ 重建失败: 没有成功的方法")
                return None
                
        except Exception as e:
            print(f"❌ EVREAL重建失败: {e}")
            return None
    
    def cleanup_temp_evreal(self, temp_evreal_dir: Path):
        """清理临时EVREAL目录"""
        try:
            if temp_evreal_dir.exists():
                shutil.rmtree(temp_evreal_dir)
                print(f"✓ 清理临时目录: {temp_evreal_dir}")
        except Exception as e:
            print(f"⚠ 清理临时目录失败: {e}")
    
    def process_all_h5_files(self):
        """处理所有H5文件"""
        h5_files = self.get_h5_files()
        if not h5_files:
            print("❌ 未找到可处理的H5文件")
            return
        
        print(f"\n🚀 开始处理 {len(h5_files)} 个H5文件...")
        
        results = []
        for i, h5_file in enumerate(h5_files, 1):
            print(f"\n{'='*60}")
            print(f"处理进度: {i}/{len(h5_files)} - {h5_file.name}")
            print(f"{'='*60}")
            
            suffix = self.get_suffix_from_filename(h5_file)
            temp_evreal_dir = None
            
            try:
                # H5→EVREAL转换
                temp_evreal_dir = self.h5_to_evreal_conversion(h5_file, suffix)
                
                # EVREAL重建
                reconstruction_dir = self.run_evreal_reconstruction(temp_evreal_dir, suffix)
                
                results.append({
                    "h5_file": h5_file.name,
                    "suffix": suffix,
                    "reconstruction_dir": reconstruction_dir,
                    "success": reconstruction_dir is not None
                })
                
            except Exception as e:
                print(f"❌ 处理文件失败: {h5_file.name}, 错误: {e}")
                results.append({
                    "h5_file": h5_file.name,
                    "suffix": suffix,
                    "reconstruction_dir": None,
                    "success": False,
                    "error": str(e)
                })
            
            finally:
                # 暂时不清理临时目录，用于调试
                if temp_evreal_dir and False:  # 临时禁用清理
                    self.cleanup_temp_evreal(temp_evreal_dir)
        
        # 输出最终结果
        self.print_final_results(results)
    
    def print_final_results(self, results: List[Dict]):
        """打印最终处理结果"""
        print(f"\n{'='*60}")
        print("🎉 处理完成! 结果汇总:")
        print(f"{'='*60}")
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"✅ 成功: {len(successful)}/{len(results)}")
        for result in successful:
            print(f"   {result['h5_file']} → reconstruction_{result['suffix']}/")
        
        if failed:
            print(f"\n❌ 失败: {len(failed)}/{len(results)}")
            for result in failed:
                error_msg = result.get("error", "未知错误")
                print(f"   {result['h5_file']}: {error_msg}")
        
        # 计算输出统计
        if successful:
            print(f"\n📊 重建图像统计:")
            for result in successful:
                if result["reconstruction_dir"] and Path(result["reconstruction_dir"]).exists():
                    reconstruction_dir = Path(result["reconstruction_dir"])
                    png_count = len(list(reconstruction_dir.rglob("*.png")))
                    methods = [d.name for d in reconstruction_dir.iterdir() if d.is_dir()]
                    print(f"   {result['suffix']}: {png_count} 张图像, {len(methods)} 种方法")


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("❌ 错误: 请指定数据集名称")
        print("使用方法: python multi_h5_reconstruction.py <dataset_name>")
        print("示例: python multi_h5_reconstruction.py lego2")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    print(f"🔥 {dataset_name} 多H5文件重建脚本")
    print("=" * 60)
    
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"当前conda环境: {conda_env}")
    if conda_env != 'Umain2':
        print("⚠️  警告: 未在Umain2环境中运行!")
        print("请使用: source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2")
    
    try:
        # 创建管理器并运行
        manager = MultiH5ReconstructionManager(dataset_name)
        manager.process_all_h5_files()
        
    except Exception as e:
        print(f"❌ 脚本执行失败: {e}")
        sys.exit(1)
    
    print(f"\n🎯 {dataset_name} 脚本执行完成!")


if __name__ == "__main__":
    main()