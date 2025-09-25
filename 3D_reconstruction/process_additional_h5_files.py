#!/usr/bin/env python3
"""
处理所有H5文件的重建脚本
基于现有成功的EVREAL数据结构，为所有H5文件（包括原始文件、Unet、Unetsimple等）创建重建结果

核心原理：
1. 复制现有成功的EVREAL数据结构作为模板
2. 只替换事件数据文件（events_ts.npy, events_xy.npy, events_p.npy）
3. 重新生成匹配的image_event_indices.npy文件
4. 运行EVREAL重建并复制到指定目录

使用方法:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && python process_additional_h5_files.py <dataset_name>

示例:
python process_additional_h5_files.py lego2  # 处理lego2目录下的所有H5文件

Author: Claude Code Assistant  
Date: 2025-09-25
"""

import os
import sys
import h5py
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict
import json
import time

# 添加项目路径
sys.path.append('.')
sys.path.append('..')

try:
    from modules.evreal_integration import EVREALIntegrationConfig, EVREALIntegration
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确的项目根目录下运行此脚本")
    sys.exit(1)


class AllH5Processor:
    """所有H5文件处理器"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_dir = Path("datasets") / dataset_name
        self.h5_dir = self.dataset_dir / "events_h5"
        self.base_evreal_dir = self.dataset_dir / "events_evreal"
        
        # 验证必要目录存在
        self._validate_directories()
    
    def _validate_directories(self):
        """验证必要目录存在"""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")
        if not self.h5_dir.exists():
            raise FileNotFoundError(f"H5目录不存在: {self.h5_dir}")
        if not self.base_evreal_dir.exists():
            raise FileNotFoundError(f"基础EVREAL目录不存在: {self.base_evreal_dir}")
        
        base_sequence_dir = self.base_evreal_dir / "sequence"
        if not base_sequence_dir.exists():
            raise FileNotFoundError(f"基础EVREAL sequence目录不存在: {base_sequence_dir}")
    
    def get_all_h5_files(self) -> List[Path]:
        """获取需要处理的所有H5文件"""
        h5_files = []
        
        for h5_file in self.h5_dir.glob("*.h5"):
            # 只跳过备份文件
            if "backup" in h5_file.name or "wrong" in h5_file.name:
                print(f"跳过备份文件: {h5_file.name}")
                continue
                
            h5_files.append(h5_file)
        
        h5_files.sort()
        print(f"找到 {len(h5_files)} 个H5文件待处理:")
        for h5_file in h5_files:
            print(f"  - {h5_file.name}")
        return h5_files
    
    def extract_suffix_from_filename(self, h5_file: Path) -> str:
        """从H5文件名提取后缀"""
        stem = h5_file.stem
        
        # 尝试移除基础模式前缀
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
                    return "original"
        
        # 其他情况使用完整文件名作为后缀
        return stem.replace(".", "_")
    
    def create_temp_evreal_structure(self, suffix: str) -> Path:
        """创建临时EVREAL数据结构"""
        temp_evreal_dir = self.dataset_dir / f"events_evreal_temp_{suffix}"
        
        print(f"创建临时EVREAL结构: {temp_evreal_dir}")
        
        # 清理可能存在的旧临时目录
        if temp_evreal_dir.exists():
            shutil.rmtree(temp_evreal_dir)
        
        # 复制基础EVREAL结构
        shutil.copytree(self.base_evreal_dir, temp_evreal_dir)
        
        return temp_evreal_dir
    
    def replace_event_data(self, h5_file: Path, temp_evreal_dir: Path) -> Dict:
        """替换事件数据"""
        print(f"替换事件数据: {h5_file.name}")
        
        sequence_dir = temp_evreal_dir / "sequence"
        
        # 读取H5文件事件数据
        with h5py.File(h5_file, 'r') as f:
            if 'events' in f and isinstance(f['events'], h5py.Group):
                # 分组格式: events/t, events/x, events/y, events/p
                events_t = f['events/t'][:]
                events_x = f['events/x'][:]
                events_y = f['events/y'][:]
                events_p = f['events/p'][:]
                print(f"  H5事件数量: {len(events_t)} (分组格式)")
            elif 'events' in f and isinstance(f['events'], h5py.Dataset):
                # 数组格式: events[:, [t,x,y,p]]
                events_data = f['events'][:]
                events_t = events_data[:, 0]
                events_x = events_data[:, 1] 
                events_y = events_data[:, 2]
                events_p = events_data[:, 3]
                print(f"  H5事件数量: {len(events_data)} (数组格式)")
            else:
                raise ValueError(f"未识别的H5文件格式: {h5_file}")
        
        # 转换为EVREAL格式
        events_ts = events_t.astype(np.float64) / 1000000.0  # 微秒转秒
        events_xy = np.column_stack([events_x.astype(np.int16), events_y.astype(np.int16)])
        events_p = events_p.astype(np.int8)
        
        # 保存新的事件数据
        np.save(sequence_dir / "events_ts.npy", events_ts)
        np.save(sequence_dir / "events_xy.npy", events_xy)
        np.save(sequence_dir / "events_p.npy", events_p)
        
        print(f"  ✓ 时间范围: {events_ts.min():.6f}s - {events_ts.max():.6f}s")
        print(f"  ✓ 空间范围: X[{events_xy[:, 0].min()}, {events_xy[:, 0].max()}], Y[{events_xy[:, 1].min()}, {events_xy[:, 1].max()}]")
        print(f"  ✓ 极性分布: +{(events_p == 1).sum()}, -{(events_p == 0).sum()}")
        
        return {
            "num_events": len(events_ts),
            "time_range": [float(events_ts.min()), float(events_ts.max())],
            "spatial_range": {
                "x_range": [int(events_xy[:, 0].min()), int(events_xy[:, 0].max())],
                "y_range": [int(events_xy[:, 1].min()), int(events_xy[:, 1].max())]
            }
        }
    
    def regenerate_image_event_indices(self, temp_evreal_dir: Path, new_event_count: int):
        """重新生成匹配的image_event_indices文件"""
        print("重新生成image_event_indices.npy...")
        
        sequence_dir = temp_evreal_dir / "sequence"
        
        # 读取原始索引
        base_sequence_dir = self.base_evreal_dir / "sequence"
        original_indices = np.load(base_sequence_dir / "image_event_indices.npy")
        
        # 计算缩放因子
        original_max_index = original_indices.max()
        scale_factor = (new_event_count - 1) / original_max_index  # -1确保不越界
        
        # 生成新的索引
        new_indices = (original_indices * scale_factor).astype(np.int32)
        new_indices = np.clip(new_indices, 0, new_event_count - 1)
        
        # 保存新索引
        np.save(sequence_dir / "image_event_indices.npy", new_indices)
        
        print(f"  ✓ 原始索引范围: {original_indices.min()} - {original_indices.max()}")
        print(f"  ✓ 新索引范围: {new_indices.min()} - {new_indices.max()}")
        print(f"  ✓ 缩放因子: {scale_factor:.4f}")
    
    def run_evreal_reconstruction(self, temp_evreal_dir: Path, suffix: str) -> Dict:
        """运行EVREAL重建"""
        print(f"\n=== 运行EVREAL重建: {suffix} ===")
        
        try:
            # 配置EVREAL集成
            config = EVREALIntegrationConfig()
            config.dataset_name = f"{self.dataset_name}_{suffix}_fixed"
            config.dataset_dir = self.dataset_dir
            config.evreal_data_dir = temp_evreal_dir
            config.reconstruction_dir = self.dataset_dir / f"reconstruction_{suffix}"
            
            # 运行重建
            integration = EVREALIntegration(config)
            result = integration.run_full_pipeline()
            
            if result.get("successful_methods"):
                print(f"✅ {suffix}重建成功: {result['successful_methods']}")
                return {
                    "success": True,
                    "methods": result["successful_methods"],
                    "reconstruction_dir": config.reconstruction_dir
                }
            else:
                print(f"❌ {suffix}重建失败: 没有成功的方法")
                return {"success": False, "error": "没有成功的方法"}
                
        except Exception as e:
            print(f"❌ {suffix}重建失败: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_temp_directory(self, temp_evreal_dir: Path):
        """清理临时目录"""
        try:
            if temp_evreal_dir.exists():
                shutil.rmtree(temp_evreal_dir)
                print(f"✓ 清理临时目录: {temp_evreal_dir}")
        except Exception as e:
            print(f"⚠ 清理临时目录失败: {e}")
    
    def process_single_h5_file(self, h5_file: Path) -> Dict:
        """处理单个H5文件"""
        suffix = self.extract_suffix_from_filename(h5_file)
        
        print(f"\n{'='*60}")
        print(f"处理H5文件: {h5_file.name} -> {suffix}")
        print(f"{'='*60}")
        
        temp_evreal_dir = None
        
        try:
            # 1. 创建临时EVREAL结构
            temp_evreal_dir = self.create_temp_evreal_structure(suffix)
            
            # 2. 替换事件数据
            event_info = self.replace_event_data(h5_file, temp_evreal_dir)
            
            # 3. 重新生成索引
            self.regenerate_image_event_indices(temp_evreal_dir, event_info["num_events"])
            
            # 4. 运行EVREAL重建
            result = self.run_evreal_reconstruction(temp_evreal_dir, suffix)
            
            result.update({
                "h5_file": h5_file.name,
                "suffix": suffix,
                "event_info": event_info
            })
            
            return result
            
        except Exception as e:
            print(f"❌ 处理{h5_file.name}失败: {e}")
            return {
                "success": False,
                "h5_file": h5_file.name,
                "suffix": suffix,
                "error": str(e)
            }
        
        finally:
            # 清理临时目录
            if temp_evreal_dir:
                self.cleanup_temp_directory(temp_evreal_dir)
    
    def process_all_h5_files(self):
        """处理所有H5文件"""
        h5_files = self.get_all_h5_files()
        
        if not h5_files:
            print("✅ 没有H5文件需要处理")
            return
        
        print(f"\n🚀 开始处理 {len(h5_files)} 个H5文件...")
        
        results = []
        for i, h5_file in enumerate(h5_files, 1):
            print(f"\n处理进度: {i}/{len(h5_files)}")
            result = self.process_single_h5_file(h5_file)
            results.append(result)
        
        # 输出最终结果
        self.print_final_results(results)
    
    def print_final_results(self, results: List[Dict]):
        """打印最终处理结果"""
        print(f"\n{'='*60}")
        print("🎉 处理完成! 结果汇总:")
        print(f"{'='*60}")
        
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        print(f"✅ 成功: {len(successful)}/{len(results)}")
        for result in successful:
            methods_count = len(result.get("methods", []))
            reconstruction_dir = result.get("reconstruction_dir", "")
            print(f"   {result['h5_file']} → {reconstruction_dir.name}/ ({methods_count}种方法)")
        
        if failed:
            print(f"\n❌ 失败: {len(failed)}/{len(results)}")
            for result in failed:
                error_msg = result.get("error", "未知错误")
                print(f"   {result['h5_file']}: {error_msg}")
        
        # 统计重建图像
        if successful:
            print(f"\n📊 重建图像统计:")
            total_images = 0
            for result in successful:
                if "reconstruction_dir" in result and result["reconstruction_dir"]:
                    reconstruction_dir = Path(result["reconstruction_dir"])
                    if reconstruction_dir.exists():
                        png_count = len(list(reconstruction_dir.rglob("*.png")))
                        method_dirs = [d for d in reconstruction_dir.iterdir() if d.is_dir()]
                        print(f"   {result['suffix']}: {png_count} 张图像, {len(method_dirs)} 种方法")
                        total_images += png_count
            
            print(f"\n📈 总计: {total_images} 张重建图像")


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("❌ 错误: 请指定数据集名称")
        print("使用方法: python process_additional_h5_files.py <dataset_name>")
        print("示例: python process_additional_h5_files.py lego2")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    print(f"🔥 {dataset_name} 所有H5文件重建脚本")
    print("=" * 60)
    
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"当前conda环境: {conda_env}")
    if conda_env != 'Umain2':
        print("⚠️  警告: 未在Umain2环境中运行!")
        print("请使用: source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2")
    
    try:
        # 创建处理器并运行
        processor = AllH5Processor(dataset_name)
        processor.process_all_h5_files()
        
    except Exception as e:
        print(f"❌ 脚本执行失败: {e}")
        sys.exit(1)
    
    print(f"\n🎯 {dataset_name} 所有H5处理完成!")


if __name__ == "__main__":
    main()