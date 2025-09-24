#!/usr/bin/env python3
"""
H5事件数据批量重建脚本
批量处理events_h5目录中的所有H5文件，为每个生成对应的重建图像

功能：
1. 扫描events_h5目录中的所有H5文件（除backup文件）
2. 对每个H5文件进行H5→EVREAL格式转换  
3. 调用EVREAL生成重建图像
4. 确保时间戳对齐和200:200完美对应
5. 根据H5文件名创建独立的重建输出目录

Author: Claude Code Assistant  
Date: 2025-09-24
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import h5py
import json

# 导入现有模块
sys.path.append('.')
try:
    from modules.format_converter import EVREALToH5Converter, ConversionConfig
    from modules.evreal_integration import EVREALIntegrationConfig, EVREALIntegration  
    from pipeline_architecture import DataFormat, FormatConverter
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

class H5BatchReconstructor:
    """H5批量重建器"""
    
    def __init__(self, h5_dir: Path, dataset_dir: Path = Path("datasets/lego")):
        self.h5_dir = Path(h5_dir)
        self.dataset_dir = Path(dataset_dir) 
        self.temp_dir = Path("temp/batch_reconstruction")
        
        # 创建临时工作目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def scan_h5_files(self) -> List[Path]:
        """扫描H5目录，找到所有需要处理的H5文件"""
        if not self.h5_dir.exists():
            print(f"错误: H5目录不存在 {self.h5_dir}")
            return []
            
        h5_files = []
        for h5_file in self.h5_dir.glob("*.h5"):
            # 跳过backup文件
            if "backup" in h5_file.name.lower() or "wrong" in h5_file.name.lower():
                print(f"跳过backup文件: {h5_file.name}")
                continue
            h5_files.append(h5_file)
            
        print(f"找到 {len(h5_files)} 个H5文件需要处理:")
        for f in h5_files:
            print(f"  - {f.name}")
            
        return h5_files
    
    def load_h5_events(self, h5_file: Path) -> Optional[DataFormat.H5Events]:
        """加载H5事件文件"""
        try:
            with h5py.File(h5_file, 'r') as f:
                events_t = f['events/t'][:]  # 时间戳（微秒）
                events_x = f['events/x'][:]  # x坐标
                events_y = f['events/y'][:]  # y坐标  
                events_p = f['events/p'][:]  # 极性
                
                # 读取元数据
                metadata = dict(f['events'].attrs)
                
            print(f"从 {h5_file.name} 加载 {len(events_t):,} 个事件")
            print(f"  时间范围: {events_t.min():.0f} - {events_t.max():.0f} μs")
            print(f"  空间范围: x[{events_x.min():.0f}, {events_x.max():.0f}] y[{events_y.min():.0f}, {events_y.max():.0f}]")
            
            return DataFormat.H5Events(
                events_t=events_t,
                events_x=events_x, 
                events_y=events_y,
                events_p=events_p,
                h5_file=h5_file,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"加载H5文件失败 {h5_file}: {e}")
            return None
    
    def h5_to_evreal_format(self, h5_events: DataFormat.H5Events, output_dir: Path) -> bool:
        """H5格式转EVREAL格式 - 复用一键式的成功配置"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            sequence_dir = output_dir / "sequence"
            sequence_dir.mkdir(parents=True, exist_ok=True)
            
            # 转换事件数据格式
            events_ts = h5_events.events_t / 1e6  # 微秒 → 秒
            events_xy = np.column_stack([h5_events.events_x, h5_events.events_y])
            events_p = h5_events.events_p
            
            # 🎯 关键：复用一键式成功的EVREAL配置
            reference_evreal_dir = self.dataset_dir / "events_evreal" / "sequence"
            if not reference_evreal_dir.exists():
                print(f"错误：参考EVREAL目录不存在 {reference_evreal_dir}")
                return False
            
            print(f"复用一键式EVREAL配置: {reference_evreal_dir}")
            
            # 1. 复制所有非事件文件（图像、时间戳、配置等）
            for file_path in reference_evreal_dir.glob("*"):
                if file_path.name not in ["events_ts.npy", "events_xy.npy", "events_p.npy"]:
                    if file_path.is_dir():
                        # 复制目录（如images目录）
                        target_dir = sequence_dir / file_path.name
                        if target_dir.exists():
                            shutil.rmtree(target_dir)
                        shutil.copytree(file_path, target_dir)
                        print(f"  复制目录: {file_path.name}")
                    else:
                        # 复制文件
                        shutil.copy2(file_path, sequence_dir / file_path.name)
                        print(f"  复制文件: {file_path.name}")
            
            # 2. 保存新的事件数据（只替换事件相关文件）
            np.save(sequence_dir / "events_ts.npy", events_ts)
            np.save(sequence_dir / "events_xy.npy", events_xy)
            np.save(sequence_dir / "events_p.npy", events_p)
            
            # 也保存到主目录（EVREAL标准格式）
            np.save(output_dir / "events_ts.npy", events_ts)
            np.save(output_dir / "events_xy.npy", events_xy) 
            np.save(output_dir / "events_p.npy", events_p)
            
            # 3. 更新元数据中的事件信息
            reference_metadata_file = reference_evreal_dir / "metadata.json"
            if reference_metadata_file.exists():
                with open(reference_metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # 只更新事件相关字段
                metadata.update({
                    "num_events": len(events_ts),
                    "time_range_s": (float(events_ts.min()), float(events_ts.max())),
                    "spatial_range": {
                        "x_range": (int(events_xy[:, 0].min()), int(events_xy[:, 0].max())),
                        "y_range": (int(events_xy[:, 1].min()), int(events_xy[:, 1].max()))
                    },
                    "source_h5": str(h5_events.h5_file),
                    "conversion_timestamp": str(np.datetime64('now'))
                })
                
                # 保存更新后的元数据
                with open(sequence_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                with open(output_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            print(f"✅ H5→EVREAL转换完成: {output_dir}")
            print(f"  ✅ 事件数据: events_ts.npy {events_ts.shape}, events_xy.npy {events_xy.shape}, events_p.npy {events_p.shape}")
            print(f"  ✅ 复用一键式配置: 图像、时间戳、配置文件等")
            
            return True
            
        except Exception as e:
            print(f"H5→EVREAL转换失败: {e}")
            return False
    
    def generate_reconstruction_name(self, h5_file: Path) -> str:
        """根据H5文件名生成重建目录名"""
        # 去掉.h5后缀，添加reconstruction前缀
        base_name = h5_file.stem  # 文件名不含后缀
        return f"reconstruction_{base_name}"
    
    def run_evreal_reconstruction(self, evreal_dir: Path, output_dir: Path, h5_name: str) -> bool:
        """调用EVREAL进行重建 - 参考一键式的成功做法"""
        try:
            # EVREAL路径和配置（与一键式完全一致）
            evreal_path = Path("/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main")
            if not evreal_path.exists():
                print(f"错误: EVREAL路径不存在 {evreal_path}")
                return False
            
            # 检查eval.py是否存在
            eval_script = evreal_path / "eval.py"
            if not eval_script.exists():
                print(f"错误: eval.py不存在 {eval_script}")
                return False
            
            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 成功的重建方法（与一键式项目记忆一致）
            methods = ["E2VID", "FireNet", "SPADE-E2VID", "SSL-E2VID"]
            
            # 首先需要创建EVREAL数据集配置
            dataset_name = f"batch_{h5_name}"
            self.create_evreal_dataset_config(evreal_dir, dataset_name, evreal_path)
            
            success_count = 0
            for method in methods:
                print(f"  运行 {method} 重建...")
                
                # EVREAL命令（与一键式完全一致）
                cmd = [
                    "python", "eval.py",
                    "-m", method,
                    "-c", "std",
                    "-d", dataset_name,
                    "-qm", "mse", "ssim", "lpips"
                ]
                
                # 激活Umain2环境并运行EVREAL（与一键式完全一致）
                env_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && {' '.join(cmd)}"
                
                try:
                    result = subprocess.run(
                        ["bash", "-c", env_cmd],
                        cwd=evreal_path,
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30分钟超时（与一键式一致）
                    )
                    
                    if result.returncode == 0:
                        print(f"    ✅ {method}: EVREAL命令执行成功")
                        success_count += 1
                    else:
                        print(f"    ❌ {method}: 重建失败 (返回码: {result.returncode})")
                        if result.stderr:
                            print(f"       stderr: {result.stderr}")
                        if result.stdout:
                            print(f"       stdout: {result.stdout}")
                        
                except subprocess.TimeoutExpired:
                    print(f"    ❌ {method}: 重建超时")
                except Exception as e:
                    print(f"    ❌ {method}: {e}")
            
            # 复制重建结果到我们的输出目录
            if success_count > 0:
                self.copy_evreal_results(evreal_path, output_dir, dataset_name, methods)
            
            print(f"重建完成: {success_count}/{len(methods)} 方法成功")
            return success_count > 0
            
        except Exception as e:
            print(f"EVREAL重建调用失败: {e}")
            return False
    
    def create_evreal_dataset_config(self, evreal_dir: Path, dataset_name: str, evreal_path: Path):
        """创建EVREAL数据集配置 - 复用一键式的成功配置"""
        try:
            # 查找一键式成功的数据集配置 - 修正路径
            config_dir = evreal_path / "config" / "dataset"  # 修正：实际路径
            config_dir.mkdir(parents=True, exist_ok=True)
            reference_config_file = config_dir / "lego.json"
            
            if not reference_config_file.exists():
                print(f"警告：参考配置文件不存在 {reference_config_file}，使用默认配置")
                # 使用默认配置
                config_data = {
                    "path": str(evreal_dir / "sequence"),
                    "load_name": dataset_name,
                    "voxel_method": "between_frames",
                    "num_bins": 5,
                    "clip_dist": 3,
                    "sensor": {
                        "name": "DVS346",
                        "resolution": [480, 640]  # H x W
                    }
                }
            else:
                # 复用一键式的成功配置，只修改路径和名称
                with open(reference_config_file, 'r') as f:
                    config_data = json.load(f)
                
                # 只更新必要字段
                config_data["path"] = str(evreal_dir / "sequence")
                config_data["load_name"] = dataset_name
                print(f"复用一键式配置: {reference_config_file}")
            
            # 创建新的配置文件
            config_file = config_dir / f"{dataset_name}.json"
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            print(f"✅ 创建EVREAL数据集配置: {config_file}")
            
        except Exception as e:
            print(f"创建EVREAL配置失败: {e}")
    
    def copy_evreal_results(self, evreal_path: Path, output_dir: Path, dataset_name: str, methods: List[str]):
        """复制EVREAL重建结果到输出目录"""
        try:
            outputs_dir = evreal_path / "outputs"
            if not outputs_dir.exists():
                print(f"警告: EVREAL输出目录不存在 {outputs_dir}")
                return
            
            # 查找重建结果
            for method in methods:
                method_pattern = f"*{dataset_name}*{method}*"
                method_dirs = list(outputs_dir.glob(method_pattern))
                
                if method_dirs:
                    source_dir = method_dirs[0]  # 取第一个匹配的目录
                    target_dir = output_dir / f"evreal_{method.lower()}"
                    
                    if source_dir.exists():
                        # 复制重建图像
                        target_dir.mkdir(parents=True, exist_ok=True)
                        for png_file in source_dir.glob("*.png"):
                            shutil.copy2(png_file, target_dir / png_file.name)
                        
                        png_count = len(list(target_dir.glob("*.png")))
                        print(f"    📁 {method}: 复制了{png_count}张重建图像到 {target_dir}")
                else:
                    print(f"    ⚠️  {method}: 未找到重建结果")
                    
        except Exception as e:
            print(f"复制EVREAL结果失败: {e}")
    
    def process_single_h5(self, h5_file: Path) -> bool:
        """处理单个H5文件"""
        print(f"\n{'='*60}")
        print(f"🚀 处理H5文件: {h5_file.name}")
        print(f"{'='*60}")
        
        # 1. 加载H5数据
        h5_events = self.load_h5_events(h5_file)
        if h5_events is None:
            return False
        
        # 2. 生成输出目录名
        recon_name = self.generate_reconstruction_name(h5_file)
        
        # 3. 创建临时EVREAL工作目录
        temp_evreal_dir = self.temp_dir / f"evreal_{h5_file.stem}"
        
        # 4. H5→EVREAL格式转换
        print(f"步骤1: H5→EVREAL格式转换")
        if not self.h5_to_evreal_format(h5_events, temp_evreal_dir):
            return False
            
        # 5. 创建重建输出目录
        output_dir = self.dataset_dir / recon_name
        
        # 6. 调用EVREAL重建
        print(f"步骤2: EVREAL重建")
        success = self.run_evreal_reconstruction(temp_evreal_dir, output_dir, h5_file.stem)
        
        # 7. 清理临时文件
        if temp_evreal_dir.exists():
            shutil.rmtree(temp_evreal_dir)
            print(f"清理临时目录: {temp_evreal_dir}")
        
        return success
    
    def run_batch_reconstruction(self) -> Dict[str, bool]:
        """运行批量重建"""
        print("🎯 H5事件数据批量重建")
        print(f"H5目录: {self.h5_dir}")
        print(f"数据集目录: {self.dataset_dir}")
        
        # 扫描H5文件
        h5_files = self.scan_h5_files()
        if not h5_files:
            print("没有找到需要处理的H5文件")
            return {}
        
        # 批量处理
        results = {}
        for i, h5_file in enumerate(h5_files):
            print(f"\n进度: [{i+1}/{len(h5_files)}]")
            success = self.process_single_h5(h5_file)
            results[h5_file.name] = success
        
        # 总结结果
        print(f"\n{'='*60}")
        print("🎉 批量重建完成 - 总结")
        print(f"{'='*60}")
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"总体结果: {success_count}/{total_count} 个文件处理成功")
        
        for h5_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"  {h5_name}: {status}")
            
        return results

def main():
    """主函数"""
    # 配置路径
    dataset_dir = Path("datasets/lego") 
    h5_dir = dataset_dir / "events_h5"
    
    # 检查输入目录
    if not h5_dir.exists():
        print(f"错误: H5目录不存在 {h5_dir}")
        print("请先运行事件数据生成pipeline")
        return False
    
    # 创建批量重建器
    reconstructor = H5BatchReconstructor(h5_dir, dataset_dir)
    
    # 执行批量重建
    results = reconstructor.run_batch_reconstruction()
    
    # 返回执行结果
    return len(results) > 0 and all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)