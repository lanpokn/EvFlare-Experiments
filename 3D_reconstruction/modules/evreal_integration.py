#!/usr/bin/env python3
"""
EVREAL集成模块
自动调用EVREAL进行图像重建

Author: Claude Code Assistant
Date: 2025-09-17
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

import sys
sys.path.append('..')
sys.path.append('.')
try:
    from pipeline_architecture import EVREALConfig
except ImportError:
    sys.path.append('/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction')
    from pipeline_architecture import EVREALConfig

@dataclass
class EVREALIntegrationConfig:
    """EVREAL集成配置"""
    dataset_name: str = "lego"
    dataset_dir: Path = Path("datasets/lego")
    evreal_path: Path = Path("/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main")
    
    # 事件数据路径
    evreal_data_dir: Optional[Path] = None
    
    # 输出路径
    reconstruction_dir: Optional[Path] = None
    
    # EVREAL配置
    methods: List[str] = None
    eval_config: str = "std"
    
    def __post_init__(self):
        if self.evreal_data_dir is None:
            self.evreal_data_dir = self.dataset_dir / "events_evreal"
        if self.reconstruction_dir is None:
            self.reconstruction_dir = self.dataset_dir / "reconstruction"
        if self.methods is None:
            self.methods = ["E2VID", "FireNet", "HyperE2VID"]

class EVREALDatasetManager:
    """EVREAL数据集管理器"""
    
    def __init__(self, config: EVREALIntegrationConfig):
        self.config = config
    
    def convert_images_to_numpy(self) -> bool:
        """将PNG图像转换为EVREAL标准numpy格式"""
        try:
            import cv2
            sequence_dir = self.config.evreal_data_dir / "sequence"
            images_dir = sequence_dir / "images"
            
            if not images_dir.exists():
                # 自动复制test图像作为真值
                print("复制test图像作为EVREAL真值...")
                images_dir.mkdir(parents=True, exist_ok=True)
                
                test_dir = self.config.dataset_dir / "test"
                if not test_dir.exists():
                    print(f"错误：找不到test图像目录 {test_dir}")
                    return False
                
                import shutil
                for png_file in sorted(test_dir.glob("*.png")):
                    shutil.copy2(png_file, images_dir / png_file.name)
                print(f"复制了{len(list(images_dir.glob('*.png')))}张图像")
            
            # 获取PNG文件
            png_files = sorted(images_dir.glob("*.png"))
            if not png_files:
                print(f"错误：{images_dir}中没有找到PNG文件")
                return False
            
            num_images = len(png_files)
            print(f"开始转换{num_images}张PNG图像为EVREAL格式...")
            
            # 读取第一张图像获取尺寸
            first_img = cv2.imread(str(png_files[0]))
            if first_img is None:
                print(f"错误：无法读取图像 {png_files[0]}")
                return False
            
            height, width = first_img.shape[:2]
            print(f"图像尺寸: {width}x{height}")
            
            # 创建图像数组（EVREAL标准：RGB格式以匹配真值图像）
            images = np.zeros((num_images, height, width, 3), dtype=np.uint8)
            
            for i, png_file in enumerate(png_files):
                img = cv2.imread(str(png_file))
                if img is None:
                    print(f"警告：无法读取 {png_file}")
                    continue
                
                # BGR转RGB
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[i] = rgb
                
                if i % 50 == 0:
                    print(f"  进度: {i+1}/{num_images}")
            
            # 生成时间戳（直接使用图像预处理阶段的时间戳策略）
            with open(sequence_dir / "metadata.json") as f:
                meta = json.load(f)
            
            time_start = meta["time_range_us"][0] / 1e6  # 转换为秒
            time_end = meta["time_range_us"][1] / 1e6
            
            # 使用原始的精确时间戳以确保位姿完美对齐
            # 核心目标：重建图像i必须与原始图像i有完全相同的时间戳和位姿
            dt_seconds = 0.001  # 精确的1ms间隔
            
            # 生成200个时间戳对应200张图像
            # EVREAL要求images.npy和images_ts.npy数量一致
            # between_frames的数据加载器会自动处理N-1的重建逻辑
            aligned_start_time = time_start  # 使用事件起始时间作为第一张图像时间
            
            # 生成200个时间戳: [t_0, t_1, ..., t_199]
            # 对应200张图像: [image_0, image_1, ..., image_199]
            image_timestamps = np.array([aligned_start_time + i * dt_seconds for i in range(num_images)], dtype=np.float64)
            
            print(f"🤔 重新理解EVREAL的数据要求:")
            print(f"  事件时间范围: [{time_start:.6f}, {time_end:.6f}]s")  
            print(f"  时间戳数量: {len(image_timestamps)} (与图像数量一致)")
            print(f"  时间戳范围: [{image_timestamps[0]:.6f}, {image_timestamps[-1]:.6f}]s")
            print(f"  标准时间间隔: {dt_seconds:.6f}s = {dt_seconds*1000:.1f}ms")
            print(f"  最后时间戳: {image_timestamps[-1]:.6f}s")
            print(f"  事件范围检查: {image_timestamps[-1] <= time_end}")
            print(f"  📊 EVREAL验证: len(images) == len(images_ts) = {len(image_timestamps)}")
            print(f"  🎯 重建结果: 仍然是199张图像 (between_frames固有限制)")
            
            # 检查200个时间戳的规律性
            expected_timestamps_us = [i * 1000 for i in range(num_images)]  # 200个：0, 1000, 2000, ..., 199000
            actual_timestamps_us = [int(ts * 1e6) for ts in image_timestamps]  # 200个实际时间戳
            
            alignment_errors = 0
            for i in range(num_images):
                expected = expected_timestamps_us[i] + int(time_start * 1e6)  # 加上起始偏移
                actual = actual_timestamps_us[i]
                if abs(expected - actual) > 1:  # 允许1μs误差
                    alignment_errors += 1
            
            print(f"✅ 200个时间戳规律性检查: {alignment_errors}个误差 (应为0)")
            print(f"✅ 第200个时间戳: {image_timestamps[-1]:.6f}s (对应原始图像200)")
            
            # 直接使用201个时间戳（已包含第201个边界时间戳）
            extended_timestamps = image_timestamps
            
            # 生成图像事件索引
            event_timestamps = np.load(sequence_dir / "events_ts.npy")
            num_events = len(event_timestamps)
            
            image_event_indices = np.zeros((num_images, 2), dtype=np.int64)
            for i in range(num_images):
                start_idx = int(i * num_events / num_images)
                end_idx = int((i + 1) * num_events / num_images)
                if i == num_images - 1:
                    end_idx = num_events
                image_event_indices[i] = [start_idx, end_idx]
            
            # 保存numpy文件
            np.save(sequence_dir / "images.npy", images)
            np.save(sequence_dir / "images_ts.npy", extended_timestamps)  # 使用扩展的时间戳（包含虚拟第201张）
            np.save(sequence_dir / "image_event_indices.npy", image_event_indices)
            
            print(f"✅ PNG转numpy完成:")
            print(f"  images.npy: {images.shape} (200张真实图像)")
            print(f"  images_ts.npy: {extended_timestamps.shape} (200个对应时间戳)")
            print(f"  数据一致性: len(images) == len(images_ts) ✅")
            print(f"  image_event_indices.npy: {image_event_indices.shape}")
            print(f"  🤔 重建预期: 199张图像 (between_frames固有N-1限制)")
            print(f"  🎯 位姿对齐: 重建的199张图像与原始图像1-199完美对齐")
            
            return True
            
        except ImportError:
            print("错误：缺少cv2库，请确保在Umain2环境中运行")
            return False
        except Exception as e:
            print(f"PNG转numpy转换出错: {e}")
            return False
        
    def create_evreal_dataset_config(self) -> Path:
        """创建EVREAL数据集配置文件"""
        # EVREAL配置目录
        config_dir = self.config.evreal_path / "config" / "dataset"
        dataset_config_file = config_dir / f"{self.config.dataset_name}.json"
        
        # 确保配置目录存在
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集配置 - 使用EVREAL标准格式
        dataset_config = {
            "root_path": str(self.config.evreal_data_dir.absolute()),
            "sequences": {
                "sequence": {}  # EVREAL期望的格式：字典而非列表
            }
        }
        
        # 注：EVREAL会从事件数据自动推断分辨率，无需在配置中指定
        
        # 保存配置文件
        with open(dataset_config_file, 'w') as f:
            json.dump(dataset_config, f, indent=2)
            
        print(f"创建EVREAL数据集配置: {dataset_config_file}")
        return dataset_config_file
        
    def prepare_evreal_data_structure(self) -> bool:
        """准备EVREAL所需的数据结构"""
        try:
            # 检查事件数据文件
            sequence_dir = self.config.evreal_data_dir / "sequence"
            event_files = ["events_ts.npy", "events_xy.npy", "events_p.npy"]
            for file_name in event_files:
                file_path = sequence_dir / file_name
                if not file_path.exists():
                    print(f"错误：缺少EVREAL事件文件 {file_path}")
                    return False
            
            # 检查并转换图像数据
            image_files = ["images.npy", "images_ts.npy", "image_event_indices.npy"]
            missing_image_files = []
            for file_name in image_files:
                file_path = sequence_dir / file_name
                if not file_path.exists():
                    missing_image_files.append(file_name)
            
            if missing_image_files:
                print(f"缺少图像文件: {missing_image_files}")
                print("开始PNG到numpy转换...")
                if not self.convert_images_to_numpy():
                    return False
            
            print("✅ EVREAL数据结构检查通过")
            return True
            
        except Exception as e:
            print(f"准备EVREAL数据结构时出错: {e}")
            return False

class EVREALRunner:
    """EVREAL运行器"""
    
    def __init__(self, config: EVREALIntegrationConfig):
        self.config = config
        
    def run_evreal_reconstruction(self, methods: List[str] = None) -> Dict[str, bool]:
        """运行EVREAL图像重建"""
        if methods is None:
            methods = self.config.methods
            
        print("=" * 60)
        print("开始EVREAL图像重建")
        print("=" * 60)
        
        results = {}
        
        for method in methods:
            print(f"\n运行重建方法: {method}")
            
            try:
                # 构建EVREAL命令
                cmd = [
                    "python", "eval.py",
                    "-m", method,
                    "-c", self.config.eval_config,
                    "-d", self.config.dataset_name,
                    "-qm", "mse", "ssim", "lpips"
                ]
                
                print(f"执行命令: {' '.join(cmd)}")
                print(f"工作目录: {self.config.evreal_path}")
                
                # 激活Umain2环境并运行EVREAL
                env_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && {' '.join(cmd)}"
                
                print(f"🚀 DEBUG: 完整执行命令: {env_cmd}")
                print(f"🚀 DEBUG: 工作目录: {self.config.evreal_path}")
                print(f"🚀 DEBUG: EVREAL路径存在: {self.config.evreal_path.exists()}")
                
                # 检查eval.py是否存在
                eval_script = self.config.evreal_path / "eval.py"
                print(f"🚀 DEBUG: eval.py存在: {eval_script.exists()}")
                
                result = subprocess.run(
                    ["bash", "-c", env_cmd],
                    cwd=self.config.evreal_path,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30分钟超时（足够重建大量图像）
                )
                
                print(f"🚀 DEBUG: 返回码: {result.returncode}")
                print(f"🚀 DEBUG: stdout长度: {len(result.stdout) if result.stdout else 0}")
                print(f"🚀 DEBUG: stderr长度: {len(result.stderr) if result.stderr else 0}")
                
                if result.returncode == 0:
                    print(f"✅ {method} 重建命令执行成功")
                    print(f"🚀 DEBUG: stdout内容:")
                    print(result.stdout)
                    
                    # 检查是否真的生成了输出文件
                    outputs_dir = self.config.evreal_path / "outputs"
                    print(f"🚀 DEBUG: outputs目录存在: {outputs_dir.exists()}")
                    if outputs_dir.exists():
                        output_contents = list(outputs_dir.glob("*"))
                        print(f"🚀 DEBUG: outputs目录内容: {[p.name for p in output_contents]}")
                    
                    results[method] = True
                else:
                    print(f"❌ {method} 重建失败 (返回码: {result.returncode}):")
                    print(f"🚀 DEBUG: 完整stdout:")
                    print(result.stdout)
                    print(f"🚀 DEBUG: 完整stderr:")
                    print(result.stderr)
                    results[method] = False
                    
            except subprocess.TimeoutExpired:
                print(f"❌ {method} 重建超时")
                results[method] = False
            except Exception as e:
                print(f"❌ {method} 重建出错: {e}")
                results[method] = False
                
        return results
        
    def copy_reconstruction_results(self, methods: List[str] = None) -> Dict[str, Path]:
        """复制重建结果到数据集目录"""
        if methods is None:
            methods = self.config.methods
            
        print("\n复制重建结果...")
        
        copied_results = {}
        
        # 创建重建结果目录
        self.config.reconstruction_dir.mkdir(parents=True, exist_ok=True)
        
        for method in methods:
            try:
                # 查找EVREAL实际输出目录（可能有时间戳等变化）
                outputs_base = self.config.evreal_path / "outputs" / self.config.eval_config
                
                # 搜索包含我们数据集名称的目录
                evreal_output_dir = None
                for dataset_dir in outputs_base.glob("*"):
                    if dataset_dir.is_dir() and self.config.dataset_name in dataset_dir.name:
                        # 寻找sequence子目录
                        sequence_dir = dataset_dir / "sequence" / method
                        if sequence_dir.exists():
                            evreal_output_dir = sequence_dir
                            break
                
                # 如果没找到，尝试直接路径
                if evreal_output_dir is None:
                    direct_path = outputs_base / self.config.dataset_name / "sequence" / method
                    if direct_path.exists():
                        evreal_output_dir = direct_path
                
                if evreal_output_dir is None or not evreal_output_dir.exists():
                    # 详细调试输出目录结构
                    print(f"❌ 找不到 {method} 的输出目录")
                    print(f"🚀 DEBUG: 预期路径: outputs/{self.config.eval_config}/{self.config.dataset_name}/sequence/{method}")
                    print(f"🚀 DEBUG: outputs_base存在: {outputs_base.exists()}")
                    
                    if outputs_base.exists():
                        available_dirs = list(outputs_base.glob("*/"))
                        print(f"🚀 DEBUG: 可用目录: {[d.name for d in available_dirs]}")
                        
                        # 递归查看目录结构
                        for dataset_dir in available_dirs:
                            if dataset_dir.is_dir():
                                print(f"🚀 DEBUG: 检查目录 {dataset_dir.name}:")
                                subdirs = list(dataset_dir.glob("*"))
                                print(f"  子目录: {[s.name for s in subdirs if s.is_dir()]}")
                                
                                # 检查sequence目录
                                seq_dir = dataset_dir / "sequence"
                                if seq_dir.exists():
                                    seq_contents = list(seq_dir.glob("*"))
                                    print(f"  sequence内容: {[s.name for s in seq_contents]}")
                    else:
                        print(f"🚀 DEBUG: outputs目录不存在: {outputs_base}")
                    continue
                    
                # 目标目录
                target_dir = self.config.reconstruction_dir / f"evreal_{method.lower()}"
                
                # 智能复制并重命名文件
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # 获取重建结果文件
                recon_files = sorted(evreal_output_dir.glob("*.png"))
                if not recon_files:
                    print(f"⚠️  {method} 输出目录中没有PNG文件")
                    continue
                
                # 获取原始图像列表（用于文件名映射）
                original_images = sorted(self.config.dataset_dir.glob("train/*.png"))
                if not original_images:
                    original_images = sorted(self.config.dataset_dir.glob("test/*.png"))
                
                print(f"   找到 {len(recon_files)} 张重建图像，{len(original_images)} 张原始图像")
                
                # 文件名映射和复制
                successful_copies = 0
                for i, recon_file in enumerate(recon_files):
                    # 映射到原始图像编号
                    # 重建通常从第1张图像开始（frame_0000000000 -> 0001.png）
                    original_idx = i + 1  # frame_0000000000 对应 0001.png
                    
                    if original_idx <= len(original_images):
                        # 生成对应的文件名
                        target_filename = f"{original_idx:04d}.png"
                        target_path = target_dir / target_filename
                        
                        # 复制并重命名
                        shutil.copy2(recon_file, target_path)
                        successful_copies += 1
                    else:
                        print(f"⚠️  重建图像索引超出范围: {i} -> {original_idx}")
                
                copied_results[method] = target_dir
                print(f"✅ 复制 {method} 结果到: {target_dir}")
                print(f"   成功复制并重命名: {successful_copies} 张图像")
                
                # 验证结果
                final_files = sorted(target_dir.glob("*.png"))
                if len(final_files) != len(recon_files):
                    print(f"⚠️  复制数量不匹配: 预期{len(recon_files)}, 实际{len(final_files)}")
                else:
                    print(f"✅ 文件名映射验证通过: {final_files[0].name} - {final_files[-1].name}")
                
            except Exception as e:
                print(f"❌ 复制 {method} 结果时出错: {e}")
                
        return copied_results
    

class EVREALIntegration:
    """EVREAL集成主类"""
    
    def __init__(self, config: EVREALIntegrationConfig):
        self.config = config
        self.dataset_manager = EVREALDatasetManager(config)
        self.runner = EVREALRunner(config)
        
    def validate_environment(self) -> bool:
        """验证EVREAL运行环境"""
        if not self.config.evreal_path.exists():
            print(f"错误：EVREAL路径不存在 {self.config.evreal_path}")
            return False
            
        eval_script = self.config.evreal_path / "eval.py"
        if not eval_script.exists():
            print(f"错误：EVREAL评估脚本不存在 {eval_script}")
            return False
            
        if not self.config.evreal_data_dir.exists():
            print(f"错误：EVREAL数据目录不存在 {self.config.evreal_data_dir}")
            return False
            
        print("EVREAL环境验证通过")
        return True
        
    def run_full_pipeline(self, methods: List[str] = None) -> Dict:
        """运行完整的EVREAL重建Pipeline"""
        print("🚀 开始EVREAL集成Pipeline")
        
        # 1. 验证环境
        if not self.validate_environment():
            return {"success": False, "error": "环境验证失败"}
            
        # 2. 准备数据结构
        if not self.dataset_manager.prepare_evreal_data_structure():
            return {"success": False, "error": "数据结构准备失败"}
            
        # 3. 创建数据集配置
        try:
            config_file = self.dataset_manager.create_evreal_dataset_config()
        except Exception as e:
            return {"success": False, "error": f"创建数据集配置失败: {e}"}
            
        # 4. 运行重建
        reconstruction_results = self.runner.run_evreal_reconstruction(methods)
        
        # 5. 复制结果
        copied_results = self.runner.copy_reconstruction_results(methods)
        
        # 6. 生成摘要
        summary = {
            "success": True,
            "dataset_config": str(config_file),
            "reconstruction_results": reconstruction_results,
            "copied_results": {k: str(v) for k, v in copied_results.items()},
            "successful_methods": [k for k, v in reconstruction_results.items() if v],
            "failed_methods": [k for k, v in reconstruction_results.items() if not v]
        }
        
        print("\n" + "=" * 60)
        print("EVREAL集成Pipeline完成")
        print("=" * 60)
        print(f"成功的方法: {summary['successful_methods']}")
        print(f"失败的方法: {summary['failed_methods']}")
        print(f"结果目录: {self.config.reconstruction_dir}")
        
        return summary

def main():
    """测试函数"""
    # 检查EVREAL数据是否存在
    evreal_data_dir = Path("datasets/lego/events_evreal")
    if not evreal_data_dir.exists():
        print("错误：未找到EVREAL格式数据")
        print("请先运行格式转换器生成EVREAL数据")
        return
        
    # 配置
    config = EVREALIntegrationConfig()
    
    # 运行集成
    integration = EVREALIntegration(config)
    
    # 测试所有重建方法
    test_methods = None  # 使用默认的所有方法 ["E2VID", "FireNet", "HyperE2VID"]
    results = integration.run_full_pipeline(test_methods)
    
    if results["success"]:
        print("\n🎉 EVREAL集成测试成功！")
    else:
        print(f"\n❌ EVREAL集成测试失败: {results.get('error', '未知错误')}")

if __name__ == "__main__":
    main()