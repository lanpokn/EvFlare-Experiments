#!/usr/bin/env python3
"""
DVS-Voltmeter仿真器封装模块
创建模块化接口，支持自定义参数

Author: Claude Code Assistant  
Date: 2025-09-17
"""

import os
import sys
import numpy as np
import cv2
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import subprocess

# 添加路径以导入架构模块
sys.path.append('..')
sys.path.append('.')
try:
    from pipeline_architecture import DataFormat, DVSConfig
except ImportError:
    sys.path.append('/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction')
    from pipeline_architecture import DataFormat, DVSConfig

# 添加DVS-Voltmeter路径
DVS_VOLTMETER_PATH = Path(__file__).parent.parent / "DVS-Voltmeter-main"
sys.path.append(str(DVS_VOLTMETER_PATH))

@dataclass
class DVSSimulatorConfig:
    """DVS仿真器配置"""
    dvs_config: DVSConfig = None
    input_dir: Path = Path("temp/dvs_input")
    output_dir: Path = Path("datasets/lego/events_dvs")  # 直接输出到数据集目录
    dvs_voltmeter_path: Path = DVS_VOLTMETER_PATH
    cleanup_temp: bool = True  # 清理临时文件
    
    def __post_init__(self):
        if self.dvs_config is None:
            self.dvs_config = DVSConfig()

class DVSSimulatorWrapper:
    """DVS-Voltmeter仿真器包装器"""
    
    def __init__(self, config: DVSSimulatorConfig):
        self.config = config
        self.dvs_voltmeter_path = config.dvs_voltmeter_path
        
        # 验证DVS-Voltmeter存在
        if not self.dvs_voltmeter_path.exists():
            raise FileNotFoundError(f"DVS-Voltmeter路径不存在: {self.dvs_voltmeter_path}")
            
        self.main_script = self.dvs_voltmeter_path / "main.py"
        if not self.main_script.exists():
            raise FileNotFoundError(f"DVS-Voltmeter主脚本不存在: {self.main_script}")
    
    def validate_input(self, image_sequence: DataFormat.ImageSequence) -> bool:
        """验证输入数据"""
        if not image_sequence.info_file.exists():
            print(f"错误：info.txt文件不存在: {image_sequence.info_file}")
            return False
            
        if len(image_sequence.image_paths) == 0:
            print("错误：图像序列为空")
            return False
            
        # 检查所有图像文件是否存在
        missing_files = [p for p in image_sequence.image_paths if not p.exists()]
        if missing_files:
            print(f"错误：缺少图像文件: {missing_files[:5]}...")  # 只显示前5个
            return False
            
        print(f"输入验证通过：{len(image_sequence.image_paths)} 个图像文件")
        return True
        
    def _prepare_dvs_config(self) -> None:
        """准备DVS配置文件"""
        # 读取原始配置
        config_file = self.dvs_voltmeter_path / "src" / "config.py"
        with open(config_file, 'r') as f:
            content = f.read()
            
        # 创建临时配置文件
        temp_config = self.config.output_dir / "temp_config.py"
        
        # 修改配置参数
        new_content = content.replace(
            f"__C.SENSOR.CAMERA_TYPE = 'DVS346'",
            f"__C.SENSOR.CAMERA_TYPE = '{self.config.dvs_config.camera_type}'"
        )
        
        # 修改K参数
        k_params_str = str(self.config.dvs_config.model_params)
        new_content = new_content.replace(
            f"__C.SENSOR.K = [6.517776598640289, 20, 0.0001, 1e-7, 5e-9, 1e-05]",
            f"__C.SENSOR.K = {k_params_str}"
        )
        
        # 修改路径 - 使用绝对路径
        input_path = str(self.config.input_dir.resolve())
        output_path = str(self.config.output_dir.resolve())
        
        new_content = new_content.replace(
            "__C.DIR.IN_PATH = '/tmp/light_source_events_cnb_5ygj/'",
            f"__C.DIR.IN_PATH = '{input_path}'"
        )
        new_content = new_content.replace(
            "__C.DIR.OUT_PATH = '/tmp/light_source_events_cnb_5ygj/'", 
            f"__C.DIR.OUT_PATH = '{output_path}'"
        )
        
        with open(temp_config, 'w') as f:
            f.write(new_content)
            
        return temp_config
        
    def _prepare_input_structure(self, image_sequence: DataFormat.ImageSequence) -> Path:
        """准备DVS-Voltmeter所需的输入目录结构"""
        # 创建子目录结构：input_dir/sequence_name/
        sequence_name = "lego_sequence"
        sequence_dir = self.config.input_dir / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制图像文件到序列目录
        for i, img_path in enumerate(image_sequence.image_paths):
            dest_path = sequence_dir / f"{i:03d}.png"
            if not dest_path.exists():
                shutil.copy(img_path, dest_path)
        
        # 创建新的info.txt，路径相对于DVS工作目录
        info_file = sequence_dir / "info.txt"
        with open(info_file, 'w') as f:
            for i, timestamp in enumerate(image_sequence.timestamps_us):
                # 相对于DVS-Voltmeter工作目录的路径
                relative_path = f"../temp/dvs_input/lego_sequence/{i:03d}.png"
                f.write(f"{relative_path} {timestamp}\n")
                
        return sequence_dir
        
    def _run_dvs_simulation(self, image_sequence: DataFormat.ImageSequence) -> bool:
        """运行DVS仿真"""
        print("开始DVS事件仿真...")
        
        # 创建输出目录
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备输入目录结构
        sequence_dir = self._prepare_input_structure(image_sequence)
        print(f"准备输入序列目录: {sequence_dir}")
        
        # 准备临时配置
        temp_config = self._prepare_dvs_config()
        
        try:
            # 备份原始配置
            original_config = self.dvs_voltmeter_path / "src" / "config.py"
            backup_config = self.dvs_voltmeter_path / "src" / "config.py.backup"
            shutil.copy(original_config, backup_config)
            
            # 使用临时配置
            shutil.copy(temp_config, original_config)
            
            # 运行DVS仿真器
            cmd = [
                "python", str(self.main_script),
                "--camera_type", self.config.dvs_config.camera_type,
                "--model_para"] + [str(p) for p in self.config.dvs_config.model_params] + [
                "--input_dir", str(self.config.input_dir),
                "--output_dir", str(self.config.output_dir)
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            
            # 切换到DVS-Voltmeter目录执行
            result = subprocess.run(
                cmd, 
                cwd=self.dvs_voltmeter_path,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                print(f"DVS仿真失败:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return False
                
            print("DVS仿真完成")
            print(f"stdout: {result.stdout}")
            return True
            
        except subprocess.TimeoutExpired:
            print("DVS仿真超时")
            return False
        except Exception as e:
            print(f"DVS仿真出错: {e}")
            return False
        finally:
            # 恢复原始配置
            try:
                if backup_config.exists():
                    shutil.copy(backup_config, original_config)
                    backup_config.unlink()  # 删除备份
            except Exception as e:
                print(f"恢复配置文件时出错: {e}")
                
            # 清理临时配置
            if temp_config.exists():
                temp_config.unlink()
                
    def _load_simulation_results(self) -> Optional[DataFormat.DVSEvents]:
        """加载仿真结果"""
        # 查找输出的事件文件，DVS会在子目录中生成
        event_files = list(self.config.output_dir.glob("**/*.txt"))
        
        if len(event_files) == 0:
            print("错误：没有找到事件输出文件")
            print(f"搜索目录: {self.config.output_dir}")
            # 列出所有文件以便调试
            all_files = list(self.config.output_dir.glob("**/*"))
            print(f"找到的所有文件: {all_files}")
            return None
            
        # 使用第一个找到的txt文件
        event_file = event_files[0]
        print(f"加载事件文件: {event_file}")
        
        try:
            # 加载事件数据：格式为 [x, y, timestamp_us, polarity]
            events = np.loadtxt(event_file)
            
            if events.size == 0:
                print("警告：事件文件为空")
                return None
                
            if events.ndim == 1:
                events = events.reshape(1, -1)
                
            print(f"加载了 {events.shape[0]} 个事件")
            print(f"时间范围: {events[:, 2].min():.0f} - {events[:, 2].max():.0f} μs")
            print(f"ON事件: {np.sum(events[:, 3] == 1)}, OFF事件: {np.sum(events[:, 3] == 0)}")
            
            # 创建元数据
            metadata = {
                "camera_type": self.config.dvs_config.camera_type,
                "resolution": self.config.dvs_config.resolution,
                "model_params": self.config.dvs_config.model_params,
                "dt_us": self.config.dvs_config.dt_us,
                "num_events": events.shape[0],
                "time_range_us": (float(events[:, 2].min()), float(events[:, 2].max()))
            }
            
            return DataFormat.DVSEvents(
                events=events,
                output_file=event_file,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"加载事件文件时出错: {e}")
            return None
    
    def simulate(self, image_sequence: DataFormat.ImageSequence) -> Optional[DataFormat.DVSEvents]:
        """执行完整的DVS仿真流程"""
        print("=" * 50)
        print("开始DVS事件相机仿真")
        print("=" * 50)
        
        # 验证输入
        if not self.validate_input(image_sequence):
            return None
            
        # 运行仿真
        if not self._run_dvs_simulation(image_sequence):
            return None
            
        # 加载结果
        result = self._load_simulation_results()
        
        if result:
            # 将事件文件复制到数据集目录并重命名
            dataset_events_file = self.config.output_dir / "lego_train_events.txt"
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            import shutil
            shutil.copy(result.output_file, dataset_events_file)
            
            # 更新结果中的输出文件路径
            result.output_file = dataset_events_file
            result.metadata['final_output_file'] = str(dataset_events_file)
            
            print("DVS仿真成功完成！")
            print(f"最终输出文件: {dataset_events_file}")
            print(f"事件数量: {result.metadata['num_events']}")
            
            # 清理临时文件
            if self.config.cleanup_temp:
                temp_dir = Path("temp")
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                    print("已清理临时文件")
        else:
            print("DVS仿真失败")
            
        return result
        
    def get_simulation_summary(self, result: DataFormat.DVSEvents) -> Dict:
        """获取仿真结果摘要"""
        if result is None:
            return {}
            
        events = result.events
        
        return {
            "num_events": len(events),
            "num_on_events": int(np.sum(events[:, 3] == 1)),
            "num_off_events": int(np.sum(events[:, 3] == 0)),
            "time_range_us": (float(events[:, 2].min()), float(events[:, 2].max())),
            "duration_ms": (events[:, 2].max() - events[:, 2].min()) / 1000,
            "spatial_range": {
                "x_min": int(events[:, 0].min()),
                "x_max": int(events[:, 0].max()),
                "y_min": int(events[:, 1].min()),
                "y_max": int(events[:, 1].max())
            },
            "event_rate_hz": len(events) / ((events[:, 2].max() - events[:, 2].min()) / 1e6),
            "output_file": str(result.output_file),
            "metadata": result.metadata
        }

def main():
    """测试函数"""
    # 首先需要有预处理的图像序列
    from image_preprocessor import ImagePreprocessor, PreprocessConfig
    
    # 创建图像预处理器
    preprocess_config = PreprocessConfig()
    preprocessor = ImagePreprocessor(preprocess_config)
    
    # 执行预处理
    image_sequence = preprocessor.process()
    if image_sequence is None:
        print("图像预处理失败")
        return
        
    # 创建DVS仿真器
    dvs_config = DVSSimulatorConfig()
    simulator = DVSSimulatorWrapper(dvs_config)
    
    # 执行仿真
    result = simulator.simulate(image_sequence)
    
    if result:
        summary = simulator.get_simulation_summary(result)
        print("\nDVS仿真结果摘要：")
        for key, value in summary.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()