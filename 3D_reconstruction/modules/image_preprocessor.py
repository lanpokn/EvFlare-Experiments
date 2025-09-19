#!/usr/bin/env python3
"""
图像序列预处理模块
将lego/train图像转换为DVS仿真器所需的输入格式

Author: Claude Code Assistant
Date: 2025-09-17
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil
from dataclasses import dataclass

import sys
sys.path.append('..')
sys.path.append('.')
try:
    from pipeline_architecture import DataFormat, TimeStampConfig, TimestampManager
except ImportError:
    sys.path.append('/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction')
    from pipeline_architecture import DataFormat, TimeStampConfig, TimestampManager

@dataclass 
class PreprocessConfig:
    """预处理配置"""
    input_dir: Path = Path("datasets/lego/train")
    output_dir: Path = Path("temp/dvs_input")
    target_resolution: Optional[Tuple[int, int]] = None  # (width, height), None表示保持原分辨率
    image_format: str = "png"  # 输出图像格式
    timestamp_config: TimeStampConfig = None
    
    def __post_init__(self):
        if self.timestamp_config is None:
            self.timestamp_config = TimeStampConfig()

class ImagePreprocessor:
    """图像序列预处理器"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.timestamp_manager = TimestampManager(config.timestamp_config)
        
    def validate_input_directory(self) -> bool:
        """验证输入目录"""
        if not self.config.input_dir.exists():
            print(f"错误：输入目录不存在 {self.config.input_dir}")
            return False
            
        image_files = self._get_image_files()
        if len(image_files) == 0:
            print(f"错误：在 {self.config.input_dir} 中没有找到图像文件")
            return False
            
        print(f"找到 {len(image_files)} 个图像文件")
        return True
        
    def _get_image_files(self) -> List[Path]:
        """获取所有图像文件，按数字顺序排序"""
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(self.config.input_dir.glob(f'*{ext}'))
            image_files.extend(self.config.input_dir.glob(f'*{ext.upper()}'))
            
        # 按文件名中的数字排序
        def extract_number(path):
            import re
            numbers = re.findall(r'\d+', path.stem)
            return int(numbers[0]) if numbers else 0
            
        image_files.sort(key=extract_number)
        return image_files
        
    def _process_single_image(self, input_path: Path, output_path: Path) -> bool:
        """处理单张图像"""
        try:
            # 读取图像
            image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"警告：无法读取图像 {input_path}")
                return False
                
            # 调整分辨率（如果需要）
            if self.config.target_resolution is not None:
                width, height = self.config.target_resolution
                image = cv2.resize(image, (width, height))
                
            # 保存处理后的图像
            cv2.imwrite(str(output_path), image)
            return True
            
        except Exception as e:
            print(f"错误：处理图像 {input_path} 时发生异常：{e}")
            return False
            
    def _generate_info_file(self, image_files: List[Path], timestamps: List[int]) -> Path:
        """生成DVS仿真器所需的info.txt文件"""
        info_file = self.config.output_dir / "info.txt"
        
        with open(info_file, 'w') as f:
            for img_file, timestamp in zip(image_files, timestamps):
                # DVS-Voltmeter要求的格式：相对路径 时间戳(微秒)
                relative_path = img_file.name
                f.write(f"{relative_path} {timestamp}\n")
                
        return info_file
        
    def _copy_and_rename_images(self, input_files: List[Path]) -> List[Path]:
        """复制并重命名图像文件为连续编号格式"""
        output_files = []
        
        for i, input_file in enumerate(input_files):
            # 生成连续编号的文件名：000.png, 001.png, etc.
            output_name = f"{i:03d}.{self.config.image_format}"
            output_path = self.config.output_dir / output_name
            
            # 处理并保存图像
            if self._process_single_image(input_file, output_path):
                output_files.append(output_path)
            else:
                print(f"跳过文件：{input_file}")
                
        return output_files
        
    def process(self) -> Optional[DataFormat.ImageSequence]:
        """执行图像序列预处理"""
        print("开始图像序列预处理...")
        
        # 验证输入
        if not self.validate_input_directory():
            return None
            
        # 创建输出目录
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取输入图像列表
        input_files = self._get_image_files()
        print(f"处理 {len(input_files)} 个图像文件")
        
        # 生成时间戳
        timestamps = self.timestamp_manager.generate_image_timestamps(len(input_files))
        print(f"时间戳范围：{timestamps[0]} - {timestamps[-1]} μs")
        print(f"时间间隔：{self.config.timestamp_config.dt_us} μs ({self.config.timestamp_config.dt_us/1000:.1f} ms)")
        
        # 复制和重命名图像文件
        output_files = self._copy_and_rename_images(input_files)
        
        if len(output_files) == 0:
            print("错误：没有成功处理任何图像")
            return None
            
        # 生成info.txt文件
        info_file = self._generate_info_file(output_files, timestamps)
        print(f"生成info文件：{info_file}")
        
        # 获取第一张图像的分辨率信息
        first_image = cv2.imread(str(output_files[0]), cv2.IMREAD_GRAYSCALE)
        height, width = first_image.shape
        print(f"图像分辨率：{width}x{height}")
        
        result = DataFormat.ImageSequence(
            image_paths=output_files,
            timestamps_us=timestamps,
            info_file=info_file
        )
        
        print("图像序列预处理完成！")
        print(f"输出目录：{self.config.output_dir}")
        print(f"处理图像数量：{len(output_files)}")
        
        return result
        
    def get_summary(self, result: DataFormat.ImageSequence) -> Dict:
        """获取预处理结果摘要"""
        if result is None:
            return {}
            
        # 读取第一张图像获取分辨率
        first_image = cv2.imread(str(result.image_paths[0]), cv2.IMREAD_GRAYSCALE)
        height, width = first_image.shape
        
        # 计算时间信息
        duration_us = result.timestamps_us[-1] - result.timestamps_us[0]
        duration_ms = duration_us / 1000
        duration_s = duration_us / 1e6
        
        return {
            "num_images": len(result.image_paths),
            "resolution": (width, height),
            "timestamp_range_us": (result.timestamps_us[0], result.timestamps_us[-1]),
            "duration_ms": duration_ms,
            "duration_s": duration_s,
            "frame_rate_fps": len(result.image_paths) / duration_s if duration_s > 0 else 0,
            "output_dir": str(self.config.output_dir),
            "info_file": str(result.info_file)
        }

def main():
    """测试函数"""
    config = PreprocessConfig()
    preprocessor = ImagePreprocessor(config)
    
    result = preprocessor.process()
    if result:
        summary = preprocessor.get_summary(result)
        print("\n预处理结果摘要：")
        for key, value in summary.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()