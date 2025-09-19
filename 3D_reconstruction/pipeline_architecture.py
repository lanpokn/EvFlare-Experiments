#!/usr/bin/env python3
"""
事件相机仿真到图像重建Pipeline架构设计
Author: Claude Code Assistant
Date: 2025-09-17

整体数据流向:
Lego训练图像 → DVS事件仿真 → EVREAL格式转换 → [外部]图像重建 → H5数据存储

模块化设计原则:
1. 每个模块职责单一，接口清晰
2. 数据格式统一，时间戳对齐
3. 支持模块独立测试和运行
4. 图像重建模块可选择性分离
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum

class PipelineStage(Enum):
    """Pipeline执行阶段"""
    IMAGE_PREPROCESSING = "image_preprocessing"
    DVS_SIMULATION = "dvs_simulation" 
    FORMAT_CONVERSION = "format_conversion"
    IMAGE_RECONSTRUCTION = "image_reconstruction"  # 可选外部执行
    H5_STORAGE = "h5_storage"
    VALIDATION = "validation"

@dataclass
class TimeStampConfig:
    """时间戳配置"""
    start_time_us: int = 0  # 起始时间（微秒）
    dt_us: int = 1000      # 时间间隔（微秒，默认1ms）
    frame_rate: float = 1000.0  # 有效帧率（Hz）

@dataclass
class DVSConfig:
    """DVS仿真器配置"""
    camera_type: str = "DVS346"
    model_params: List[float] = None  # K参数
    dt_us: int = 1000  # 1ms间隔
    resolution: Tuple[int, int] = (346, 240)  # (width, height)
    
    def __post_init__(self):
        if self.model_params is None:
            # DVS346默认参数
            self.model_params = [6.517776598640289, 20, 0.0001, 1e-7, 5e-9, 1e-05]

@dataclass
class EVREALConfig:
    """EVREAL格式配置"""
    num_bins: int = 5
    voxel_method: str = "between_frames"
    keep_ratio: float = 1.0
    output_methods: List[str] = None  # 重建方法列表
    
    def __post_init__(self):
        if self.output_methods is None:
            self.output_methods = ["E2VID", "FireNet", "HyperE2VID"]

@dataclass
class PipelineConfig:
    """Pipeline总配置"""
    dataset_name: str = "lego"
    input_dir: Path = Path("datasets/lego/train")
    output_dir: Path = Path("outputs")
    temp_dir: Path = Path("temp")
    
    timestamp_config: TimeStampConfig = None
    dvs_config: DVSConfig = None  
    evreal_config: EVREALConfig = None
    
    # 执行选项
    skip_reconstruction: bool = False  # 是否跳过图像重建（外部执行）
    cleanup_temp: bool = True  # 是否清理临时文件
    
    def __post_init__(self):
        if self.timestamp_config is None:
            self.timestamp_config = TimeStampConfig()
        if self.dvs_config is None:
            self.dvs_config = DVSConfig()
        if self.evreal_config is None:
            self.evreal_config = EVREALConfig()

class PipelineInterface:
    """Pipeline接口定义"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.current_stage = None
        self.stage_results = {}  # 存储各阶段结果
        
    def execute_stage(self, stage: PipelineStage) -> Dict:
        """执行指定阶段"""
        raise NotImplementedError
        
    def get_stage_output(self, stage: PipelineStage) -> Optional[Dict]:
        """获取阶段输出结果"""
        return self.stage_results.get(stage.value)
        
    def validate_stage_output(self, stage: PipelineStage) -> bool:
        """验证阶段输出"""
        raise NotImplementedError

class DataFormat:
    """数据格式定义"""
    
    @dataclass
    class ImageSequence:
        """图像序列格式"""
        image_paths: List[Path]
        timestamps_us: List[int]
        info_file: Path  # DVS输入所需的info.txt
        
    @dataclass 
    class DVSEvents:
        """DVS事件格式"""
        events: np.ndarray  # Shape: (N, 4) - [x, y, timestamp_us, polarity]
        output_file: Path   # txt文件路径
        metadata: Dict      # 包含分辨率等信息
        
    @dataclass
    class EVREALFormat:
        """EVREAL标准格式"""
        events_ts: np.ndarray    # 时间戳（秒）
        events_xy: np.ndarray    # 坐标 [x,y] shape=(N,2)
        events_p: np.ndarray     # 极性 0/1
        images: Optional[np.ndarray] = None       # 目标图像（可选）
        images_ts: Optional[np.ndarray] = None    # 图像时间戳（可选）
        image_event_indices: Optional[np.ndarray] = None  # 图像事件对应（可选）
        metadata: Dict = None    # 传感器信息
        
    @dataclass
    class H5Events:
        """H5存储格式"""
        events_t: np.ndarray  # events/t
        events_x: np.ndarray  # events/x  
        events_y: np.ndarray  # events/y
        events_p: np.ndarray  # events/p
        h5_file: Path
        metadata: Dict

# 数据格式转换工具
class FormatConverter:
    """格式转换器"""
    
    @staticmethod
    def dvs_to_evreal(dvs_events: DataFormat.DVSEvents) -> DataFormat.EVREALFormat:
        """DVS格式转EVREAL格式
        DVS-Voltmeter输出格式: [timestamp_us, x, y, polarity]
        EVREAL期望格式: events_ts(秒), events_xy(x,y), events_p(0/1)
        """
        events = dvs_events.events
        
        # 时间戳转换：微秒 → 秒 (第0列)
        events_ts = events[:, 0] / 1e6
        
        # 坐标提取并交换为EVREAL期望的y,x顺序 (第1,2列)
        # EVREAL使用(ys, xs)索引，所以需要交换顺序
        # DVS格式是[t, x, y, p]，分析实际坐标范围：
        # 第2列∈[20,639] 是X坐标 (图像宽度640)  
        # 第3列∈[0,479] 是Y坐标 (图像高度480)
        dvs_x = events[:, 1].astype(np.int32)  # DVS X坐标 [20, 639]
        dvs_y = events[:, 2].astype(np.int32)  # DVS Y坐标 [0, 479]
        
        # 直接使用DVS的X,Y坐标
        events_xy = np.column_stack([dvs_x, dvs_y])  # [x, y]顺序
        
        # 极性转换 (第3列)
        events_p = events[:, 3].astype(np.int32)
        
        return DataFormat.EVREALFormat(
            events_ts=events_ts,
            events_xy=events_xy, 
            events_p=events_p,
            metadata=dvs_events.metadata
        )
    
    @staticmethod
    def dvs_to_h5(dvs_events: DataFormat.DVSEvents) -> DataFormat.H5Events:
        """DVS格式转H5格式"""
        events = dvs_events.events
        
        return DataFormat.H5Events(
            events_t=events[:, 2],  # 保持微秒时间戳
            events_x=events[:, 0],
            events_y=events[:, 1], 
            events_p=events[:, 3],
            h5_file=dvs_events.output_file.with_suffix('.h5'),
            metadata=dvs_events.metadata
        )
    
    @staticmethod
    def h5_to_dvs(h5_events: DataFormat.H5Events) -> DataFormat.DVSEvents:
        """H5格式转回DVS格式"""
        events = np.column_stack([
            h5_events.events_x,
            h5_events.events_y,
            h5_events.events_t,
            h5_events.events_p
        ])
        
        return DataFormat.DVSEvents(
            events=events,
            output_file=h5_events.h5_file.with_suffix('.txt'),
            metadata=h5_events.metadata
        )

# 时间戳管理器
class TimestampManager:
    """时间戳对齐管理"""
    
    def __init__(self, config: TimeStampConfig):
        self.config = config
        
    def generate_image_timestamps(self, num_images: int) -> List[int]:
        """为图像序列生成时间戳"""
        timestamps = []
        for i in range(num_images):
            timestamp_us = self.config.start_time_us + i * self.config.dt_us
            timestamps.append(timestamp_us)
        return timestamps
    
    def align_reconstruction_timestamps(self, 
                                     original_timestamps: List[int],
                                     reconstruction_files: List[Path]) -> Dict[int, Path]:
        """对齐重建图像的时间戳"""
        if len(original_timestamps) != len(reconstruction_files):
            raise ValueError("时间戳和重建文件数量不匹配")
            
        return dict(zip(original_timestamps, reconstruction_files))
    
    def validate_timestamp_alignment(self, 
                                   original_ts: List[int],
                                   dvs_ts: np.ndarray,
                                   reconstruction_ts: List[int]) -> bool:
        """验证时间戳对齐正确性"""
        # 检查DVS事件时间戳范围
        dvs_min, dvs_max = dvs_ts.min(), dvs_ts.max()
        original_min, original_max = min(original_ts), max(original_ts)
        
        # DVS事件应该在原始图像时间范围内
        if dvs_min < original_min or dvs_max > original_max:
            return False
            
        # 重建时间戳应该与原始时间戳匹配
        if set(original_ts) != set(reconstruction_ts):
            return False
            
        return True

if __name__ == "__main__":
    # 架构测试
    config = PipelineConfig()
    print("Pipeline架构设计完成")
    print(f"数据集: {config.dataset_name}")
    print(f"DVS配置: {config.dvs_config}")
    print(f"EVREAL配置: {config.evreal_config}")
    print(f"时间戳配置: {config.timestamp_config}")