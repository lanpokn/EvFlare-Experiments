#!/usr/bin/env python3
"""
数据格式转换器
DVS输出 → EVREAL格式 → H5格式

Author: Claude Code Assistant
Date: 2025-09-17
"""

import os
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

import sys
sys.path.append('..')
sys.path.append('.')
try:
    from pipeline_architecture import DataFormat, FormatConverter
except ImportError:
    sys.path.append('/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction')
    from pipeline_architecture import DataFormat, FormatConverter

@dataclass
class ConversionConfig:
    """格式转换配置"""
    dataset_name: str = "lego"
    dataset_dir: Path = Path("datasets/lego")
    
    # EVREAL输出配置
    evreal_dir: Optional[Path] = None
    
    # H5输出配置  
    h5_dir: Optional[Path] = None
    
    # 元数据
    sensor_resolution: Tuple[int, int] = (480, 640)  # (height, width) - EVREAL要求格式
    
    def __post_init__(self):
        if self.evreal_dir is None:
            self.evreal_dir = self.dataset_dir / "events_evreal"
        if self.h5_dir is None:
            self.h5_dir = self.dataset_dir / "events_h5"

class DVSToEVREALConverter:
    """DVS格式转EVREAL格式转换器"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        
    def load_dvs_events(self, dvs_file: Path) -> Optional[DataFormat.DVSEvents]:
        """加载DVS事件文件"""
        if not dvs_file.exists():
            print(f"错误：DVS事件文件不存在 {dvs_file}")
            return None
            
        try:
            print(f"加载DVS事件文件: {dvs_file}")
            events = np.loadtxt(dvs_file)
            
            if events.size == 0:
                print("警告：事件文件为空")
                return None
                
            if events.ndim == 1:
                events = events.reshape(1, -1)
                
            # 验证事件格式：[timestamp_us, x, y, polarity]
            if events.shape[1] != 4:
                print(f"错误：事件格式不正确，期望4列，实际{events.shape[1]}列")
                return None
                
            print(f"成功加载 {events.shape[0]} 个事件")
            print(f"时间范围: {events[:, 0].min():.0f} - {events[:, 0].max():.0f} μs")
            print(f"空间范围: x[{events[:, 1].min():.0f}, {events[:, 1].max():.0f}] y[{events[:, 2].min():.0f}, {events[:, 2].max():.0f}]")
            print(f"极性分布: ON={np.sum(events[:, 3] == 1)}, OFF={np.sum(events[:, 3] == 0)}")
            
            metadata = {
                "num_events": events.shape[0],
                "time_range_us": (float(events[:, 0].min()), float(events[:, 0].max())),
                "spatial_range": {
                    "x_range": (int(events[:, 1].min()), int(events[:, 1].max())),
                    "y_range": (int(events[:, 2].min()), int(events[:, 2].max()))
                },
                "sensor_resolution": self.config.sensor_resolution,
                "source_file": str(dvs_file)
            }
            
            return DataFormat.DVSEvents(
                events=events,
                output_file=dvs_file,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"加载DVS事件文件时出错: {e}")
            return None
    
    def convert_to_evreal(self, dvs_events: DataFormat.DVSEvents) -> Optional[DataFormat.EVREALFormat]:
        """转换为EVREAL格式"""
        print("转换为EVREAL格式...")
        
        try:
            # 使用架构中的转换器
            evreal_format = FormatConverter.dvs_to_evreal(dvs_events)
            
            # 添加额外的元数据
            evreal_format.metadata = {
                **dvs_events.metadata,
                "format": "EVREAL",
                "conversion_timestamp": str(np.datetime64('now')),
                "sensor_resolution": self.config.sensor_resolution
            }
            
            print(f"EVREAL格式转换完成:")
            print(f"  事件时间戳: {evreal_format.events_ts.shape[0]} 个 (秒)")
            print(f"  事件坐标: {evreal_format.events_xy.shape}")
            print(f"  事件极性: {evreal_format.events_p.shape[0]} 个")
            
            return evreal_format
            
        except Exception as e:
            print(f"转换为EVREAL格式时出错: {e}")
            return None
    
    def save_evreal_format(self, evreal_data: DataFormat.EVREALFormat, sequence_name: str = "lego_sequence") -> bool:
        """保存EVREAL格式文件"""
        print(f"保存EVREAL格式到: {self.config.evreal_dir}")
        
        try:
            # 创建输出目录
            self.config.evreal_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存6个必需的npy文件
            np.save(self.config.evreal_dir / "events_ts.npy", evreal_data.events_ts)
            np.save(self.config.evreal_dir / "events_xy.npy", evreal_data.events_xy)
            np.save(self.config.evreal_dir / "events_p.npy", evreal_data.events_p)
            
            # 如果有目标图像，也保存（可选）
            if evreal_data.images is not None:
                np.save(self.config.evreal_dir / "images.npy", evreal_data.images)
            if evreal_data.images_ts is not None:
                np.save(self.config.evreal_dir / "images_ts.npy", evreal_data.images_ts)
            if evreal_data.image_event_indices is not None:
                np.save(self.config.evreal_dir / "image_event_indices.npy", evreal_data.image_event_indices)
            
            # 保存元数据
            metadata_file = self.config.evreal_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                # 处理numpy类型以便JSON序列化
                metadata_serializable = {}
                for key, value in evreal_data.metadata.items():
                    if isinstance(value, (np.integer, np.floating)):
                        metadata_serializable[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        metadata_serializable[key] = value.tolist()
                    else:
                        metadata_serializable[key] = value
                        
                json.dump(metadata_serializable, f, indent=2)
            
            print("EVREAL格式文件保存成功:")
            print(f"  events_ts.npy: {evreal_data.events_ts.shape}")
            print(f"  events_xy.npy: {evreal_data.events_xy.shape}")
            print(f"  events_p.npy: {evreal_data.events_p.shape}")
            print(f"  metadata.json: 元数据文件")
            
            return True
            
        except Exception as e:
            print(f"保存EVREAL格式时出错: {e}")
            return False

class EVREALToH5Converter:
    """EVREAL格式转H5格式转换器"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        
    def convert_dvs_to_h5(self, dvs_events: DataFormat.DVSEvents, output_name: str = "lego_events.h5") -> bool:
        """直接从DVS格式转换并保存为H5"""
        print(f"转换DVS事件到H5格式: {output_name}")
        
        try:
            # 创建输出目录
            self.config.h5_dir.mkdir(parents=True, exist_ok=True)
            
            # H5文件路径
            h5_file = self.config.h5_dir / output_name
            
            events = dvs_events.events
            
            # 保存为H5格式
            with h5py.File(h5_file, 'w') as f:
                # 创建events组
                events_group = f.create_group('events')
                
                # 按照用户要求的格式保存
                events_group.create_dataset('t', data=events[:, 2])  # 时间戳(微秒)
                events_group.create_dataset('x', data=events[:, 0])  # x坐标
                events_group.create_dataset('y', data=events[:, 1])  # y坐标
                events_group.create_dataset('p', data=events[:, 3])  # 极性
                
                # 保存元数据作为属性
                events_group.attrs['num_events'] = events.shape[0]
                events_group.attrs['time_range_us'] = dvs_events.metadata['time_range_us']
                events_group.attrs['sensor_width'] = self.config.sensor_resolution[0]
                events_group.attrs['sensor_height'] = self.config.sensor_resolution[1]
                events_group.attrs['source_file'] = str(dvs_events.output_file)
                
            print(f"H5文件保存成功: {h5_file}")
            print(f"  events/t: {events.shape[0]} 个时间戳")
            print(f"  events/x: {events.shape[0]} 个x坐标")
            print(f"  events/y: {events.shape[0]} 个y坐标")
            print(f"  events/p: {events.shape[0]} 个极性值")
            
            return True
            
        except Exception as e:
            print(f"保存H5格式时出错: {e}")
            return False
    
    def load_h5_events(self, h5_file: Path) -> Optional[DataFormat.H5Events]:
        """加载H5事件文件"""
        if not h5_file.exists():
            print(f"错误：H5文件不存在 {h5_file}")
            return None
            
        try:
            with h5py.File(h5_file, 'r') as f:
                events_t = f['events/t'][:]
                events_x = f['events/x'][:]
                events_y = f['events/y'][:]
                events_p = f['events/p'][:]
                
                # 读取元数据
                metadata = dict(f['events'].attrs)
                
            print(f"从H5文件加载 {len(events_t)} 个事件")
            
            return DataFormat.H5Events(
                events_t=events_t,
                events_x=events_x,
                events_y=events_y,
                events_p=events_p,
                h5_file=h5_file,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"加载H5文件时出错: {e}")
            return None
    
    def h5_to_dvs_txt(self, h5_events: DataFormat.H5Events, output_file: Path) -> bool:
        """H5格式转回DVS txt格式"""
        try:
            # 重组事件数据
            events = np.column_stack([
                h5_events.events_x,
                h5_events.events_y, 
                h5_events.events_t,
                h5_events.events_p
            ])
            
            # 保存为txt格式
            np.savetxt(output_file, events, fmt='%1.0f')
            print(f"H5转DVS txt完成: {output_file}")
            return True
            
        except Exception as e:
            print(f"H5转DVS txt时出错: {e}")
            return False
    
    def h5_to_evreal_npy(self, h5_events: DataFormat.H5Events, output_dir: Path) -> bool:
        """H5格式转回EVREAL npy格式"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 转换时间戳：微秒 → 秒
            events_ts = h5_events.events_t / 1e6
            events_xy = np.column_stack([h5_events.events_x, h5_events.events_y])
            events_p = h5_events.events_p
            
            # 保存EVREAL格式
            np.save(output_dir / "events_ts.npy", events_ts)
            np.save(output_dir / "events_xy.npy", events_xy)
            np.save(output_dir / "events_p.npy", events_p)
            
            print(f"H5转EVREAL格式完成: {output_dir}")
            return True
            
        except Exception as e:
            print(f"H5转EVREAL格式时出错: {e}")
            return False

class FormatConverterPipeline:
    """格式转换Pipeline"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.dvs_to_evreal = DVSToEVREALConverter(config)
        self.evreal_to_h5 = EVREALToH5Converter(config)
        
    def convert_dvs_events(self, dvs_file: Path, sequence_name: str = "lego_sequence") -> Dict[str, bool]:
        """完整的DVS事件转换流程"""
        print("=" * 60)
        print("开始DVS事件格式转换")
        print("=" * 60)
        
        results = {"evreal": False, "h5": False}
        
        # 1. 加载DVS事件
        dvs_events = self.dvs_to_evreal.load_dvs_events(dvs_file)
        if dvs_events is None:
            return results
            
        # 2. 转换为EVREAL格式
        evreal_data = self.dvs_to_evreal.convert_to_evreal(dvs_events)
        if evreal_data is not None:
            results["evreal"] = self.dvs_to_evreal.save_evreal_format(evreal_data, sequence_name)
        
        # 3. 转换为H5格式
        h5_name = f"{sequence_name}.h5"
        results["h5"] = self.evreal_to_h5.convert_dvs_to_h5(dvs_events, h5_name)
        
        print("=" * 60)
        print("格式转换完成")
        print(f"EVREAL格式: {'成功' if results['evreal'] else '失败'}")
        print(f"H5格式: {'成功' if results['h5'] else '失败'}")
        print("=" * 60)
        
        return results

def main():
    """测试函数"""
    # 配置
    config = ConversionConfig()
    
    # 查找DVS输出文件
    dvs_file = Path("datasets/lego/events_dvs/lego_train_events.txt")
    
    if not dvs_file.exists():
        print(f"DVS事件文件不存在: {dvs_file}")
        print("请先运行DVS仿真器生成事件数据")
        # 兼容旧路径
        old_dvs_file = Path("temp/dvs_output/lego_sequence.txt")
        if old_dvs_file.exists():
            print(f"找到旧路径的DVS文件: {old_dvs_file}")
            dvs_file = old_dvs_file
        else:
            return
    
    # 执行转换
    pipeline = FormatConverterPipeline(config)
    results = pipeline.convert_dvs_events(dvs_file)
    
    # 显示结果摘要
    if results["h5"] or results["evreal"]:
        print(f"\n格式转换完成!")
        print(f"EVREAL格式: {config.evreal_dir}")
        print(f"H5格式: {config.h5_dir}")
        print(f"双向转换功能已实现，可通过脚本调用")

if __name__ == "__main__":
    main()