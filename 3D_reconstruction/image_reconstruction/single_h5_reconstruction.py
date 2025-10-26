#!/usr/bin/env python3
"""
单个H5文件图像重建脚本
专门处理独立的H5文件，不依赖数据集结构

用法:
    python single_h5_reconstruction.py <h5_file> <output_dir>

Author: Claude Code Assistant
Date: 2025-10-25
"""

import os
import sys
import h5py
import shutil
import numpy as np
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

# 添加modules路径
sys.path.append('.')
sys.path.append('..')


class SingleH5Reconstructor:
    """单个H5文件重建器"""

    def __init__(self, h5_file: Path, output_dir: Path, num_images: int = 200):
        self.h5_file = Path(h5_file)
        # 确保output_dir是相对于image_reconstruction文件夹的
        script_dir = Path(__file__).parent
        self.output_dir = script_dir / output_dir
        self.temp_dir = script_dir / "temp" / "single_reconstruction"

        # EVREAL路径
        self.evreal_path = Path("/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main")

        # 重建图像数量（可配置）
        self.num_images = num_images

        # 支持的重建方法
        self.methods = [
            "E2VID",
            "E2VID+",
            "FireNet",
            "FireNet+",
            "SPADE-E2VID",
            "SSL-E2VID",
            "ET-Net",
            "HyperE2VID"
        ]

        # 验证输入
        if not self.h5_file.exists():
            raise FileNotFoundError(f"H5文件不存在: {self.h5_file}")
        if not self.evreal_path.exists():
            raise FileNotFoundError(f"EVREAL路径不存在: {self.evreal_path}")

    def load_h5_events(self) -> Optional[Dict]:
        """加载H5事件数据"""
        print(f"📂 加载H5文件: {self.h5_file.name}")

        try:
            with h5py.File(self.h5_file, 'r') as f:
                # 检测格式
                if 'events' in f and isinstance(f['events'], h5py.Group):
                    # 分组格式
                    events_t = f['events/t'][:]
                    events_x = f['events/x'][:]
                    events_y = f['events/y'][:]
                    events_p = f['events/p'][:]
                    print(f"  ✓ 格式: 分组格式 (events/t, events/x, events/y, events/p)")

                elif 'events' in f and isinstance(f['events'], h5py.Dataset):
                    # 数组格式
                    events_data = f['events'][:]
                    events_t = events_data[:, 0]
                    events_x = events_data[:, 1]
                    events_y = events_data[:, 2]
                    events_p = events_data[:, 3]
                    print(f"  ✓ 格式: 数组格式 (events[:, [t,x,y,p]])")

                else:
                    print(f"  ❌ 未识别的H5文件格式")
                    return None

            # 统计信息
            num_events = len(events_t)
            print(f"  ✓ 事件数量: {num_events:,}")
            print(f"  ✓ 时间范围: {events_t.min():.0f} - {events_t.max():.0f} μs")
            print(f"  ✓ 空间范围: X[{events_x.min():.0f}, {events_x.max():.0f}], Y[{events_y.min():.0f}, {events_y.max():.0f}]")
            print(f"  ✓ 极性分布: +{(events_p == 1).sum():,}, -{(events_p == 0).sum():,}")

            return {
                'events_t': events_t,
                'events_x': events_x,
                'events_y': events_y,
                'events_p': events_p,
                'num_events': num_events,
                'time_range_us': (float(events_t.min()), float(events_t.max())),
                'spatial_range': {
                    'x_range': (int(events_x.min()), int(events_x.max())),
                    'y_range': (int(events_y.min()), int(events_y.max()))
                }
            }

        except Exception as e:
            print(f"  ❌ 加载H5文件失败: {e}")
            return None

    def create_evreal_structure(self, events_data: Dict) -> Path:
        """创建EVREAL数据结构"""
        print("\n🔧 创建EVREAL数据结构...")

        # 清理旧的临时目录
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        # 创建目录结构
        evreal_dir = self.temp_dir / "evreal_data"
        sequence_dir = evreal_dir / "sequence"
        sequence_dir.mkdir(parents=True, exist_ok=True)

        # 转换事件数据为EVREAL格式
        events_ts = events_data['events_t'].astype(np.float64) / 1000000.0  # 微秒→秒
        events_xy = np.column_stack([
            events_data['events_x'].astype(np.int16),
            events_data['events_y'].astype(np.int16)
        ])
        events_p = events_data['events_p'].astype(np.int8)

        # 处理极性格式: 如果是-1/1转换为0/1
        if events_p.min() < 0:
            events_p = ((events_p + 1) // 2).astype(np.int8)  # -1→0, 1→1
            print(f"  ✓ 极性转换: -1/1 → 0/1")

        # 保存事件数据
        np.save(sequence_dir / "events_ts.npy", events_ts)
        np.save(sequence_dir / "events_xy.npy", events_xy)
        np.save(sequence_dir / "events_p.npy", events_p)

        print(f"  ✓ events_ts.npy: {events_ts.shape}")
        print(f"  ✓ events_xy.npy: {events_xy.shape}")
        print(f"  ✓ events_p.npy: {events_p.shape}, 范围=[{events_p.min()}, {events_p.max()}]")

        # 生成虚拟的图像时间戳和索引（EVREAL要求）
        # 使用配置的图像数量，均匀分布
        num_images = self.num_images
        time_start = events_ts.min()
        time_end = events_ts.max()
        dt = (time_end - time_start) / (num_images - 1)

        images_ts = np.array([time_start + i * dt for i in range(num_images)], dtype=np.float64)

        # 生成图像事件索引
        num_events = len(events_ts)
        image_event_indices = np.zeros((num_images, 2), dtype=np.int64)
        for i in range(num_images):
            start_idx = int(i * num_events / num_images)
            end_idx = int((i + 1) * num_events / num_images)
            if i == num_images - 1:
                end_idx = num_events
            image_event_indices[i] = [start_idx, end_idx]

        np.save(sequence_dir / "images_ts.npy", images_ts)
        np.save(sequence_dir / "image_event_indices.npy", image_event_indices)

        # 创建虚拟images.npy（纯黑图像）满足EVREAL数据加载器要求
        # 检测图像分辨率
        height = int(events_data['spatial_range']['y_range'][1] - events_data['spatial_range']['y_range'][0] + 1)
        width = int(events_data['spatial_range']['x_range'][1] - events_data['spatial_range']['x_range'][0] + 1)
        dummy_images = np.zeros((num_images, height, width, 3), dtype=np.uint8)
        np.save(sequence_dir / "images.npy", dummy_images)

        print(f"  ✓ images_ts.npy: {images_ts.shape}")
        print(f"  ✓ image_event_indices.npy: {image_event_indices.shape}")
        print(f"  ✓ images.npy: {dummy_images.shape} (虚拟黑色图像)")

        # 保存元数据 (EVREAL要求的格式)
        metadata = {
            'num_events': int(events_data['num_events']),
            'time_range_us': events_data['time_range_us'],
            'spatial_range': events_data['spatial_range'],
            'num_images': num_images,
            'source_file': str(self.h5_file),
            'sensor_resolution': [height, width]  # EVREAL要求: [height, width]
        }

        with open(sequence_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ metadata.json")

        return evreal_dir

    def create_evreal_config(self, evreal_dir: Path, dataset_name: str) -> Path:
        """创建EVREAL数据集配置"""
        config_dir = self.evreal_path / "config" / "dataset"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_file = config_dir / f"{dataset_name}.json"

        config = {
            "root_path": str(evreal_dir.absolute()),
            "sequences": {
                "sequence": {}
            }
        }

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ 创建EVREAL配置: {config_file}")
        return config_file

    def run_evreal_reconstruction(self, dataset_name: str) -> Dict[str, bool]:
        """运行EVREAL重建"""
        print(f"\n🚀 开始EVREAL重建...")

        results = {}

        for method in self.methods:
            print(f"\n  运行方法: {method}")

            try:
                cmd = [
                    "python", "eval.py",
                    "-m", method,
                    "-c", "std",
                    "-d", dataset_name,
                    "-qm", "mse", "ssim", "lpips"
                ]

                env_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && {' '.join(cmd)}"

                result = subprocess.run(
                    ["bash", "-c", env_cmd],
                    cwd=self.evreal_path,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10分钟超时
                )

                if result.returncode == 0:
                    print(f"    ✅ {method} 重建成功")
                    results[method] = True
                else:
                    print(f"    ❌ {method} 重建失败 (返回码: {result.returncode})")
                    if result.stderr:
                        print(f"       完整错误:\n{result.stderr}")  # 打印完整错误
                    results[method] = False

            except subprocess.TimeoutExpired:
                print(f"    ❌ {method} 超时")
                results[method] = False
            except Exception as e:
                print(f"    ❌ {method} 出错: {e}")
                results[method] = False

        return results

    def copy_results(self, dataset_name: str, results: Dict[str, bool]) -> Dict[str, Path]:
        """复制重建结果"""
        print(f"\n📦 复制重建结果到: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        copied = {}

        outputs_base = self.evreal_path / "outputs" / "std"

        for method, success in results.items():
            if not success:
                continue

            # 查找输出目录
            evreal_output = None
            for dataset_dir in outputs_base.glob("*"):
                if dataset_dir.is_dir() and dataset_name in dataset_dir.name:
                    sequence_dir = dataset_dir / "sequence" / method
                    if sequence_dir.exists():
                        evreal_output = sequence_dir
                        break

            if evreal_output is None:
                print(f"  ⚠️  未找到 {method} 的输出")
                continue

            # 目标目录
            target_dir = self.output_dir / f"evreal_{method.lower().replace('+', '_plus')}"

            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            # 复制PNG文件
            recon_files = sorted(evreal_output.glob("*.png"))
            if not recon_files:
                print(f"  ⚠️  {method} 没有PNG输出")
                continue

            for i, recon_file in enumerate(recon_files):
                target_file = target_dir / f"{i+1:04d}.png"
                shutil.copy2(recon_file, target_file)

            copied[method] = target_dir
            print(f"  ✅ {method}: {len(recon_files)} 张图像 → {target_dir}")

        return copied

    def cleanup(self):
        """清理临时文件"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 清理临时目录: {self.temp_dir}")

    def run(self) -> Dict:
        """运行完整重建流程"""
        print("=" * 60)
        print("单个H5文件图像重建")
        print("=" * 60)

        try:
            # 1. 加载H5数据
            events_data = self.load_h5_events()
            if events_data is None:
                return {"success": False, "error": "加载H5数据失败"}

            # 2. 创建EVREAL数据结构
            evreal_dir = self.create_evreal_structure(events_data)

            # 3. 创建EVREAL配置
            dataset_name = f"single_h5_{self.h5_file.stem}"
            self.create_evreal_config(evreal_dir, dataset_name)

            # 4. 运行EVREAL重建
            reconstruction_results = self.run_evreal_reconstruction(dataset_name)

            # 5. 复制结果
            copied_results = self.copy_results(dataset_name, reconstruction_results)

            # 6. 清理临时文件
            self.cleanup()

            # 7. 生成摘要
            successful = [k for k, v in reconstruction_results.items() if v]
            failed = [k for k, v in reconstruction_results.items() if not v]

            print("\n" + "=" * 60)
            print("🎉 重建完成!")
            print("=" * 60)
            print(f"✅ 成功方法 ({len(successful)}): {', '.join(successful)}")
            print(f"❌ 失败方法 ({len(failed)}): {', '.join(failed)}")
            print(f"📂 输出目录: {self.output_dir.absolute()}")
            print(f"📊 总图像数: {sum(len(list(d.glob('*.png'))) for d in copied_results.values())}")

            return {
                "success": True,
                "successful_methods": successful,
                "failed_methods": failed,
                "output_dir": str(self.output_dir.absolute()),
                "copied_results": {k: str(v) for k, v in copied_results.items()}
            }

        except Exception as e:
            print(f"\n❌ 重建失败: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python single_h5_reconstruction.py <h5_file> <output_dir> [num_images]")
        print("示例: python single_h5_reconstruction.py input.h5 ./output")
        print("      python single_h5_reconstruction.py input.h5 ./output 40  # 生成40张图像")
        sys.exit(1)

    h5_file = sys.argv[1]
    output_dir = sys.argv[2]
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 200  # 默认200张

    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"当前conda环境: {conda_env}")
    if conda_env != 'Umain2':
        print("⚠️  警告: 未在Umain2环境中运行!")
        print("建议使用: source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # 创建重建器并运行
    print(f"配置: 生成 {num_images} 张重建图像")
    reconstructor = SingleH5Reconstructor(h5_file, output_dir, num_images)
    result = reconstructor.run()

    if result["success"]:
        print("\n✅ 完成!")
    else:
        print(f"\n❌ 失败: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
