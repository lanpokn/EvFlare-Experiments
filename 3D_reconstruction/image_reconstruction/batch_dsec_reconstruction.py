#!/usr/bin/env python3
"""
DSEC数据集批量H5重建脚本 (支持断点续存)

功能:
- 扫描DSEC_data目录下的所有方法文件夹
- 每个方法中每5个H5文件取1个进行重建
- 输出结构与DSEC_data一致，H5文件变成文件夹
- 处理顺序: 先处理同名H5的所有方法，再处理下一个H5
- 断点续存: 可从中断处继续，自动跳过已完成的任务

使用方法:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2
python batch_dsec_reconstruction.py <dsec_data_dir> <output_base_dir>

示例:
python batch_dsec_reconstruction.py \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data" \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data_reconstructed"

Author: Claude Code Assistant
Date: 2025-10-26
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Set
import subprocess

# 添加single_h5_reconstruction的路径
sys.path.append(str(Path(__file__).parent))
from single_h5_reconstruction import SingleH5Reconstructor


class DSECBatchProcessor:
    """DSEC数据集批量处理器 (支持断点续存)"""

    def __init__(self, dsec_data_dir: Path, output_base_dir: Path, num_images: int = 40):
        self.dsec_data_dir = Path(dsec_data_dir)
        self.output_base_dir = Path(output_base_dir)

        # 排除的目录
        self.exclude_dirs = {'visualize', '.', '..'}

        # 采样间隔
        self.sample_interval = 5  # 每5个取1个

        # 重建图像数量（DSEC默认40张）
        self.num_images = num_images

        # 进度文件路径
        self.progress_file = self.output_base_dir / ".batch_progress.json"

        # 已完成的任务集合
        self.completed_tasks: Set[str] = self.load_progress()

        if not self.dsec_data_dir.exists():
            raise FileNotFoundError(f"DSEC数据目录不存在: {self.dsec_data_dir}")

    def load_progress(self) -> Set[str]:
        """加载进度文件，返回已完成任务的集合"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    completed = set(data.get('completed', []))
                    print(f"📂 加载进度文件: 已完成 {len(completed)} 个任务")
                    return completed
            except Exception as e:
                print(f"⚠️  进度文件读取失败，将重新开始: {e}")
                return set()
        return set()

    def save_progress(self, method_name: str, h5_basename: str):
        """保存单个任务的完成状态"""
        task_id = f"{method_name}:{h5_basename}"
        self.completed_tasks.add(task_id)

        # 确保输出目录存在
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # 保存到JSON文件
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'completed': sorted(list(self.completed_tasks)),
                    'total_completed': len(self.completed_tasks),
                    'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存进度失败: {e}")

    def is_task_completed(self, method_name: str, h5_basename: str) -> bool:
        """检查任务是否已完成"""
        task_id = f"{method_name}:{h5_basename}"

        # 首先检查进度文件记录
        if task_id in self.completed_tasks:
            return True

        # 其次检查输出目录是否存在且包含重建结果
        output_dir = self.output_base_dir / method_name / h5_basename
        if output_dir.exists():
            # 检查是否包含重建方法目录（evreal_*）
            recon_dirs = list(output_dir.glob("evreal_*"))
            if recon_dirs:
                # 检查是否有PNG文件
                for recon_dir in recon_dirs:
                    png_files = list(recon_dir.glob("*.png"))
                    if png_files:
                        # 标记为已完成并保存
                        self.completed_tasks.add(task_id)
                        return True

        return False

    def get_method_directories(self) -> List[Path]:
        """获取所有方法目录（排除visualize）"""
        method_dirs = []

        for item in self.dsec_data_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name in self.exclude_dirs:
                continue

            # 检查是否包含H5文件
            h5_files = list(item.glob("*.h5"))
            if h5_files:
                method_dirs.append(item)
                print(f"✓ 发现方法: {item.name} ({len(h5_files)} 个H5文件)")

        return sorted(method_dirs)

    def get_all_h5_basenames(self, method_dirs: List[Path]) -> List[str]:
        """获取所有唯一的H5文件basename（不含扩展名）"""
        all_basenames = set()

        for method_dir in method_dirs:
            for h5_file in method_dir.glob("*.h5"):
                all_basenames.add(h5_file.stem)

        # 排序并返回
        sorted_basenames = sorted(list(all_basenames))
        print(f"✓ 发现唯一H5文件: {len(sorted_basenames)} 个")

        return sorted_basenames

    def sample_h5_basenames(self, basenames: List[str]) -> List[str]:
        """从H5文件basename列表中按间隔采样"""
        sampled = basenames[::self.sample_interval]
        print(f"✓ 采样结果: {len(basenames)} → {len(sampled)} 个文件 (间隔={self.sample_interval})")
        return sampled

    def process_single_h5(self, method_dir: Path, h5_basename: str) -> bool:
        """处理单个H5文件在特定方法目录中的重建

        Returns:
            bool: 是否成功
        """
        method_name = method_dir.name
        h5_file = method_dir / f"{h5_basename}.h5"

        # 检查H5文件是否存在
        if not h5_file.exists():
            print(f"    ⚠️  文件不存在: {h5_file.name}")
            return False

        # 输出目录：方法名/H5basename/
        h5_output_dir = self.output_base_dir / method_name / h5_basename

        try:
            # 创建重建器并运行
            reconstructor = SingleH5Reconstructor(h5_file, h5_output_dir, self.num_images)
            result = reconstructor.run()

            if result["success"]:
                print(f"    ✅ 成功: {len(result['successful_methods'])} 种方法")
                return True
            else:
                print(f"    ❌ 失败: {result.get('error', '未知错误')}")
                return False

        except Exception as e:
            print(f"    ❌ 异常: {e}")
            return False

    def process_all_methods(self):
        """处理所有H5文件（新顺序：先处理同名H5的所有方法）"""
        print("=" * 60)
        print("DSEC数据集批量H5重建 (支持断点续存)")
        print("=" * 60)
        print(f"输入目录: {self.dsec_data_dir}")
        print(f"输出目录: {self.output_base_dir}")
        print(f"采样间隔: 每{self.sample_interval}个取1个")
        print(f"重建图像数: {self.num_images} 张")
        print("=" * 60)

        # 获取所有方法目录
        method_dirs = self.get_method_directories()

        if not method_dirs:
            print("❌ 未找到包含H5文件的方法目录")
            return

        print(f"\n共找到 {len(method_dirs)} 个方法目录")
        print(f"方法列表: {[d.name for d in method_dirs]}")

        # 获取所有唯一的H5文件basename
        all_basenames = self.get_all_h5_basenames(method_dirs)

        if not all_basenames:
            print("❌ 未找到H5文件")
            return

        # 采样H5文件
        sampled_basenames = self.sample_h5_basenames(all_basenames)

        # 创建输出基础目录
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # 统计数据
        stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'skipped_tasks': 0
        }

        start_time = time.time()

        # 新的处理顺序：外层循环是H5文件，内层循环是方法
        for h5_idx, h5_basename in enumerate(sampled_basenames, 1):
            print(f"\n{'#'*60}")
            print(f"H5文件进度: [{h5_idx}/{len(sampled_basenames)}] {h5_basename}")
            print(f"{'#'*60}")

            # 对每个方法处理这个H5文件
            for method_idx, method_dir in enumerate(method_dirs, 1):
                method_name = method_dir.name
                print(f"\n  [{method_idx}/{len(method_dirs)}] 方法: {method_name}")

                stats['total_tasks'] += 1

                # 检查是否已完成
                if self.is_task_completed(method_name, h5_basename):
                    print(f"    ⏭️  跳过 (已完成)")
                    stats['skipped_tasks'] += 1
                    continue

                # 处理H5文件
                success = self.process_single_h5(method_dir, h5_basename)

                if success:
                    stats['completed_tasks'] += 1
                    # 保存进度
                    self.save_progress(method_name, h5_basename)
                else:
                    stats['failed_tasks'] += 1

            # 每处理完一个H5文件组，显示当前统计
            print(f"\n  当前进度: 完成{stats['completed_tasks']} | "
                  f"跳过{stats['skipped_tasks']} | "
                  f"失败{stats['failed_tasks']} / "
                  f"总计{stats['total_tasks']}")

        # 总结
        elapsed_time = time.time() - start_time
        self.print_summary_new(stats, len(sampled_basenames), len(method_dirs), elapsed_time)

    def print_summary_new(self, stats: Dict, num_h5_files: int, num_methods: int, elapsed_time: float):
        """打印新的处理总结"""
        print(f"\n{'='*60}")
        print("🎉 批量处理完成!")
        print(f"{'='*60}")

        print(f"\n总体统计:")
        print(f"  H5文件数: {num_h5_files} 个")
        print(f"  方法数量: {num_methods} 个")
        print(f"  总任务数: {stats['total_tasks']} 个")
        print(f"  ✅ 成功完成: {stats['completed_tasks']} 个")
        print(f"  ⏭️  跳过已完成: {stats['skipped_tasks']} 个")
        print(f"  ❌ 失败: {stats['failed_tasks']} 个")
        print(f"  ⏱️  总耗时: {elapsed_time/60:.1f} 分钟")

        if stats['total_tasks'] > 0:
            success_rate = (stats['completed_tasks'] + stats['skipped_tasks']) / stats['total_tasks'] * 100
            print(f"  📊 成功率: {success_rate:.1f}%")

        print(f"\n输出目录: {self.output_base_dir}")
        print(f"进度文件: {self.progress_file}")
        print(f"{'='*60}")


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python batch_dsec_reconstruction.py <dsec_data_dir> <output_base_dir> [num_images]")
        print("\n示例:")
        print("  python batch_dsec_reconstruction.py \\")
        print("    /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data \\")
        print("    /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data_reconstructed")
        print("\n  # 自定义图像数量（默认40张）")
        print("  python batch_dsec_reconstruction.py DSEC_data DSEC_data_reconstructed 80")
        sys.exit(1)

    dsec_data_dir = sys.argv[1]
    output_base_dir = sys.argv[2]
    num_images = int(sys.argv[3]) if len(sys.argv) > 3 else 40  # 默认40张

    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"当前conda环境: {conda_env}")
    if conda_env != 'Umain2':
        print("⚠️  警告: 未在Umain2环境中运行!")
        print("建议使用: source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    try:
        # 创建处理器并运行
        print(f"配置: 每个H5文件生成 {num_images} 张重建图像\n")
        processor = DSECBatchProcessor(dsec_data_dir, output_base_dir, num_images)
        processor.process_all_methods()

    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
