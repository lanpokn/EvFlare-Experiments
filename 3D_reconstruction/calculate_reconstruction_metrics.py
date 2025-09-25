#!/usr/bin/env python3
"""
重建图像质量指标计算脚本
计算所有重建结果与test/train真值的PSNR、SSIM、LPIPS指标

功能说明：
- 对N个H5文件的重建结果分别与test和train真值计算指标
- 每个重建方法产生2个指标结果（vs test, vs train）
- 计算200张图片的平均指标值
- 输出最佳结果排名和详细指标文件

使用方法:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && python calculate_reconstruction_metrics.py <dataset_name>

示例:
python calculate_reconstruction_metrics.py lego2

Author: Claude Code Assistant
Date: 2025-09-25
"""

import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm

# 图像质量指标计算
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:
    print("❌ 缺少scikit-image库，请安装：pip install scikit-image")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ 缺少torch库")
    TORCH_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("⚠️  lpips库不可用，将跳过LPIPS计算")
    LPIPS_AVAILABLE = False


class ReconstructionMetricsCalculator:
    """重建图像质量指标计算器"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_dir = Path("datasets") / dataset_name
        
        # 验证目录存在
        self._validate_directories()
        
        # 初始化LPIPS模型
        if TORCH_AVAILABLE and LPIPS_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"使用设备: {self.device}")
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_enabled = True
            except Exception as e:
                print(f"⚠️  LPIPS模型初始化失败: {e}")
                self.lpips_enabled = False
        else:
            self.device = 'cpu'
            self.lpips_model = None
            self.lpips_enabled = False
        
        # 结果存储
        self.results = []
    
    def _validate_directories(self):
        """验证必要目录存在"""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")
        
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        
        if not self.train_dir.exists():
            raise FileNotFoundError(f"训练真值目录不存在: {self.train_dir}")
        if not self.test_dir.exists():
            raise FileNotFoundError(f"测试真值目录不存在: {self.test_dir}")
    
    def get_reconstruction_directories(self) -> List[Path]:
        """获取所有重建目录"""
        reconstruction_dirs = []
        
        for item in self.dataset_dir.iterdir():
            if item.is_dir() and item.name.startswith("reconstruction"):
                # 跳过旧的reconstruction目录，只处理有后缀的重建目录
                if item.name == "reconstruction":
                    print(f"  跳过旧的重建目录: {item.name}")
                    continue
                reconstruction_dirs.append(item)
        
        reconstruction_dirs.sort()
        print(f"找到 {len(reconstruction_dirs)} 个重建目录:")
        for rec_dir in reconstruction_dirs:
            print(f"  - {rec_dir.name}")
        
        return reconstruction_dirs
    
    def get_method_directories(self, reconstruction_dir: Path) -> List[Path]:
        """获取重建目录下的所有方法目录"""
        method_dirs = []
        
        for item in reconstruction_dir.iterdir():
            if item.is_dir() and item.name.startswith("evreal_"):
                method_dirs.append(item)
        
        method_dirs.sort()
        return method_dirs
    
    def load_images(self, image_dir: Path, expected_count: int = 200) -> List[np.ndarray]:
        """加载图像序列"""
        images = []
        missing_count = 0
        
        for i in range(1, expected_count + 1):
            img_path = image_dir / f"{i:04d}.png"
            if not img_path.exists():
                missing_count += 1
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                missing_count += 1
                continue
            
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        if missing_count > 0:
            print(f"  ⚠ 缺少 {missing_count} 张图像")
        print(f"  加载了 {len(images)}/{expected_count} 张图像")
        return images
    
    def calculate_psnr_batch(self, pred_images: List[np.ndarray], gt_images: List[np.ndarray]) -> float:
        """批量计算PSNR"""
        if len(pred_images) != len(gt_images):
            print(f"⚠ 图像数量不匹配: {len(pred_images)} vs {len(gt_images)}")
            min_len = min(len(pred_images), len(gt_images))
            pred_images = pred_images[:min_len]
            gt_images = gt_images[:min_len]
        
        psnr_values = []
        for pred, gt in zip(pred_images, gt_images):
            if pred.shape != gt.shape:
                print(f"⚠ 图像尺寸不匹配: {pred.shape} vs {gt.shape}")
                continue
            
            try:
                psnr_val = psnr(gt, pred, data_range=255)
                if not np.isnan(psnr_val) and not np.isinf(psnr_val):
                    psnr_values.append(psnr_val)
            except Exception as e:
                print(f"⚠ PSNR计算失败: {e}")
                continue
        
        return np.mean(psnr_values) if psnr_values else 0.0
    
    def calculate_ssim_batch(self, pred_images: List[np.ndarray], gt_images: List[np.ndarray]) -> float:
        """批量计算SSIM"""
        if len(pred_images) != len(gt_images):
            min_len = min(len(pred_images), len(gt_images))
            pred_images = pred_images[:min_len]
            gt_images = gt_images[:min_len]
        
        ssim_values = []
        for pred, gt in zip(pred_images, gt_images):
            if pred.shape != gt.shape:
                continue
            
            try:
                # 转换为灰度图计算SSIM
                pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY) if len(pred.shape) == 3 else pred
                gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY) if len(gt.shape) == 3 else gt
                
                ssim_val = ssim(gt_gray, pred_gray, data_range=255)
                if not np.isnan(ssim_val) and not np.isinf(ssim_val):
                    ssim_values.append(ssim_val)
            except Exception as e:
                print(f"⚠ SSIM计算失败: {e}")
                continue
        
        return np.mean(ssim_values) if ssim_values else 0.0
    
    def calculate_lpips_batch(self, pred_images: List[np.ndarray], gt_images: List[np.ndarray]) -> float:
        """批量计算LPIPS"""
        if not self.lpips_enabled:
            return 1.0  # 返回默认值，表示不可用
        
        if len(pred_images) != len(gt_images):
            min_len = min(len(pred_images), len(gt_images))
            pred_images = pred_images[:min_len]
            gt_images = gt_images[:min_len]
        
        lpips_values = []
        batch_size = 10  # 批量处理避免内存问题
        
        for i in range(0, len(pred_images), batch_size):
            batch_pred = pred_images[i:i+batch_size]
            batch_gt = gt_images[i:i+batch_size]
            
            try:
                # 转换为tensor格式 [batch, 3, H, W], 范围[-1, 1]
                pred_tensors = []
                gt_tensors = []
                
                for pred, gt in zip(batch_pred, batch_gt):
                    if pred.shape != gt.shape:
                        continue
                    
                    # 归一化到[-1, 1]
                    pred_norm = (pred.astype(np.float32) / 255.0) * 2.0 - 1.0
                    gt_norm = (gt.astype(np.float32) / 255.0) * 2.0 - 1.0
                    
                    # HWC -> CHW
                    pred_tensor = torch.from_numpy(pred_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    gt_tensor = torch.from_numpy(gt_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    
                    pred_tensors.append(pred_tensor)
                    gt_tensors.append(gt_tensor)
                
                if pred_tensors and gt_tensors:
                    pred_batch = torch.cat(pred_tensors, dim=0)
                    gt_batch = torch.cat(gt_tensors, dim=0)
                    
                    with torch.no_grad():
                        lpips_batch = self.lpips_model(pred_batch, gt_batch)
                        lpips_values.extend(lpips_batch.cpu().numpy().flatten())
                
            except Exception as e:
                print(f"⚠ LPIPS计算失败: {e}")
                continue
        
        return np.mean(lpips_values) if lpips_values else 1.0
    
    def calculate_metrics_for_pair(self, pred_dir: Path, gt_dir: Path, pair_name: str) -> Dict:
        """计算一对图像目录的指标"""
        print(f"  计算指标: {pair_name}")
        
        # 加载图像
        pred_images = self.load_images(pred_dir)
        gt_images = self.load_images(gt_dir)
        
        if not pred_images or not gt_images:
            print(f"  ❌ 图像加载失败: {pair_name}")
            return {
                "pair_name": pair_name,
                "psnr": 0.0,
                "ssim": 0.0, 
                "lpips": 1.0,
                "image_count": 0,
                "error": "图像加载失败"
            }
        
        # 计算指标
        start_time = time.time()
        
        psnr_val = self.calculate_psnr_batch(pred_images, gt_images)
        ssim_val = self.calculate_ssim_batch(pred_images, gt_images)
        lpips_val = self.calculate_lpips_batch(pred_images, gt_images)
        
        elapsed_time = time.time() - start_time
        
        result = {
            "pair_name": pair_name,
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "lpips": float(lpips_val),
            "image_count": min(len(pred_images), len(gt_images)),
            "calculation_time_s": float(elapsed_time)
        }
        
        print(f"    PSNR: {psnr_val:.3f}, SSIM: {ssim_val:.3f}, LPIPS: {lpips_val:.3f} ({elapsed_time:.1f}s)")
        
        return result
    
    def process_all_reconstructions(self):
        """处理所有重建结果"""
        reconstruction_dirs = self.get_reconstruction_directories()
        
        if not reconstruction_dirs:
            print("❌ 未找到重建目录")
            return
        
        print(f"\n🚀 开始计算 {len(reconstruction_dirs)} 个重建目录的指标...")
        
        # 遍历每个重建目录
        for rec_dir in tqdm(reconstruction_dirs, desc="处理重建目录"):
            rec_name = rec_dir.name  # 如：reconstruction_original
            
            print(f"\n{'='*60}")
            print(f"处理: {rec_name}")
            print(f"{'='*60}")
            
            # 获取方法目录
            method_dirs = self.get_method_directories(rec_dir)
            
            if not method_dirs:
                print(f"  ⚠ {rec_name} 中未找到方法目录")
                continue
            
            print(f"  找到 {len(method_dirs)} 个方法目录")
            
            # 遍历每个方法
            for method_dir in method_dirs:
                method_name = method_dir.name.replace("evreal_", "")  # 如：e2vid
                
                print(f"\n--- 方法: {method_name} ---")
                
                # 分别与train和test计算指标
                for gt_type in ["train", "test"]:
                    gt_dir = self.dataset_dir / gt_type
                    pair_name = f"{rec_name}_{method_name}_vs_{gt_type}"
                    
                    result = self.calculate_metrics_for_pair(method_dir, gt_dir, pair_name)
                    
                    # 添加元信息
                    result.update({
                        "dataset_name": self.dataset_name,
                        "reconstruction_name": rec_name,
                        "method_name": method_name,
                        "ground_truth_type": gt_type,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    self.results.append(result)
        
        print(f"\n✅ 完成所有指标计算，共 {len(self.results)} 个结果")
    
    def save_results(self):
        """保存结果到文件"""
        if not self.results:
            print("❌ 没有结果可保存")
            return
        
        # 创建输出目录
        output_dir = self.dataset_dir / "metrics_results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果JSON
        json_file = output_dir / f"detailed_metrics_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV表格
        csv_file = output_dir / f"metrics_summary_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"✅ 详细结果已保存: {json_file}")
        print(f"✅ CSV表格已保存: {csv_file}")
        
        return json_file, csv_file
    
    def analyze_best_results(self):
        """分析最佳结果"""
        if not self.results:
            print("❌ 没有结果可分析")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*60}")
        print("🏆 最佳结果分析")
        print(f"{'='*60}")
        
        # 按方法和真值类型分组分析
        for gt_type in ["train", "test"]:
            print(f"\n--- 真值类型: {gt_type} ---")
            
            gt_data = df[df['ground_truth_type'] == gt_type]
            
            if gt_data.empty:
                print(f"  没有 {gt_type} 的结果")
                continue
            
            # 按方法分组
            methods = gt_data['method_name'].unique()
            
            for method in sorted(methods):
                method_data = gt_data[gt_data['method_name'] == method]
                
                if method_data.empty:
                    continue
                
                print(f"\n  方法: {method}")
                
                # 找出每个指标的最佳结果
                best_psnr = method_data.loc[method_data['psnr'].idxmax()]
                best_ssim = method_data.loc[method_data['ssim'].idxmax()]
                best_lpips = method_data.loc[method_data['lpips'].idxmin()]  # LPIPS越小越好
                
                print(f"    最佳PSNR: {best_psnr['psnr']:.3f} ({best_psnr['reconstruction_name']})")
                print(f"    最佳SSIM: {best_ssim['ssim']:.3f} ({best_ssim['reconstruction_name']})")
                print(f"    最佳LPIPS: {best_lpips['lpips']:.3f} ({best_lpips['reconstruction_name']})")
        
        # 全局最佳结果
        print(f"\n🎯 全局最佳结果:")
        
        overall_best_psnr = df.loc[df['psnr'].idxmax()]
        overall_best_ssim = df.loc[df['ssim'].idxmax()]
        overall_best_lpips = df.loc[df['lpips'].idxmin()]
        
        print(f"  最佳PSNR: {overall_best_psnr['psnr']:.3f}")
        print(f"    - {overall_best_psnr['reconstruction_name']} + {overall_best_psnr['method_name']} vs {overall_best_psnr['ground_truth_type']}")
        
        print(f"  最佳SSIM: {overall_best_ssim['ssim']:.3f}")
        print(f"    - {overall_best_ssim['reconstruction_name']} + {overall_best_ssim['method_name']} vs {overall_best_ssim['ground_truth_type']}")
        
        print(f"  最佳LPIPS: {overall_best_lpips['lpips']:.3f}")
        print(f"    - {overall_best_lpips['reconstruction_name']} + {overall_best_lpips['method_name']} vs {overall_best_lpips['ground_truth_type']}")


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("❌ 错误: 请指定数据集名称")
        print("使用方法: python calculate_reconstruction_metrics.py <dataset_name>")
        print("示例: python calculate_reconstruction_metrics.py lego2")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    print(f"🔥 {dataset_name} 重建图像质量指标计算")
    print("=" * 60)
    
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"当前conda环境: {conda_env}")
    if conda_env != 'Umain2':
        print("⚠️  警告: 未在Umain2环境中运行!")
        print("请使用: source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2")
    
    try:
        # 创建计算器并运行
        calculator = ReconstructionMetricsCalculator(dataset_name)
        
        # 处理所有重建结果
        calculator.process_all_reconstructions()
        
        # 保存结果
        json_file, csv_file = calculator.save_results()
        
        # 分析最佳结果
        calculator.analyze_best_results()
        
        print(f"\n🎯 指标计算完成!")
        print(f"📊 结果文件: {json_file}")
        print(f"📊 CSV文件: {csv_file}")
        
    except Exception as e:
        print(f"❌ 脚本执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()