#!/usr/bin/env python3
"""
通用视频生成脚本 - 适用于任意数据集的图像序列
支持原始图像和3DGS渲染结果的视频生成，自动转换为灰度图
使用ffmpeg进行视频编码以避免OpenCV依赖问题
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
from PIL import Image
import tempfile
import shutil

def convert_to_grayscale_pil(image_path, output_path):
    """使用PIL将图像转换为灰度图"""
    with Image.open(image_path) as img:
        if img.mode != 'L':
            gray_img = img.convert('L')
            # 转换为RGB模式以便ffmpeg处理
            rgb_img = Image.new('RGB', gray_img.size)
            rgb_img.paste(gray_img)
            rgb_img.save(output_path)
        else:
            # 已经是灰度图，转换为RGB
            rgb_img = Image.new('RGB', img.size)
            rgb_img.paste(img)
            rgb_img.save(output_path)

def create_video_from_images(image_dir, output_video_path, fps=10, force_grayscale=False):
    """
    使用ffmpeg从图像序列创建视频
    
    Args:
        image_dir (str): 包含图像的目录
        output_video_path (str): 输出视频文件路径
        fps (int): 输出视频的帧率
        force_grayscale (bool): 是否强制转换为灰度图
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"错误: 目录 {image_dir} 不存在")
        return False
    
    # 获取排序后的PNG文件列表
    image_files = sorted([f for f in image_dir.glob("*.png")])
    
    if not image_files:
        print(f"错误: 在 {image_dir} 中未找到PNG文件")
        return False
    
    print(f"在 {image_dir} 中找到 {len(image_files)} 张图像")
    
    # 检查第一张图像
    try:
        with Image.open(image_files[0]) as img:
            width, height = img.size
            print(f"图像尺寸: {width}x{height}")
    except Exception as e:
        print(f"错误: 无法读取第一张图像 {image_files[0]}: {e}")
        return False
    
    # 创建临时目录处理图像
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 如果需要转换为灰度图，先处理所有图像
        if force_grayscale:
            print("正在转换为灰度图...")
            for i, img_path in enumerate(image_files):
                temp_img_path = temp_path / f"{i:06d}.png"
                try:
                    convert_to_grayscale_pil(img_path, temp_img_path)
                except Exception as e:
                    print(f"警告: 无法处理图像 {img_path}: {e}")
                    continue
                
                if (i + 1) % 50 == 0:
                    print(f"  已处理 {i + 1}/{len(image_files)} 张图像")
            
            # 使用临时目录中的图像
            input_pattern = str(temp_path / "%06d.png")
        else:
            # 创建符号链接以保持文件名顺序
            for i, img_path in enumerate(image_files):
                temp_img_path = temp_path / f"{i:06d}.png"
                try:
                    shutil.copy2(img_path, temp_img_path)
                except Exception as e:
                    print(f"警告: 无法复制图像 {img_path}: {e}")
                    continue
            
            input_pattern = str(temp_path / "%06d.png")
        
        # 使用ffmpeg创建视频
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y 覆盖输出文件
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',  # 使用H.264编码
            '-pix_fmt', 'yuv420p',  # 兼容性好的像素格式
            '-crf', '18',  # 高质量编码
            str(output_video_path)
        ]
        
        try:
            print("正在使用ffmpeg创建视频...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print(f"视频创建成功: {output_video_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg错误: {e}")
            print(f"stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            print("错误: 未找到ffmpeg命令。请确保已安装ffmpeg。")
            return False

def process_dataset(dataset_name, fps=10, force_grayscale=False, output_dir="videos"):
    """
    处理指定数据集的所有图像序列
    
    Args:
        dataset_name (str): 数据集名称（如lego2）
        fps (int): 视频帧率
        force_grayscale (bool): 是否强制转换为灰度图
        output_dir (str): 输出目录
    """
    base_dir = Path(__file__).parent / "datasets" / dataset_name
    
    if not base_dir.exists():
        print(f"错误: 数据集目录 {base_dir} 不存在")
        return False
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    print(f"正在处理数据集: {dataset_name}")
    print(f"输出目录: {output_path.absolute()}")
    print(f"视频帧率: {fps} FPS")
    print(f"灰度转换: {'是' if force_grayscale else '否'}")
    print("=" * 50)
    
    # 1. 处理原始图像 - train和test
    for split in ["train", "test"]:
        image_dir = base_dir / split
        if image_dir.exists():
            total_count += 1
            output_video = output_path / f"{dataset_name}_{split}{'_grayscale' if force_grayscale else ''}.mp4"
            
            print(f"\n--- 处理 {split} 序列 ---")
            print(f"输入: {image_dir}")
            print(f"输出: {output_video}")
            
            if create_video_from_images(image_dir, output_video, fps, force_grayscale):
                success_count += 1
            else:
                print(f"创建 {split} 视频失败")
    
    # 2. 处理3DGS渲染结果
    renders_dir = base_dir / "3dgs_results" / "final_renders"
    if renders_dir.exists():
        for render_method in renders_dir.iterdir():
            if render_method.is_dir():
                total_count += 1
                # 提取方法名（去掉数据集前缀）
                method_name = render_method.name.replace(f"{dataset_name}_", "")
                output_video = output_path / f"{dataset_name}_3dgs_{method_name}.mp4"
                
                print(f"\n--- 处理 3DGS渲染: {method_name} ---")
                print(f"输入: {render_method}")
                print(f"输出: {output_video}")
                
                if create_video_from_images(render_method, output_video, fps, False):  # 3DGS渲染结果通常已经是灰度图
                    success_count += 1
                else:
                    print(f"创建 {method_name} 渲染视频失败")
    
    print(f"\n{'='*20} 总结 {'='*20}")
    print(f"成功创建 {success_count}/{total_count} 个视频")
    
    if success_count == total_count:
        print("所有视频创建成功！")
        return True
    else:
        print(f"有 {total_count - success_count} 个视频创建失败")
        return False

def main():
    parser = argparse.ArgumentParser(description="为数据集图像序列创建视频")
    parser.add_argument("dataset", help="数据集名称（如lego2）")
    parser.add_argument("--fps", type=int, default=10, help="视频帧率（默认：10）")
    parser.add_argument("--grayscale", action="store_true", help="将原始图像转换为灰度图")
    parser.add_argument("--output-dir", type=str, default="videos", help="输出目录（默认：videos）")
    
    args = parser.parse_args()
    
    success = process_dataset(
        dataset_name=args.dataset,
        fps=args.fps,
        force_grayscale=args.grayscale,
        output_dir=args.output_dir
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())