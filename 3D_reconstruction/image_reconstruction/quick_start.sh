#!/bin/bash
# 快速启动脚本 - 事件相机图像重建
#
# 用法: ./quick_start.sh <dataset_name>
# 示例: ./quick_start.sh lego2

if [ $# -lt 1 ]; then
    echo "❌ 错误: 请指定数据集名称"
    echo "用法: ./quick_start.sh <dataset_name>"
    echo "示例: ./quick_start.sh lego2"
    exit 1
fi

DATASET_NAME=$1

echo "=========================================="
echo "事件相机图像重建 - 快速启动"
echo "数据集: $DATASET_NAME"
echo "=========================================="

# 激活Umain2环境
echo "🔧 激活Umain2环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Umain2

# 检查环境
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
echo "当前环境: $CURRENT_ENV"

if [ "$CURRENT_ENV" != "Umain2" ]; then
    echo "⚠️  警告: 未成功激活Umain2环境"
    exit 1
fi

# 运行H5重建
echo ""
echo "🚀 开始H5到图像重建..."
python h5_to_images.py $DATASET_NAME

echo ""
echo "✅ 完成！"
echo "=========================================="
