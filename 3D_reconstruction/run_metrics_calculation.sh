#!/bin/bash
# 重建图像质量指标计算脚本的快速调用入口
# 
# 使用方法:
# ./run_metrics_calculation.sh <dataset_name>
# 
# 示例:
# ./run_metrics_calculation.sh lego2
# ./run_metrics_calculation.sh ship

if [ $# -eq 0 ]; then
    echo "❌ 错误: 请指定数据集名称"
    echo "使用方法: ./run_metrics_calculation.sh <dataset_name>"
    echo "示例: ./run_metrics_calculation.sh lego2"
    exit 1
fi

DATASET_NAME=$1

echo "🔥 启动 $DATASET_NAME 重建图像质量指标计算..."
echo "激活Umain2环境并运行Python脚本..."

# 激活conda环境并运行Python脚本
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate Umain2 && \
python calculate_reconstruction_metrics.py "$DATASET_NAME"

echo "✅ $DATASET_NAME 指标计算完成!"