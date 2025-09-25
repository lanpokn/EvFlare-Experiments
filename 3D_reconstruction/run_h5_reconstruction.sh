#!/bin/bash
# 所有H5文件重建脚本的快速调用入口
# 
# 使用方法:
# ./run_h5_reconstruction.sh <dataset_name>
# 
# 示例:
# ./run_h5_reconstruction.sh lego2   # 处理lego2下所有H5文件
# ./run_h5_reconstruction.sh ship    # 处理ship下所有H5文件

if [ $# -eq 0 ]; then
    echo "❌ 错误: 请指定数据集名称"
    echo "使用方法: ./run_h5_reconstruction.sh <dataset_name>"
    echo "示例: ./run_h5_reconstruction.sh lego2"
    exit 1
fi

DATASET_NAME=$1

echo "🔥 启动 $DATASET_NAME 所有H5文件重建..."
echo "激活Umain2环境并运行Python脚本..."

# 激活conda环境并运行Python脚本
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate Umain2 && \
python process_additional_h5_files.py "$DATASET_NAME"

echo "✅ $DATASET_NAME 所有H5文件处理完成!"