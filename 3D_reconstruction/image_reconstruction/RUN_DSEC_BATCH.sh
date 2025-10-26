#!/bin/bash
# DSEC数据集批量重建 - 支持断点续存
#
# 功能特性:
# - 断点续存: 可从中断处继续，自动跳过已完成任务
# - 新处理顺序: 先处理同名H5的所有方法，再处理下一个H5
# - 进度跟踪: .batch_progress.json记录已完成任务

echo "============================================================"
echo "DSEC数据集批量H5重建 (支持断点续存)"
echo "============================================================"

# 激活Umain2环境
echo "🔧 激活Umain2环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Umain2

# 检查环境
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
echo "当前环境: $CURRENT_ENV"

if [ "$CURRENT_ENV" != "Umain2" ]; then
    echo "❌ 错误: 未成功激活Umain2环境"
    exit 1
fi

# 设置路径
DSEC_DATA="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data"
OUTPUT_DIR="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data_reconstructed"
NUM_IMAGES=40  # 每个H5生成40张图像

echo ""
echo "配置:"
echo "  输入目录: $DSEC_DATA"
echo "  输出目录: $OUTPUT_DIR"
echo "  图像数量: $NUM_IMAGES 张/H5文件"
echo "  采样间隔: 每5个H5取1个"
echo "  处理顺序: 先处理同名H5的所有方法"
echo "  断点续存: 支持 (进度文件: $OUTPUT_DIR/.batch_progress.json)"
echo ""

# 检查进度文件
PROGRESS_FILE="$OUTPUT_DIR/.batch_progress.json"
if [ -f "$PROGRESS_FILE" ]; then
    echo "📂 检测到进度文件，将从上次中断处继续"
    # 显示已完成任务数
    COMPLETED=$(grep -o "\"total_completed\":" "$PROGRESS_FILE" | wc -l)
    if [ $COMPLETED -gt 0 ]; then
        echo "  已完成任务数: $(grep "total_completed" "$PROGRESS_FILE" | grep -o '[0-9]*')"
    fi
    echo ""
fi

# 确认执行
read -p "是否开始批量重建? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 运行批量重建
echo ""
echo "🚀 开始批量重建..."
echo "提示: 可随时按Ctrl+C中断，下次运行将自动从中断处继续"
echo ""

python batch_dsec_reconstruction.py "$DSEC_DATA" "$OUTPUT_DIR" "$NUM_IMAGES"

EXITCODE=$?

echo ""
if [ $EXITCODE -eq 0 ]; then
    echo "✅ 脚本执行完成！"
else
    echo "⚠️  脚本执行被中断 (退出代码: $EXITCODE)"
    echo "提示: 下次运行将自动从中断处继续"
fi
echo "============================================================"
