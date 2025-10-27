#!/bin/bash
# 快速启动脚本：集成EFR、PFD-A、PFD-B、UNet新权重的完整3D重建实验
# 作者：Claude
# 日期：2025-10-27

set -e  # 遇到错误立即退出

# ==================== 配置区 ====================
DATASET="lego2"  # 数据集名称（根据实际修改）
UNET_DIR="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main"
EFR_DIR="${UNET_DIR}/ext/EFR-main"
PFD_DIR="${UNET_DIR}/ext/PFD"
PROJECT_DIR="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction"
INPUT_H5="${PROJECT_DIR}/datasets/${DATASET}/events_h5/${DATASET}_sequence_new.h5"
OUTPUT_DIR="${PROJECT_DIR}/datasets/${DATASET}/events_h5"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==================== 函数定义 ====================

print_header() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

print_info() {
    echo -e "${YELLOW}[INFO] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "文件不存在: $1"
        return 1
    else
        print_info "文件存在: $1"
        return 0
    fi
}

# ==================== 主流程 ====================

print_header "🚀 事件相机3D重建完整实验 - 快速启动脚本"

# 步骤0: 环境检查
print_header "步骤0: 环境检查"
print_info "当前工作目录: $(pwd)"
print_info "数据集: ${DATASET}"

# 检查conda环境
if ! command -v conda &> /dev/null; then
    print_error "conda未安装或未在PATH中"
    exit 1
fi

# 激活Umain2环境
print_info "激活Umain2环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Umain2

# 验证输入H5文件
if ! check_file "$INPUT_H5"; then
    print_error "输入H5文件不存在，请先运行阶段1生成基础H5文件"
    exit 1
fi

print_success "环境检查完成"

# ==================== 阶段2: 多方法事件处理 ====================
print_header "阶段2: 多方法事件处理"

# 步骤2.1: EFR处理
print_info "2.1 执行EFR处理..."
cd "$EFR_DIR"

# 检查可执行文件
if [ ! -f "build/comb_filter_batch" ]; then
    print_info "EFR可执行文件不存在，开始编译..."
    mkdir -p build && cd build
    cmake .. && make
    cd ..
fi

# 执行EFR处理
python batch_efr_processor.py \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --base_frequency 50 \
    --rho1 0.6 \
    --delta_t 10000

if check_file "${OUTPUT_DIR}/${DATASET}_sequence_new_EFR.h5"; then
    print_success "EFR处理完成"
else
    print_error "EFR处理失败"
fi

cd "$PROJECT_DIR"

# 步骤2.2: PFD-A处理
print_info "2.2 执行PFD-A处理..."
cd "$PFD_DIR"

# 检查可执行文件
if [ ! -f "build_wsl/PFDs_WSL" ]; then
    print_info "PFD可执行文件不存在，开始编译..."
    mkdir -p build_wsl && cd build_wsl
    cmake -DCMAKE_BUILD_TYPE=Release .. && make
    cd ..
fi

# 执行PFD-A处理
python batch_pfd_processor.py \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --score_select 1 \
    --delta_t0 20000 \
    --delta_t 20000

if check_file "${OUTPUT_DIR}/${DATASET}_sequence_new_PFDA.h5"; then
    print_success "PFD-A处理完成"
else
    print_error "PFD-A处理失败"
fi

cd "$PROJECT_DIR"

# 步骤2.3: PFD-B处理
print_info "2.3 执行PFD-B处理..."
cd "$PFD_DIR"

# 执行PFD-B处理
python batch_pfd_processor.py \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --score_select 0 \
    --delta_t0 20000 \
    --delta_t 20000

if check_file "${OUTPUT_DIR}/${DATASET}_sequence_new_PFDB.h5"; then
    print_success "PFD-B处理完成"
else
    print_error "PFD-B处理失败"
fi

cd "$PROJECT_DIR"

# 步骤2.4: UNet新权重处理
print_info "2.4 执行UNet新权重处理..."
cd "$UNET_DIR"

# 检查新权重
NEW_CHECKPOINT="checkpoints/event_voxel_deflare_physics_noRandom_noTen_method/checkpoint_epoch_0031_iter_040000.pth"
if ! check_file "$NEW_CHECKPOINT"; then
    print_error "新权重文件不存在: $NEW_CHECKPOINT"
    print_info "跳过UNet新权重处理"
else
    # 使用新的inference配置
    python main.py inference \
        --config configs/inference_config_new_checkpoint.yaml \
        --input "$INPUT_H5" \
        --output "${OUTPUT_DIR}/${DATASET}_sequence_new_UnetNew.h5"

    if check_file "${OUTPUT_DIR}/${DATASET}_sequence_new_UnetNew.h5"; then
        print_success "UNet新权重处理完成"
    else
        print_error "UNet新权重处理失败"
    fi
fi

cd "$PROJECT_DIR"

# 步骤2.5: 验证所有H5文件
print_info "2.5 验证所有H5文件..."
ls -lh "${OUTPUT_DIR}/"*.h5

print_success "阶段2完成：多方法事件处理"

# ==================== 阶段3: EVREAL多方法重建 ====================
print_header "阶段3: EVREAL多方法重建"

print_info "执行批量H5重建（处理所有H5文件）..."
python process_additional_h5_files.py "$DATASET"

# 统计重建图像
TOTAL_IMAGES=$(find "datasets/${DATASET}/reconstruction_"* -name "*.png" 2>/dev/null | wc -l)
print_info "重建图像总数: ${TOTAL_IMAGES}"

print_success "阶段3完成：EVREAL多方法重建"

# ==================== 完成总结 ====================
print_header "🎉 阶段2-3完成！"

echo "下一步操作："
echo ""
echo "【Windows环境 - 3DGS训练】"
echo "1. 切换到Windows PowerShell"
echo "2. cd E:\\BaiduSyncdisk\\2025\\event_flick_flare\\experiments\\3D_reconstruction"
echo "3. conda activate gaussian_splatting"
echo "4. python generate_json_configs.py ${DATASET} spade_e2vid"
echo "5. train_3dgs_batch.bat ${DATASET} spade_e2vid"
echo ""
echo "【Windows环境 - 3DGS渲染评估】"
echo "6. python render_and_evaluate.py --dataset ${DATASET} --method spade_e2vid --weights-dir \"gaussian-splatting/output\""
echo ""

print_info "详细教程请参考: COMPLETE_RECONSTRUCTION_TUTORIAL.md"
