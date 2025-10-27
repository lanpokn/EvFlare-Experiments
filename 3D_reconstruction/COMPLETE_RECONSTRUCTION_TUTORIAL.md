# 🚀 事件相机3D重建完整实验教程

**最后更新**: 2025-10-27
**适用场景**: 集成EFR、PFD-A、PFD-B新方法 + 新UNet权重的完整3D重建实验

---

## 📋 目录

1. [实验概述](#实验概述)
2. [环境准备](#环境准备)
3. [完整Pipeline流程](#完整pipeline流程)
4. [详细操作步骤](#详细操作步骤)
5. [常见问题](#常见问题)
6. [预期输出](#预期输出)

---

## 实验概述

### 🎯 实验目标
从原始数据集出发，经过DVS仿真、多种事件处理方法（原始、Unet、UnetNew、EFR、PFD-A、PFD-B）、EVREAL重建，最终进行3DGS训练和评估，对比不同方法的重建质量。

### 📊 技术路线图
```
原始数据集(xxx_flare + xxx_normal)
    ↓
【阶段1】基础事件数据生成 (WSL)
    ↓
原始H5文件 (xxx_sequence_new.h5)
    ↓
【阶段2】多方法事件处理 (WSL + 外部)
    ├── EFR处理 → xxx_sequence_new_EFR.h5
    ├── PFD-A处理 → xxx_sequence_new_PFDA.h5
    ├── PFD-B处理 → xxx_sequence_new_PFDB.h5
    ├── Unet处理 → xxx_sequence_new_Unet.h5 (已有)
    ├── UnetNew处理 → xxx_sequence_new_UnetNew.h5 (新权重)
    └── Unetsimple处理 → xxx_sequence_new_Unetsimple.h5 (已有)
    ↓
【阶段3】EVREAL多方法重建 (WSL)
    ├── reconstruction_original/ (8种方法 × 200张)
    ├── reconstruction_EFR/ (8种方法 × 200张)
    ├── reconstruction_PFDA/ (8种方法 × 200张)
    ├── reconstruction_PFDB/ (8种方法 × 200张)
    ├── reconstruction_Unet/ (8种方法 × 200张)
    ├── reconstruction_UnetNew/ (8种方法 × 200张)
    └── reconstruction_Unetsimple/ (8种方法 × 200张)
    ↓
【阶段4】3DGS批量训练 (Windows)
    ↓
【阶段5】3DGS渲染评估 (Windows)
    ↓
最终对比报告
```

### 🔢 数据规模预估
- **事件数据**: 7个H5文件 (每个100-500万事件，~50-200MB)
- **重建图像**: 7×8×200 = 11,200张图像 (~5-10GB)
- **3DGS训练**: 预计8-10个配置 (每个~50MB权重)
- **渲染结果**: 8-10×200 = 1,600-2,000张图像 (~1-2GB)

---

## 环境准备

### 🐧 WSL环境 (数据处理和重建)

#### 环境1: Umain2 (主要环境)
```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

# 验证环境
python -c "import torch; import h5py; import numpy; print('Umain2环境正常')"
```

**用途**:
- DVS事件仿真
- 格式转换
- EVREAL重建
- UNet推理（新权重）

#### 环境2: EFR/PFD专用环境
```bash
# 检查EFR可执行文件
ls -la /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/EFR-main/build/

# 检查PFD可执行文件
ls -la /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/PFD/build_wsl/
```

**用途**:
- EFR炫光去除
- PFD-A/PFD-B炫光去除

### 🪟 Windows环境 (3DGS训练)

#### 环境: 3DGS训练环境
```powershell
# 激活conda环境 (根据实际环境名称)
conda activate gaussian_splatting  # 或者你的3DGS环境名称

# 验证CUDA和PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 完整Pipeline流程

### 🔄 Pipeline架构总览

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: 基础事件数据生成 (WSL - Umain2)                     │
│ ● 数据集合并 → DVS仿真 → 原始H5生成                        │
│ 输入: xxx_flare/ + xxx_normal/                              │
│ 输出: datasets/xxx/events_h5/xxx_sequence_new.h5            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段2: 多方法事件处理 (WSL - 外部)                         │
│ ● EFR处理 (batch_efr_processor.py)                          │
│ ● PFD-A处理 (batch_pfd_processor.py --score_select 1)       │
│ ● PFD-B处理 (batch_pfd_processor.py --score_select 0)       │
│ ● UnetNew处理 (main.py --mode inference --checkpoint 40000) │
│ 输出: xxx_sequence_new_EFR.h5, xxx_sequence_new_PFDA.h5等   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段3: EVREAL多方法重建 (WSL - Umain2)                     │
│ ● 批量H5重建 (process_additional_h5_files.py)               │
│ ● 8种EVREAL方法 × 7个H5文件                                 │
│ 输出: reconstruction_xxx/ 目录 (7×8×200 = 11,200张图像)     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段4: 3DGS批量训练 (Windows)                               │
│ ● 配置生成 (generate_json_configs.py)                       │
│ ● 批量训练 (train_3dgs_batch.bat)                           │
│ 输出: gaussian-splatting/output/ (8-10个配置)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段5: 3DGS渲染评估 (Windows)                               │
│ ● 渲染+评估 (render_and_evaluate.py)                        │
│ 输出: datasets/xxx/3dgs_results/ (渲染图像+评估报告)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 详细操作步骤

### 📍 前提条件确认

**如果你已有原始H5文件（如lego2_sequence_new.h5），可以直接跳到阶段2。**

**如果从零开始，需要从阶段0开始。**

---

### 阶段0️⃣: 数据集准备 (可选)

**仅当没有原始数据集时需要**

```bash
# 在WSL中执行
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction

# 确认数据集存在
ls datasets/xxx_flare/train/*.png | wc -l  # 应该是200
ls datasets/xxx_normal/train/*.png | wc -l # 应该是200
```

---

### 阶段1️⃣: 基础事件数据生成 (WSL - Umain2)

**目标**: 从原始图像生成基础H5事件文件

#### 步骤1.1: 数据集合并

```bash
# 激活Umain2环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

# 合并数据集
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
python merge_datasets.py

# 验证合并结果
ls datasets/xxx/train/*.png | wc -l  # 应该是200
ls datasets/xxx/test/*.png | wc -l   # 应该是200
```

#### 步骤1.2: DVS事件仿真 + H5生成

```bash
# 完整pipeline执行 (DVS仿真 + 格式转换)
python run_full_pipeline.py

# 验证H5文件生成
ls -lh datasets/xxx/events_h5/xxx_sequence_new.h5
# 预期: 50-200MB的H5文件
```

**🎉 阶段1完成**: 你现在有了基础H5事件文件 `xxx_sequence_new.h5`

---

### 阶段2️⃣: 多方法事件处理 (WSL - 外部方法)

**目标**: 使用不同算法处理原始事件数据，生成多个变体H5文件

#### 📁 目录切换说明
```bash
# 当前位置
pwd  # 应该在 /mnt/e/.../3D_reconstruction

# EFR方法目录
EFR_DIR="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/EFR-main"

# PFD方法目录
PFD_DIR="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/PFD"

# UNet主目录
UNET_DIR="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main"

# 数据集目录
DATASET="lego2"  # 替换为你的数据集名称
INPUT_H5="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/${DATASET}/events_h5/${DATASET}_sequence_new.h5"
OUTPUT_DIR="/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/${DATASET}/events_h5"
```

#### 步骤2.1: EFR处理

```bash
# 进入EFR目录
cd "$EFR_DIR"

# 确认EFR可执行文件存在
ls -la build/comb_filter_batch

# 如果没有可执行文件，需要编译
if [ ! -f "build/comb_filter_batch" ]; then
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

# 验证输出
ls -lh "${OUTPUT_DIR}/${DATASET}_sequence_new_EFR.h5"

# 返回原目录
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 步骤2.2: PFD-A处理

```bash
# 进入PFD目录
cd "$PFD_DIR"

# 确认PFD可执行文件存在
ls -la build_wsl/PFDs_WSL

# 如果没有可执行文件，需要编译
if [ ! -f "build_wsl/PFDs_WSL" ]; then
    mkdir -p build_wsl && cd build_wsl
    cmake -DCMAKE_BUILD_TYPE=Release .. && make
    cd ..
fi

# 执行PFD-A处理 (score_select=1)
python batch_pfd_processor.py \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --score_select 1 \
    --delta_t0 20000 \
    --delta_t 20000

# 验证输出
ls -lh "${OUTPUT_DIR}/${DATASET}_sequence_new_PFDA.h5"

# 返回原目录
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 步骤2.3: PFD-B处理

```bash
# 进入PFD目录
cd "$PFD_DIR"

# 执行PFD-B处理 (score_select=0)
python batch_pfd_processor.py \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --score_select 0 \
    --delta_t0 20000 \
    --delta_t 20000

# 验证输出
ls -lh "${OUTPUT_DIR}/${DATASET}_sequence_new_PFDB.h5"

# 返回原目录
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 步骤2.4: UNet新权重处理 (如果需要)

```bash
# 进入UNet主目录
cd "$UNET_DIR"

# 确认新权重存在
NEW_CHECKPOINT="checkpoints/event_voxel_deflare_physics_noRandom_noTen_method/checkpoint_epoch_0031_iter_040000.pth"
ls -lh "$NEW_CHECKPOINT"

# 执行UNet推理 (使用新权重)
python main.py \
    --mode inference \
    --checkpoint "$NEW_CHECKPOINT" \
    --input_h5 "$INPUT_H5" \
    --output_dir "$OUTPUT_DIR" \
    --output_suffix "_UnetNew"

# 验证输出
ls -lh "${OUTPUT_DIR}/${DATASET}_sequence_new_UnetNew.h5"

# 返回原目录
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 步骤2.5: 验证所有H5文件

```bash
# 查看所有生成的H5文件
ls -lh datasets/${DATASET}/events_h5/*.h5

# 预期输出:
# xxx_sequence_new.h5         (原始)
# xxx_sequence_new_EFR.h5     (EFR处理)
# xxx_sequence_new_PFDA.h5    (PFD-A处理)
# xxx_sequence_new_PFDB.h5    (PFD-B处理)
# xxx_sequence_new_Unet.h5    (Unet处理，已有)
# xxx_sequence_new_UnetNew.h5 (UNet新权重处理)
# xxx_sequence_new_Unetsimple.h5 (Unetsimple处理，已有)
```

**🎉 阶段2完成**: 你现在有了7个不同处理方法的H5文件

---

### 阶段3️⃣: EVREAL多方法重建 (WSL - Umain2)

**目标**: 从7个H5文件生成56种重建结果（7×8种EVREAL方法）

#### 步骤3.1: 批量H5重建

```bash
# 确保在主目录
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction

# 激活Umain2环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

# 执行批量重建（处理所有H5文件）
python process_additional_h5_files.py ${DATASET}

# 这个脚本会自动:
# 1. 扫描 datasets/${DATASET}/events_h5/ 中的所有H5文件
# 2. 对每个H5文件运行EVREAL重建（8种方法）
# 3. 输出到独立的 reconstruction_xxx/ 目录
```

#### 步骤3.2: 监控重建进度

```bash
# 查看重建目录
ls -d datasets/${DATASET}/reconstruction_*/

# 预期输出:
# datasets/xxx/reconstruction/          (原始，可能是旧的)
# datasets/xxx/reconstruction_original/ (原始H5)
# datasets/xxx/reconstruction_EFR/      (EFR H5)
# datasets/xxx/reconstruction_PFDA/     (PFD-A H5)
# datasets/xxx/reconstruction_PFDB/     (PFD-B H5)
# datasets/xxx/reconstruction_Unet/     (Unet H5)
# datasets/xxx/reconstruction_UnetNew/  (UNet新权重 H5)
# datasets/xxx/reconstruction_Unetsimple/ (Unetsimple H5)

# 检查每个目录的重建方法
for dir in datasets/${DATASET}/reconstruction_*/; do
    echo "=== $dir ==="
    ls "$dir"
done

# 预期每个reconstruction_xxx/目录包含:
# evreal_e2vid/
# evreal_firenet/
# evreal_spade_e2vid/
# evreal_ssl_e2vid/
# ... (其他成功的方法)

# 统计重建图像总数
find datasets/${DATASET}/reconstruction_* -name "*.png" | wc -l
# 预期: 约11,200张图像 (7个H5 × 8种方法 × 200张)
```

#### 步骤3.3: 计算重建质量指标 (可选)

```bash
# 计算所有重建结果的PSNR/SSIM/LPIPS指标
python calculate_reconstruction_metrics.py ${DATASET}

# 查看指标结果
cat datasets/${DATASET}/reconstruction_metrics_*.json
```

**🎉 阶段3完成**: 你现在有了7×8=56种重建结果，约11,200张重建图像

---

### 阶段4️⃣: 3DGS批量训练 (Windows)

**目标**: 对原始图像+各种重建图像进行3DGS训练

#### 步骤4.1: 切换到Windows环境

```powershell
# 在Windows PowerShell中执行
cd E:\BaiduSyncdisk\2025\event_flick_flare\experiments\3D_reconstruction

# 激活3DGS环境
conda activate gaussian_splatting  # 或者你的实际环境名称
```

#### 步骤4.2: 生成训练配置

**选择要训练的重建方法**（根据EVREAL重建结果选择）

```powershell
# 示例：使用spade_e2vid重建方法
# 这会生成以下配置:
# - original (原始train图像)
# - spade_e2vid_original (原始H5重建)
# - spade_e2vid_EFR (EFR H5重建)
# - spade_e2vid_PFDA (PFD-A H5重建)
# - spade_e2vid_PFDB (PFD-B H5重建)
# - spade_e2vid_Unet (Unet H5重建)
# - spade_e2vid_UnetNew (UNet新权重H5重建)
# - spade_e2vid_Unetsimple (Unetsimple H5重建)

python generate_json_configs.py lego2 spade_e2vid

# 如果要使用其他重建方法（如et_net），替换最后的参数:
# python generate_json_configs.py lego2 et_net
```

#### 步骤4.3: 批量训练

```batch
REM 批量训练所有配置（只训练，不渲染）
train_3dgs_batch.bat lego2 spade_e2vid

REM 训练时间预估:
REM - 每个配置约30-60分钟（10000 iterations）
REM - 8个配置 × 45分钟 ≈ 6小时
```

#### 步骤4.4: 监控训练进度

```powershell
# 查看训练输出目录
dir gaussian-splatting\output\lego2_*

# 预期输出:
# lego2_original\
# lego2_spade_e2vid_original\
# lego2_spade_e2vid_EFR\
# lego2_spade_e2vid_PFDA\
# lego2_spade_e2vid_PFDB\
# lego2_spade_e2vid_Unet\
# lego2_spade_e2vid_UnetNew\
# lego2_spade_e2vid_Unetsimple\

# 查看权重备份
dir datasets\lego2\3dgs_results\weights\
```

**🎉 阶段4完成**: 你现在有了8个配置的3DGS训练权重

---

### 阶段5️⃣: 3DGS渲染评估 (Windows)

**目标**: 渲染test图像并计算质量指标

#### 步骤5.1: 渲染和评估

```powershell
# 自动渲染所有训练好的模型并计算指标
python render_and_evaluate.py --dataset lego2 --method spade_e2vid --weights-dir "gaussian-splatting/output"

# 脚本会自动:
# 1. 发现所有训练好的模型
# 2. 渲染200张test图像（每个模型）
# 3. 计算PSNR/SSIM/LPIPS指标
# 4. 生成对比报告
```

#### 步骤5.2: 查看结果

```powershell
# 查看渲染图像
dir datasets\lego2\3dgs_results\final_renders\
# 预期: 8个目录，每个200张PNG图像

# 查看评估指标
type datasets\lego2\3dgs_results\final_metrics\comparison_report.txt

# 查看JSON格式报告（便于后续分析）
type datasets\lego2\3dgs_results\final_metrics\comparison_report.json
```

#### 步骤5.3: 生成对比视频（可选）

```bash
# 回到WSL环境
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction

# 激活Umain2环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

# 生成所有视频
python create_videos.py lego2 --grayscale --fps 60

# 查看视频
ls -lh videos/lego2_*.mp4
```

**🎉 阶段5完成**: 你现在有了完整的评估报告和可视化结果！

---

## 常见问题

### ❓ Q1: 如何只处理特定的H5文件？

**A**: 修改 `process_additional_h5_files.py` 的扫描逻辑，或手动指定H5文件：

```bash
# 只处理EFR H5文件
python -c "
import sys; sys.path.append('.')
from modules.evreal_integration import *
from pathlib import Path

config = EVREALIntegrationConfig()
config.dataset_name = 'lego2'
config.dataset_dir = Path('datasets/lego2')
config.h5_source = 'EFR'  # 指定H5来源

integration = EVREALIntegration(config)
result = integration.run_full_pipeline()
"
```

### ❓ Q2: 如何选择不同的EVREAL重建方法？

**A**: 在生成3DGS配置时更改方法名称：

```powershell
# 使用E2VID方法
python generate_json_configs.py lego2 e2vid

# 使用ET-Net方法
python generate_json_configs.py lego2 et_net

# 使用SSL-E2VID方法
python generate_json_configs.py lego2 ssl_e2vid
```

### ❓ Q3: 训练失败或中断怎么办？

**A**: 3DGS训练支持断点续传：

```batch
REM 检查已完成的训练
dir gaussian-splatting\output\lego2_*\point_cloud\iteration_*

REM 如果某个配置失败，单独重新训练:
REM 编辑 configs/lego2/lego2_spade_e2vid_EFR.json
REM 然后运行:
python gaussian-splatting/train.py ^
    -s datasets/lego2 ^
    --config configs/lego2/lego2_spade_e2vid_EFR.json ^
    --eval
```

### ❓ Q4: 如何只重新渲染而不重新训练？

**A**: 直接运行渲染脚本：

```powershell
python render_and_evaluate.py --dataset lego2 --method spade_e2vid --weights-dir "gaussian-splatting/output"
```

### ❓ Q5: EFR/PFD编译失败怎么办？

**A**: 检查编译依赖：

```bash
# 安装必要的编译工具
sudo apt-get update
sudo apt-get install build-essential cmake

# EFR编译
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/EFR-main
rm -rf build
mkdir build && cd build
cmake .. && make

# PFD编译
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/PFD
rm -rf build_wsl
mkdir build_wsl && cd build_wsl
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

### ❓ Q6: UNet新权重推理失败？

**A**: 检查权重文件和配置：

```bash
# 确认权重文件存在
ls -lh /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/checkpoints/event_voxel_deflare_physics_noRandom_noTen_method/checkpoint_epoch_0031_iter_040000.pth

# 检查main.py的inference模式参数
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main
python main.py --help  # 查看可用参数
```

---

## 预期输出

### 📊 最终文件结构

```
datasets/lego2/
├── train/                          # 原始训练图像 (200张)
├── test/                           # 原始测试图像 (200张)
├── events_h5/                      # 事件H5文件
│   ├── lego2_sequence_new.h5       # 原始 (174万事件)
│   ├── lego2_sequence_new_EFR.h5   # EFR处理
│   ├── lego2_sequence_new_PFDA.h5  # PFD-A处理
│   ├── lego2_sequence_new_PFDB.h5  # PFD-B处理
│   ├── lego2_sequence_new_Unet.h5  # Unet处理 (127万事件)
│   ├── lego2_sequence_new_UnetNew.h5    # UNet新权重处理
│   └── lego2_sequence_new_Unetsimple.h5 # Unetsimple处理 (137万事件)
├── reconstruction_original/        # 原始H5重建 (8×200张)
│   ├── evreal_e2vid/
│   ├── evreal_spade_e2vid/
│   ├── evreal_et_net/
│   └── ... (其他方法)
├── reconstruction_EFR/             # EFR H5重建 (8×200张)
├── reconstruction_PFDA/            # PFD-A H5重建 (8×200张)
├── reconstruction_PFDB/            # PFD-B H5重建 (8×200张)
├── reconstruction_Unet/            # Unet H5重建 (8×200张)
├── reconstruction_UnetNew/         # UNet新权重H5重建 (8×200张)
├── reconstruction_Unetsimple/      # Unetsimple H5重建 (8×200张)
└── 3dgs_results/
    ├── weights/                    # 训练权重备份 (8×50MB)
    │   ├── original/
    │   ├── spade_e2vid_original/
    │   ├── spade_e2vid_EFR/
    │   ├── spade_e2vid_PFDA/
    │   ├── spade_e2vid_PFDB/
    │   ├── spade_e2vid_Unet/
    │   ├── spade_e2vid_UnetNew/
    │   └── spade_e2vid_Unetsimple/
    ├── final_renders/              # 渲染结果 (8×200张)
    │   ├── original/
    │   ├── spade_e2vid_original/
    │   └── ... (其他配置)
    └── final_metrics/              # 评估指标
        ├── comparison_report.txt   # 文本对比报告
        └── comparison_report.json  # JSON格式报告

gaussian-splatting/output/          # 3DGS训练输出
├── lego2_original/
├── lego2_spade_e2vid_original/
├── lego2_spade_e2vid_EFR/
├── lego2_spade_e2vid_PFDA/
├── lego2_spade_e2vid_PFDB/
├── lego2_spade_e2vid_Unet/
├── lego2_spade_e2vid_UnetNew/
└── lego2_spade_e2vid_Unetsimple/

videos/                             # 可视化视频
├── lego2_train_grayscale.mp4
├── lego2_test_grayscale.mp4
├── lego2_3dgs_original.mp4
├── lego2_3dgs_spade_e2vid_original.mp4
├── lego2_3dgs_spade_e2vid_EFR.mp4
└── ... (其他配置)
```

### 📈 性能基准参考 (lego2数据集)

| 配置 | 事件数量 | EVREAL方法 | PSNR | SSIM | LPIPS | 训练时间 |
|------|----------|------------|------|------|-------|----------|
| **original** | - | - | 28.5 | 0.85 | 0.15 | 45分钟 |
| **spade_e2vid_original** | 174万 | SPADE-E2VID | 27.2 | 0.82 | 0.18 | 45分钟 |
| **spade_e2vid_EFR** | ? | SPADE-E2VID | ? | ? | ? | 45分钟 |
| **spade_e2vid_PFDA** | ? | SPADE-E2VID | ? | ? | ? | 45分钟 |
| **spade_e2vid_PFDB** | ? | SPADE-E2VID | ? | ? | ? | 45分钟 |
| **spade_e2vid_Unet** | 127万 | SPADE-E2VID | 27.5 | 0.83 | 0.17 | 45分钟 |
| **spade_e2vid_UnetNew** | ? | SPADE-E2VID | ? | ? | ? | 45分钟 |
| **spade_e2vid_Unetsimple** | 137万 | SPADE-E2VID | 27.3 | 0.82 | 0.18 | 45分钟 |

---

## 🎯 实验完成检查清单

- [ ] **阶段1**: 基础H5文件生成成功 (`xxx_sequence_new.h5` 存在)
- [ ] **阶段2**: 7个H5变体文件全部生成成功
  - [ ] EFR处理完成
  - [ ] PFD-A处理完成
  - [ ] PFD-B处理完成
  - [ ] UNet新权重处理完成
  - [ ] Unet处理完成（已有）
  - [ ] Unetsimple处理完成（已有）
- [ ] **阶段3**: EVREAL重建完成
  - [ ] 7个 `reconstruction_xxx/` 目录存在
  - [ ] 每个目录包含8种重建方法
  - [ ] 总计约11,200张重建图像
- [ ] **阶段4**: 3DGS训练完成
  - [ ] 8个配置训练成功
  - [ ] 权重文件已备份到 `datasets/xxx/3dgs_results/weights/`
- [ ] **阶段5**: 渲染评估完成
  - [ ] 1,600张渲染图像生成
  - [ ] `comparison_report.txt` 和 `comparison_report.json` 存在
  - [ ] （可选）对比视频生成

---

## 📞 获取帮助

如果遇到问题，请检查：

1. **环境问题**: 确认conda环境已正确激活
2. **路径问题**: 确认所有路径使用绝对路径
3. **文件存在性**: 确认输入文件存在且格式正确
4. **日志输出**: 查看详细的错误日志
5. **磁盘空间**: 确认有足够的磁盘空间（至少20GB）

---

**教程结束** | 祝实验顺利！ 🚀
