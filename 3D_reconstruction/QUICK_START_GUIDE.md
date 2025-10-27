# ⚡ 三维重建实验 - 快速启动指南

**适用场景**: 你已有基础H5文件（如lego2_sequence_new.h5），想要快速执行完整实验
**预计时间**:
- WSL部分（阶段2-3）：约2-4小时
- Windows部分（阶段4-5）：约6-8小时

---

## 🎯 一键式执行（推荐）

### WSL环境 - 事件处理+EVREAL重建

```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction

# 编辑quick_start_new_methods.sh，修改DATASET变量为你的数据集名称
# DATASET="lego2"  # 改为你的数据集

# 执行完整流程（阶段2-3）
./quick_start_new_methods.sh
```

**这个脚本会自动完成**:
1. ✅ 环境检查和激活
2. ✅ EFR炫光去除
3. ✅ PFD-A炫光去除
4. ✅ PFD-B炫光去除
5. ✅ UNet新权重处理
6. ✅ EVREAL批量重建（所有H5文件 × 8种方法）

### Windows环境 - 3DGS训练+渲染

```powershell
cd E:\BaiduSyncdisk\2025\event_flick_flare\experiments\3D_reconstruction

# 激活3DGS环境
conda activate gaussian_splatting  # 或你的实际环境名称

# 1. 生成配置
python generate_json_configs.py lego2 spade_e2vid

# 2. 批量训练（约6小时）
train_3dgs_batch.bat lego2 spade_e2vid

# 3. 渲染评估（约30分钟）
python render_and_evaluate.py --dataset lego2 --method spade_e2vid --weights-dir "gaussian-splatting/output"
```

---

## 📋 分步执行（高级用户）

### 阶段2: 多方法事件处理

#### 前提条件
```bash
# 确认输入H5文件存在
ls -lh datasets/lego2/events_h5/lego2_sequence_new.h5
```

#### 步骤2.1: EFR处理
```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/EFR-main

python batch_efr_processor.py \
    --input_h5 "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5/lego2_sequence_new.h5" \
    --output_dir "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5" \
    --base_frequency 50 \
    --rho1 0.6 \
    --delta_t 10000

# 返回主目录
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 步骤2.2: PFD-A处理
```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/PFD

python batch_pfd_processor.py \
    --input_h5 "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5/lego2_sequence_new.h5" \
    --output_dir "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5" \
    --score_select 1 \
    --delta_t0 20000 \
    --delta_t 20000

cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 步骤2.3: PFD-B处理
```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/PFD

python batch_pfd_processor.py \
    --input_h5 "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5/lego2_sequence_new.h5" \
    --output_dir "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5" \
    --score_select 0 \
    --delta_t0 20000 \
    --delta_t 20000

cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 步骤2.4: UNet新权重处理
```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main

# 使用新的inference配置文件（已指向新权重）
python main.py inference \
    --config configs/inference_config_new_checkpoint.yaml \
    --input "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5/lego2_sequence_new.h5" \
    --output "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego2/events_h5/lego2_sequence_new_UnetNew.h5"

cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction
```

#### 验证H5文件
```bash
ls -lh datasets/lego2/events_h5/*.h5

# 预期输出：
# lego2_sequence_new.h5         (原始)
# lego2_sequence_new_EFR.h5     (EFR)
# lego2_sequence_new_PFDA.h5    (PFD-A)
# lego2_sequence_new_PFDB.h5    (PFD-B)
# lego2_sequence_new_Unet.h5    (Unet，已有)
# lego2_sequence_new_UnetNew.h5 (UNet新权重)
# lego2_sequence_new_Unetsimple.h5 (Unetsimple，已有)
```

---

### 阶段3: EVREAL批量重建

```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction

# 激活Umain2环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

# 批量重建所有H5文件
python process_additional_h5_files.py lego2

# 查看重建结果
ls -d datasets/lego2/reconstruction_*/

# 统计重建图像总数
find datasets/lego2/reconstruction_* -name "*.png" | wc -l
# 预期：约11,200张图像（7个H5 × 8种方法 × 200张）
```

---

### 阶段4-5: 3DGS训练和评估（Windows）

见上方"Windows环境 - 3DGS训练+渲染"部分

---

## 🔍 关键参数说明

### EFR参数
- `--base_frequency 50`: 炫光频率（Hz），fluorescent灯通常为50Hz或60Hz
- `--rho1 0.6`: 主反馈系数，控制滤波强度
- `--delta_t 10000`: 事件聚合时间窗口（μs），10ms

### PFD参数
- `--score_select 1`: PFD-A模式（使用方差评分）
- `--score_select 0`: PFD-B模式（使用邻域评分）
- `--delta_t0 20000`: 第一阶段时间窗口（μs），20ms
- `--delta_t 20000`: 第二阶段时间窗口（μs），20ms

### UNet新权重
- 权重路径: `checkpoints/event_voxel_deflare_physics_noRandom_noTen_method/checkpoint_epoch_0031_iter_040000.pth`
- 配置文件: `configs/inference_config_new_checkpoint.yaml`（已创建）

---

## ⚠️ 常见问题

### Q1: EFR/PFD可执行文件不存在？
**A**: 需要编译：
```bash
# EFR编译
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/EFR-main
mkdir -p build && cd build && cmake .. && make

# PFD编译
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/ext/PFD
mkdir -p build_wsl && cd build_wsl && cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

### Q2: UNet推理失败？
**A**: 检查：
1. 权重文件是否存在: `ls -lh /mnt/e/.../checkpoint_epoch_0031_iter_040000.pth`
2. 配置文件是否正确: `cat configs/inference_config_new_checkpoint.yaml`
3. CUDA是否可用: `python -c "import torch; print(torch.cuda.is_available())"`

### Q3: EVREAL重建太慢？
**A**: EVREAL是CPU密集型任务，预计：
- 单个H5文件 × 8种方法 ≈ 20-40分钟
- 7个H5文件 ≈ 2-4小时

可以修改`process_additional_h5_files.py`，只处理部分H5文件。

### Q4: 3DGS训练中断？
**A**: 单独重新训练失败的配置：
```batch
python gaussian-splatting/train.py ^
    -s datasets/lego2 ^
    --config configs/lego2/lego2_spade_e2vid_EFR.json ^
    --eval
```

---

## 📊 预期结果

### 最终文件结构
```
datasets/lego2/
├── events_h5/                  # 7个H5文件（约1-2GB）
├── reconstruction_original/    # 8种方法 × 200张
├── reconstruction_EFR/         # 8种方法 × 200张
├── reconstruction_PFDA/        # 8种方法 × 200张
├── reconstruction_PFDB/        # 8种方法 × 200张
├── reconstruction_Unet/        # 8种方法 × 200张
├── reconstruction_UnetNew/     # 8种方法 × 200张（新权重）
├── reconstruction_Unetsimple/  # 8种方法 × 200张
└── 3dgs_results/
    ├── weights/                # 8个配置权重
    ├── final_renders/          # 8×200张渲染图像
    └── final_metrics/          # 评估报告
```

### 数据规模
- **总重建图像**: 约11,200张（~5-10GB）
- **总渲染图像**: 约1,600张（~1-2GB）
- **总训练权重**: 约400MB（8×50MB）

---

## 📞 获取帮助

- **详细教程**: 查看 `COMPLETE_RECONSTRUCTION_TUTORIAL.md`
- **项目文档**: 查看 `CLAUDE.md`
- **快速脚本**: 使用 `quick_start_new_methods.sh`

---

**快速启动指南结束** | 开始你的实验吧！ 🚀
