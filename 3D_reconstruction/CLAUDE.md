# 事件相机3D重建项目指导书

**最后更新**: 2025-10-27
**状态**: Production Ready

---

## 📚 项目概述

事件相机3D重建完整pipeline，支持DVS仿真、多种炫光去除方法（EFR、PFD-A、PFD-B、UNet）、EVREAL重建、3DGS训练评估。

### 核心功能
- ✅ DVS事件仿真 → EVREAL/H5格式转换
- ✅ 多种炫光去除: EFR、PFD-A、PFD-B、UNet、UnetSimple
- ✅ EVREAL重建: 8种方法 × 多个H5文件
- ✅ 3DGS训练评估: 完整的渲染和指标计算

### 成功案例（lego2数据集）
- **事件数据**: 174万事件 → 7种处理方法 → 7个H5文件
- **重建图像**: 7×8×200 = 11,200张重建图像
- **3DGS训练**: 8个配置，完整PSNR/SSIM/LPIPS评估

---

## 🚀 快速使用指南

### ⚡ WSL环境 - 事件处理+EVREAL重建

```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction

# 一键执行完整流程
./quick_start_new_methods.sh

# 或查看详细教程
cat QUICK_START_GUIDE.md
```

### ⚡ Windows环境 - 3DGS训练+评估

```powershell
# 1. 配置生成
python generate_json_configs.py lego2 spade_e2vid

# 2. 批量训练
train_3dgs_batch.bat lego2 spade_e2vid

# 3. 渲染评估
python render_and_evaluate.py --dataset lego2 --method spade_e2vid --weights-dir "gaussian-splatting/output"
```

---

## 📁 完整Pipeline流程

```
阶段1: 基础数据 (WSL - Umain2)
├─ 数据集合并: xxx_flare + xxx_normal → xxx
├─ DVS仿真: 图像 → 事件流
└─ 格式转换: DVS → EVREAL + H5

阶段2: 多方法事件处理 (外部 - Unet_main项目)
⚠️ 此阶段在Unet_main项目中独立完成，生成多种处理后的H5文件
├─ EFR、PFD-A、PFD-B、UNet新权重等处理
└─ 输出: 多个H5文件放入datasets/xxx/events_h5/目录

阶段3: EVREAL重建 (WSL - Umain2)
└─ 批量重建: process_additional_h5_files.py

阶段4: 3DGS训练 (Windows)
└─ 批量训练: train_3dgs_batch.bat

阶段5: 3DGS评估 (Windows)
└─ 渲染评估: render_and_evaluate.py
```

---

## 🔧 核心脚本

### WSL环境脚本
- `merge_datasets.py`: 数据集合并
- `run_full_pipeline.py`: DVS仿真+格式转换
- `process_additional_h5_files.py`: EVREAL批量重建
- `calculate_reconstruction_metrics.py`: 重建质量评估

### 外部依赖
- **Unet_main项目**: 负责EFR、PFD-A、PFD-B、UNet等炫光去除方法
  - 在该项目中处理原始H5文件
  - 生成多种处理后的H5文件
  - 将结果H5文件复制到本项目的`datasets/xxx/events_h5/`目录

### Windows环境脚本
- `generate_json_configs.py`: 生成3DGS训练配置
- `train_3dgs_batch.bat`: 批量3DGS训练
- `render_and_evaluate.py`: 渲染+评估
- `create_videos.py`: 生成对比视频

---

## 🗂️ 标准数据结构

```
datasets/lego2/
├── train/                          # 原始训练图像 (200张flare)
├── test/                           # 原始测试图像 (200张normal)
├── events_h5/                      # H5事件文件
│   ├── lego2_sequence_new.h5       # 原始 (174万事件)
│   ├── lego2_sequence_new_EFR.h5
│   ├── lego2_sequence_new_PFDA.h5
│   ├── lego2_sequence_new_PFDB.h5
│   ├── lego2_sequence_new_Unet.h5
│   ├── lego2_sequence_new_UnetNew.h5
│   └── lego2_sequence_new_Unetsimple.h5
├── reconstruction_original/        # EVREAL重建 (8×200张)
├── reconstruction_EFR/
├── reconstruction_PFDA/
├── reconstruction_PFDB/
├── reconstruction_Unet/
├── reconstruction_UnetNew/
├── reconstruction_Unetsimple/
└── 3dgs_results/
    ├── weights/                    # 训练权重
    ├── final_renders/              # 渲染结果
    └── final_metrics/              # 评估指标
```

---

## ⚙️ 技术参数

### DVS仿真
- 时间间隔: 1ms → 等效1000fps
- DVS型号: DVS346
- 事件格式: `[timestamp_us, x, y, polarity]`

### 炫光去除方法
⚠️ **重要**: 所有炫光去除处理在Unet_main项目中完成
- 生成的H5文件需要复制到`datasets/xxx/events_h5/`目录
- 本项目只负责EVREAL重建和3DGS训练评估

### EVREAL重建
- 8种方法: E2VID, FireNet, SPADE-E2VID, SSL-E2VID, ET-Net等
- 完美200:200图像对应
- 输出格式: 480×640 PNG

### 3DGS训练
- 训练参数: --eval, 10000 iterations
- 渲染: 200张test图像
- 评估指标: PSNR, SSIM, LPIPS

---

## 📖 详细文档

- **完整教程**: `COMPLETE_RECONSTRUCTION_TUTORIAL.md` - 5阶段详细说明
- **快速指南**: `QUICK_START_GUIDE.md` - 精简版操作步骤
- **自动化脚本**: `quick_start_new_methods.sh` - 一键执行

---

## ⚠️ 重要注意事项

### 环境要求
- **WSL环境**: Umain2 (DVS仿真、EVREAL重建)
- **Windows环境**: gaussian_splatting (3DGS训练)
- **环境保护铁律**: 只能添加新包，不能升级/降级现有包！

### 外部处理说明
- **炫光去除**: 所有EFR、PFD、UNet处理在Unet_main项目中完成
- **H5文件准备**: 将处理后的H5文件放入`datasets/xxx/events_h5/`目录
- **命名约定**: 建议使用后缀区分不同方法（如`_EFR.h5`、`_PFDA.h5`等）

---

## 🎯 预期输出

### 数据规模
- **H5文件**: 7个 (~1-2GB)
- **重建图像**: 11,200张 (~5-10GB)
- **3DGS权重**: 8个配置 (~400MB)
- **渲染图像**: 1,600张 (~1-2GB)

### 时间预估
- **WSL部分** (阶段2-3): 2-4小时
- **Windows部分** (阶段4-5): 6-8小时

---

## 📞 故障排除

### 常见问题
1. **EFR/PFD编译失败**: 安装`build-essential cmake`
2. **UNet推理失败**: 检查权重文件和CUDA
3. **EVREAL重建慢**: 正常现象，CPU密集型任务
4. **3DGS训练中断**: 使用单独命令重新训练失败配置

### 获取帮助
- 查看详细教程: `COMPLETE_RECONSTRUCTION_TUTORIAL.md`
- 查看快速指南: `QUICK_START_GUIDE.md`
- 检查TODO: `TODO.md`

---

**项目文档精简版** | 更多细节请查看详细教程
