# 事件相机图像重建模块

## 功能说明

**输入**: H5事件数据文件（DVS格式）
**输出**: 重建图像（PNG格式，8种EVREAL方法）

本模块是从主项目中复用的核心代码，专注于H5→图像重建，不涉及3D重建。

---

## 快速使用

### 环境要求
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2
```

### 基础用法

**处理单个数据集的所有H5文件**:
```bash
cd image_reconstruction
python h5_to_images.py <dataset_name>

# 示例：处理lego2数据集的所有H5文件
python h5_to_images.py lego2
```

**脚本会自动**:
1. 扫描 `datasets/<dataset_name>/events_h5/` 中的所有H5文件
2. 为每个H5文件生成EVREAL格式数据
3. 运行8种重建方法（E2VID, FireNet, SPADE-E2VID, SSL-E2VID等）
4. 输出到 `datasets/<dataset_name>/reconstruction_<suffix>/`

---

## 输出结构

```
datasets/<dataset_name>/
├── events_h5/
│   ├── <name>_sequence_new.h5          # 原始H5
│   ├── <name>_sequence_new_Unet.h5     # Unet处理后
│   └── <name>_sequence_new_Unetsimple.h5
└── reconstruction_original/             # 原始H5重建结果
    ├── evreal_e2vid/                    # E2VID方法（200张PNG）
    ├── evreal_firenet/                  # FireNet方法（200张PNG）
    ├── evreal_spade_e2vid/              # SPADE-E2VID方法（200张PNG）
    ├── evreal_ssl_e2vid/                # SSL-E2VID方法（200张PNG）
    └── ...（其他方法）
└── reconstruction_Unet/                 # Unet H5重建结果
    └── ...（同上）
└── reconstruction_Unetsimple/           # Unetsimple H5重建结果
    └── ...（同上）
```

---

## 核心技术

**复用的代码模块**:
- `h5_to_images.py` - 主脚本（复用自 `process_additional_h5_files.py`）
- `modules/format_converter.py` - H5↔EVREAL格式转换
- `modules/evreal_integration.py` - EVREAL重建集成
- `pipeline_architecture.py` - 数据结构定义

**关键技术**:
1. **动态索引重新映射**: 自动适配不同事件数量的H5文件
2. **智能结构复用**: 基于成功案例的EVREAL数据结构模板
3. **完美200:200对应**: 通过智能补全实现输入输出图像完美对应

---

## 支持的重建方法（8种）

| 方法 | 特点 | 预期成功率 |
|------|------|-----------|
| E2VID | 经典方法 | ✅ 高 |
| E2VID+ | 增强版 | ⚠️ 中（路径兼容性） |
| FireNet | 快速方法 | ✅ 高 |
| FireNet+ | 增强版 | ⚠️ 中（路径兼容性） |
| SPADE-E2VID | 空间自适应 | ✅ 高 |
| SSL-E2VID | 自监督学习 | ✅ 高 |
| ET-Net | 事件-纹理网络 | ✅ 高 |
| HyperE2VID | 最新方法 | ⚠️ 中（内存要求高） |

---

## 成功案例（lego2数据集）

| H5文件 | 事件数量 | 成功方法 | 重建图像 | 最佳质量 |
|--------|----------|----------|----------|----------|
| lego2_original | 174万 | 8/8 | 1600张 | ET-Net (MSE=0.037) |
| lego2_Unet | 127万 | 8/8 | 1600张 | ET-Net (MSE=0.037) |
| lego2_Unetsimple | 137万 | 8/8 | 1600张 | SPADE-E2VID |

---

## 与主项目的关系

本模块是**主项目的子集**，专注于图像重建：
- ✅ **包含**: H5加载、格式转换、EVREAL重建
- ❌ **不包含**: DVS仿真、数据集合并、3DGS训练

如需完整功能，请使用主项目的 `process_additional_h5_files.py`。

---

## 独立修改指南

由于是复制的代码，你可以自由修改而不影响主项目：

### 修改重建方法列表
编辑 `modules/evreal_integration.py:52-61`

### 修改输出目录结构
编辑 `h5_to_images.py:209`（`config.reconstruction_dir`）

### 添加自定义后处理
在 `modules/evreal_integration.py:351-481` 的 `copy_reconstruction_results` 方法中添加

---

## 依赖项

- Python 3.x
- Umain2 conda环境
- EVREAL框架（/mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main）
- 系统库：h5py, numpy, opencv-python

---

**Author**: Claude Code Assistant
**Date**: 2025-10-25
**Based on**: `process_additional_h5_files.py` (成熟稳定版本)
