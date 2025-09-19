# 3D Reconstruction with Event Camera Simulation Project

## 项目总体架构
这是一个完整的事件相机3D重建实验项目，包含核心Pipeline：
1. **数据集处理**: 合并炫光和正常光照数据集 ✅已完成
2. **图像预处理**: train图像 → DVS仿真输入格式 ✅已完成  
3. **DVS事件仿真**: 图像序列 → 事件数据 ✅已完成
4. **格式转换**: DVS → EVREAL + H5格式 ✅已完成
5. **图像重建**: EVREAL事件重建 🔄部分完成(环境问题)
6. **时间戳对齐**: 原始图像-事件-重建图像对齐 📋待实施

## 🎯 当前实验状态（2025-09-19更新）

### ✅ 已完成模块
1. **Pipeline架构设计** (`pipeline_architecture.py`)
   - 完整的数据流和接口定义
   - 支持DVS、EVREAL、H5三种格式的转换

2. **图像预处理模块** (`modules/image_preprocessor.py`)
   - lego/train (200张flare版本) → DVS输入格式
   - 自动生成info.txt和时间戳
   - 时间间隔: 1ms (1000μs)

3. **DVS仿真器封装** (`modules/dvs_simulator.py`)
   - 成功仿真生成 4,771,501 个事件
   - 时间范围: 0-479μs，空间: 640x480
   - 极性分布: ON=2,457,813, OFF=2,313,688
   - 输出位置: `datasets/lego/events_dvs/lego_train_events.txt`

4. **格式转换器** (`modules/format_converter.py`)
   - DVS txt → EVREAL npy格式 (events_ts/xy/p.npy)
   - DVS txt → H5格式 (events/t/x/y/p)
   - 支持双向转换 (H5 ↔ DVS, H5 ↔ EVREAL)
   - 输出位置: `datasets/lego/events_evreal/sequence/`

### ✅ 已完成模块 
5. **EVREAL集成模块** (`modules/evreal_integration.py`)
   - **当前状态**: ✅ Pipeline调用成功，❌ 重建图像生成问题
   - **已解决问题**:
     * 跨平台路径兼容性 (WindowsPath在WSL中的问题) ✅
     * EVREAL依赖安装 (tabulate, yachalk, pyiqa等) ✅ 
     * 数据集配置格式匹配EVREAL标准 ✅
     * 目录结构调整 (events_evreal/sequence/) ✅
     * 真值图像配置 (lego/test复制到sequence/images/) ✅
     * PNG→numpy转换功能集成 ✅
   - **当前核心问题**: ❌ **EVREAL重建图像未生成**
     * EVREAL运行成功，但只生成空的指标文件
     * 缺少重建的PNG图像文件（frame_*.png）
     * 可能原因: E2VID模型文件、数据格式、路径问题

6. **一键式主控Pipeline** (`run_full_pipeline.py`)
   - **当前状态**: ✅ 100%完成，真正一键运行
   - **功能**: 从lego_flare+lego_normal → 完整重建流程
   - **验证结果**: 
     * 数据集合并 ✅
     * 图像预处理 ✅  
     * DVS事件仿真 ✅
     * 格式转换 ✅
     * EVREAL调用 ✅
   - **问题**: 最终重建图像数量为0

### 📋 待实施模块  
6. **时间戳对齐系统**
   - 确保原始图像、仿真事件、重建图像时间戳完全对应
   - 文件命名规范化
   
7. **主控Pipeline脚本**
   - 端到端自动化执行
   - 配置管理系统
   
8. **结果验证模块**
   - 文件完整性检查
   - 时间戳一致性验证
   - 图像质量评估

## 🚨 遇到的核心问题

### 1. 路径管理问题 (已解决)
- **问题**: DVS输出在temp目录，不符合数据存储规范
- **解决**: 修正为直接输出到 `datasets/lego/events_dvs/`
- **状态**: 已更新所有相关模块的路径引用

## 🎯 实验工作流程（更新版）

### 阶段1: 数据准备 ✅完成
```bash
# 数据集合并
python merge_datasets.py

# 图像预处理
python modules/image_preprocessor.py
```

### 阶段2: 事件仿真 ✅完成  
```bash
# DVS事件仿真
python modules/dvs_simulator.py

# 格式转换
python modules/format_converter.py
```

### 阶段3: 图像重建 ✅可以执行
```bash
# EVREAL图像重建
python modules/evreal_integration.py
```

### 阶段4: 结果分析 📋待实施
- 时间戳对齐验证
- 重建质量评估
- 与原始图像对比

## 📁 当前数据结构
```
datasets/lego/
├── train/                  # 原始训练图像 (200张)
├── test/                   # 原始测试图像 (200张) 
├── events_dvs/            # DVS原始事件数据
│   └── lego_train_events.txt
├── events_evreal/         # EVREAL标准格式
│   ├── events_ts.npy
│   ├── events_xy.npy  
│   ├── events_p.npy
│   └── metadata.json
├── events_h5/             # H5格式
│   └── lego_sequence.h5
└── reconstruction/        # 重建结果目录(预留)
    └── evreal_*/
```

## ⚠️ 重要提醒
- **严重问题必须停下反馈**: 环境依赖、内存不足、关键文件丢失等
- **路径一致性**: 所有模块已更新使用数据集目录而非temp
- **时间戳格式**: DVS使用微秒，EVREAL使用秒，H5保持微秒
- **文件命名**: 严格按照 `lego_train_events.txt` 等规范命名
- **🚨 环境保护铁律**: 绝对不可以破坏conda环境中的已有包！只能添加新包，不能升级/降级现有包！

## 数据集结构

### 原始数据集
- `lego_flare/`: 带炫光的lego数据集
- `lego_normal/`: 无炫光的lego数据集
- 两个数据集的位姿（transform_matrix）完全相同
- 每个数据集包含200张训练图片

### 合并后数据集
- `lego/`: 合并后的完整数据集
  - `train/`: 200张训练图片（来自lego_flare，包含炫光）
  - `test/`: 200张测试图片（来自lego_normal，无炫光）
  - `transforms_train.json`: 训练集位姿文件
  - `transforms_test.json`: 测试集位姿文件
  - `points3d.ply`: 3D点云文件

### 路径修正问题
原始transforms文件中的路径格式为 `"train\\r_00001"`，但实际文件名为 `0001.png`。
合并脚本已自动修正为正确格式：`"train/0001"` 和 `"test/0001"`。

## 脚本文件

### merge_datasets.py
数据集合并脚本，功能包括：
1. 自动检测所有 `*_flare` 和 `*_normal` 数据集对
2. 将flare版本作为训练集，normal版本作为测试集
3. 修正transforms文件中的路径格式错误
4. 验证合并结果的完整性

使用方法：
```bash
python3 merge_datasets.py
```

## 实验设计思路
- 训练阶段：使用带炫光的图片训练模型，让模型学会处理炫光干扰
- 测试阶段：使用无炫光的图片验证模型在理想条件下的重建质量
- 位姿完全相同确保了训练和测试的一致性

## 📦 模块详细说明

### 3D Gaussian Splatting (`gaussian-splatting/`)
**功能**: 实时3D场景重建和新视角合成
**特色**: 集成了PDTS智能视图选择算法，提升训练效率

#### 核心特性:
- **实时渲染**: ≥30fps的1080p新视角合成
- **智能视图选择**: PDTS神经网络预测损失，替代直接计算
- **混合采样策略**: Bootstrap阶段随机采样 + Hybrid阶段网络指导采样
- **高质量重建**: 保持SOTA视觉质量的同时提升训练速度

#### 关键文件:
- `train.py`: 主训练脚本（集成PDTS选择逻辑）
- `render.py`: 渲染脚本，生成新视角图像
- `pdts_integration.py`: PDTS风险学习器实现
- `scene/`: 场景数据加载和管理
- `gaussian_renderer/`: 高斯光栅化渲染器

#### 运行环境:
- **推荐**: Windows + CUDA GPU
- **原因**: 用户熟悉Windows环境，GPU加速效果更好

### DVS事件相机仿真器 (`DVS-Voltmeter-main/`)  
**功能**: 基于随机过程的DVS事件相机仿真器
**论文**: "DVS-Voltmeter: Stochastic Process-based Event Simulator" (ECCV 2022)

#### 核心算法:
- **随机过程建模**: 模拟真实DVS传感器的噪声特性
- **像素级仿真**: 每个像素独立的事件生成过程
- **参数化模型**: 支持DVS346、DVS240等不同相机型号

#### 输入要求:
- 高帧率图像序列（通过视频插值获得）
- 图像格式: PNG格式，按数字编号命名
- 配置文件: `info.txt` 描述序列信息

#### 输出数据:
- 事件数据流: (x, y, timestamp, polarity)
- 可视化结果: 事件体素网格表示

#### 关键文件:
- `main.py`: 主仿真脚本
- `src/simulator.py`: 事件仿真器核心算法  
- `src/config.py`: 相机参数配置
- `src/visualize.py`: 事件数据可视化

## 🔄 数据流向设计

```
原始数据集 → 数据预处理 → 3DGS训练 → 图像序列生成 → 事件仿真 → 事件数据集
    ↓              ↓           ↓            ↓           ↓
lego_flare/   merge_datasets  gaussian-    render.py   DVS-Voltmeter  
lego_normal/       ↓         splatting/      ↓             ↓
    ↓         datasets/lego/     ↓       高帧率图像     events.txt
合并数据集      训练测试分离    3D模型      序列        + 可视化
```

## 📋 实验TODO清单

### 近期任务 (用户执行):
1. **3DGS训练测试**: 在Windows环境运行gaussian-splatting
2. **图像序列生成**: 使用训练好的模型渲染高帧率图像序列  
3. **序列质量检查**: 确保图像连续性和时间一致性

### 中期任务 (事件仿真):
1. **DVS仿真准备**: 准备高帧率图像序列输入
2. **参数调优**: 调整DVS相机参数匹配实验需求
3. **仿真执行**: 生成事件数据和可视化结果
4. **质量评估**: 事件数据的噪声水平和真实性分析

### 长期任务 (系统集成):
1. **端到端Pipeline**: 自动化完整工作流程
2. **多场景扩展**: 支持更多数据集和场景类型
3. **性能优化**: 各模块的速度和内存优化
4. **结果分析**: 炫光对事件相机仿真影响的定量分析

## 环境要求

### Linux环境 (当前WSL2):
- Python 3.x
- Umain环境（已有相关Python库）
- 用于数据预处理和分析

### Windows环境 (用户负责):
- CUDA支持的GPU
- Python + PyTorch
- 用于3DGS训练和渲染

### DVS仿真环境:
```bash
easydict == 1.9
pytorch >= 1.8  
numpy >= 1.20.1
opencv-python == 4.5.1.48
tqdm == 4.49.0
```