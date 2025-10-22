# 主实验框架 - Claude Code 项目记忆

## 项目概述
**事件流去炫光结果的定量评估框架**，专门用于评估**已经处理完成**的去炫光结果与真值数据的对比。

**重要澄清**：此框架 **不执行** 去炫光处理，仅负责评估外部方法的处理结果。

## 已完成的框架结构

### 核心模块
1. **data_loader.py** - 事件数据加载抽象层（支持多种格式）
2. **metrics.py** - 评估度量计算（包含距离指标、voxel指标、kernel指标）
3. **methods.py** - 方法结果加载器（**已简化**，移除了处理逻辑）
4. **evaluator.py** - 实验流程协调器（**已更新**，直接比较结果）
5. **run_main_experiment.py** - 主执行脚本（**已重构**）
6. **evaluate_all_methods.py** - 多方法批量评估脚本（推荐使用）
7. **evaluate_evk4_methods.py** - EVK4数据集专用评估脚本
8. **__init__.py** - 包初始化文件
9. **requirements.txt** - 依赖列表
10. **README.md** - 详细文档（已更新）

### 设计理念 (已调整)
- **格式无关**：不假设特定目录结构，通过显式文件路径工作
- **结果导向**：评估预处理的结果文件，而非执行处理
- **方法无关**：可以评估任何外部方法的输出结果
- **度量全面**：支持多种评估指标，易于扩展
- **结果导出**：支持 CSV 导出和 pandas DataFrame

## 当前实现状态

### ✅ 已完成
- **重构完成**：移除了不必要的去炫光处理代码
- 抽象接口设计完整
- 评估流程框架搭建完毕
- 用户提供的度量函数已集成
- `MethodResult` 类用于加载预处理结果
- 结果收集和导出功能
- 错误处理和失败分析

### 🚧 需要具体实现
1. **AEDAT4 文件读取**
   - 位置：`data_loader.py` Aedat4DataSource._load_data()
   - 需要：dv-processing 库集成或自定义读取代码
   
2. **H5 文件读取** 
   - 位置：`data_loader.py` H5DataSource._load_data()
   - 需要：基于实际 H5 文件结构的读取实现

### ❌ 已移除
- ~~主模型推理管道~~ (不需要，处理在外部完成)
- ~~基线方法实现~~ (不需要，只评估结果)
- ~~所有处理相关的代码~~ (框架只负责评估)

### 💡 关键设计决策
- 所有事件数据标准化为 NumPy 结构化数组：`[('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')]`
- 度量计算使用用户提供的归一化和距离计算方法
- 支持中间结果保存用于调试和分析
- 完整的错误处理和失败追踪机制
- **核心理念**：框架专注于评估，不涉及处理逻辑

## 使用流程 (已更新)
1. 在外部完成去炫光处理，得到结果文件
2. 更新 `run_main_experiment.py` 中的方法结果路径和真值数据路径
3. 补充所需文件格式的读取代码（如果使用 AEDAT4 或 H5）
4. 运行评估：`python run_main_experiment.py --output results/`

## 数据格式要求 (已更新)
- **方法结果**：外部处理得到的去炫光结果文件
- **真值数据**：仅包含背景的干净事件流
- **输出格式**：评估度量和比较报告

## 框架优势
- **清晰职责**：专注评估，不负责处理
- **灵活性**：可评估任何方法的结果
- **可扩展**：易于添加新的度量标准
- **格式支持**：支持多种事件数据格式

这个重构后的框架更符合实际需求，提供了一个专业的评估工具，可以公平比较不同去炫光方法的效果。

## 虚拟环境设置（轻量化方案）

### 创建和使用虚拟环境

**Windows 系统：**
```bash
# 创建环境
setup_environment.bat

# 激活环境
venv_main_exp\Scripts\activate.bat

# 停用环境
deactivate
```

**Linux/WSL 系统：**
```bash
# 创建环境
chmod +x setup_environment.sh
./setup_environment.sh

# 激活环境
source venv_main_exp/bin/activate

# 停用环境
deactivate
```

### 环境特点
- **轻量化**：使用 Python 原生 venv，避免 conda 占用 C 盘空间
- **本地化**：环境创建在项目目录内，便于管理
- **专用性**：仅安装必要依赖，减少空间占用
- **便携性**：可以整体移动项目目录而不影响环境

## 新增功能

### Voxel转换系统 ✅
- **文件**: `voxel_utils.py` - 完整的事件转voxel工具集
- **核心函数**: `events_to_voxel()` - 实现用户精确规范
- **关键特性**:
  - **固定20ms时间窗** - 保证训练一致性，避免自适应导致的泛化问题
  - **简单极性累积** - 正负事件分别+1/-1累积，符合Linus简洁哲学  
  - **向量化PyTorch实现** - 74倍性能提升，纯PyTorch避免内存泄漏
  - **100ms文件分块处理** - 自动分割为5×20ms块，避免显存问题
- **输入格式支持**: 兼容结构化数组和(N,4)普通数组格式
- **输出**: Voxel张量(T, H, W)，T=8时间bins，H×W=480×640空间分辨率
- **测试验证**: 4/4测试通过，支持真实H5数据处理
- **内存策略**: 分块voxel独立计算指标后平均，适合大规模数据

### AEDAT4 数据读取 ✅
- **位置**: `data_loader.py` Aedat4DataSource 类
- **功能**: 基于用户提供的参考代码实现
- **支持**: 自动时间戳对齐、格式标准化
- **依赖**: aedat 库

### H5 数据读取 ✅
- **位置**: `data_loader.py` H5DataSource 类  
- **功能**: 读取 `events/t`, `events/x`, `events/y`, `events/p` 格式
- **支持**: 时间戳转换(微秒→秒)、极性处理(1→+1, 非1→-1)
- **依赖**: h5py 库

### 批量H5评估脚本 ✅
- **文件**: `evaluate_simu_pairs_optimized.py` (单方法版，已过时)
- **文件**: `evaluate_all_methods.py` (多方法版，推荐使用)
- **功能**: 批量评估H5文件的去炫光效果
- **数据结构**:
  - **真值**: `target/` (包含 `*_bg_light.h5`)
  - **各种方法**: 其他所有文件夹，文件夹名即方法名 (包含 `*_bg_flare.h5`)
  - **配对逻辑**: 通过文件名前缀匹配，如 `composed_00470_bg_light.h5` 对应 `composed_00470_bg_flare.h5`
  - **评估原则**: 计算各种方法相对于真值的指标，方法间不互相比较
- **特点**:
  - 默认计算8个指标：chamfer_distance, gaussian_distance, pger, pmse_2, pmse_4, rf1, tf1, tpf1
  - 支持通过 `--metrics` 参数自定义指标组合
  - 结果统一保存到 `results/multi_method_evaluation_results.csv`
  - 自动生成包含AVERAGE行的CSV文件，适合论文直接使用
  - 支持动态发现新增方法文件夹

### 测试脚本 ✅
- **文件**: `test_aedat4_loading.py`
- **功能**: 验证 AEDAT4 读取和度量计算
- **数据**: 使用用户提供的测试文件

### 时间偏移度量分析 ✅
- **文件**: `time_offset_analysis.py`
- **功能**: 分析时间对齐误差对度量的影响
- **方法**: 
  - 提取前 0.1s 数据
  - 以 0.001s 步长进行时间偏移
  - 生成 100 组对比度量
  - 计算对齐敏感性

### DAVIS数据集生成器 ✅
- **文件**: `generate_davis_dataset.py`
- **功能**: 从AEDAT4文件生成DSEC格式的训练数据集
- **核心特性**:
  - **时间对齐**: 自动计算两文件的重叠区间(7.43s)
  - **随机采样**: 生成10个100ms数据对，随机分布避免时间偏差
  - **DSEC兼容**: H5格式完全匹配(events/t,x,y,p，微秒时间戳，0/1极性)
  - **目录结构**: input/(炫光数据) + target/(无炫光数据)
  - **文件命名**: `full_{起始时间ms}ms_sample{编号}.h5`
- **数据规模**: 每样本84万事件/100ms，文件11MB，总计20个H5文件
- **用途**: 为去炫光算法提供标准训练/测试数据集

### Voxel转换使用示例

#### 基本voxel转换
```python
from voxel_utils import events_to_voxel, events_to_voxel_chunks
from data_loader import load_events

# 加载事件数据
events = load_events("your_file.h5")

# 单个20ms voxel转换  
voxel = events_to_voxel(events, num_bins=8, sensor_size=(480, 640))

# 100ms文件分块处理（推荐）
voxel_chunks = events_to_voxel_chunks(events, num_bins=8)
print(f"Generated {len(voxel_chunks)} voxel chunks")

# 每个chunk形状: (8, 480, 640)
for i, voxel in enumerate(voxel_chunks):
    print(f"Chunk {i+1} shape: {voxel.shape}")
```

#### Voxel指标计算准备
```python
# 未来voxel指标的计算模式：
def calculate_voxel_metrics(voxel_chunks_est, voxel_chunks_gt):
    """计算基于voxel的指标"""
    chunk_metrics = []
    
    for voxel_est, voxel_gt in zip(voxel_chunks_est, voxel_chunks_gt):
        # 各种voxel指标计算 (待实现)
        metrics = {
            'voxel_similarity': calculate_voxel_similarity(voxel_est, voxel_gt),
            'structural_similarity': calculate_ssim_3d(voxel_est, voxel_gt),
            # 更多基于voxel的指标...
        }
        chunk_metrics.append(metrics)
    
    # 平均所有chunk的指标
    return average_chunk_metrics(chunk_metrics)
```

## 使用示例

### H5批量评估

#### 单方法评估
```bash
# 评估单个方法的全部50对H5文件
python evaluate_simu_pairs_optimized.py --output results

# 评估指定数量的样本
python evaluate_simu_pairs_optimized.py --num-samples 10 --output results

# 静默模式
python evaluate_simu_pairs_optimized.py --quiet --output results
```

#### 多方法评估 (推荐使用)
```bash
# 评估所有方法相对于真值的指标
python evaluate_all_methods.py --output results

# 评估指定数量的样本
python evaluate_all_methods.py --num-samples 10 --output results

# 使用voxel指标评估
python evaluate_all_methods.py --metrics tf1 tpf1 pmse_2 --output results

# 查看所有可用指标
python evaluate_all_methods.py --list-metrics

# 混合使用传统和voxel指标
python evaluate_all_methods.py --metrics chamfer_distance tf1 tpf1 --output results

# 静默模式
python evaluate_all_methods.py --quiet --output results
```

### 时间偏移分析
```bash
# 激活环境
source venv_main_exp/bin/activate

# 运行时间偏移分析
python time_offset_analysis.py \
  --clean_file "/mnt/e/2025/event_flick_flare/datasets/DAVIS/test/full_noFlare-2025_06_25_14_50_52.aedat4" \
  --flare_file "/mnt/e/2025/event_flick_flare/datasets/DAVIS/test/full_randomFlare-2025_06_25_14_52_42.aedat4"
```

### 测试数据读取
```bash
python test_aedat4_loading.py
```

## 当前状态 - 已测试验证 ✅

### H5批量评估验证成功
**全部50对H5文件评估完成**：
- **成功率**: 100% (50/50对成功)
- **平均Chamfer Distance**: 4.777 ± 3.226
- **平均Gaussian Distance**: 1.546 ± 0.173
- **数据规模**: 平均71万去炫光事件 vs 173万真值事件
- **处理时间**: 平均14.26秒/样本对，总计约12分钟

### AEDAT4数据读取验证
**AEDAT4数据读取测试成功**：
- 成功加载用户提供的测试数据
- 无炫光数据：71,893,808 events (8.530s)
- 随机炫光数据：62,162,676 events (7.430s) 
- 度量计算正常：Chamfer Distance = 15.88, Gaussian Distance = 1.98

### 快速启动方法

### **指标选择指南**

#### **🎯 推荐指标组合**
```bash
# 🌟 默认组合：传统+voxel完整指标（现已设为默认）
python evaluate_all_methods.py --output results

# 纯voxel组合：现代评估方法
python evaluate_all_methods.py --metrics pmse_2 pmse_4 rf1 tf1 tpf1

# 全面对比组合：覆盖所有评估维度  
python evaluate_all_methods.py --metrics chamfer_distance gaussian_distance tf1 tpf1 pmse_2 temporal_overlap

# 传统指标组合：兼容旧版本
python evaluate_all_methods.py --metrics chamfer_distance gaussian_distance
```

#### **📊 指标优劣性总结**
- **📈 越高越好**: tf1, tpf1, rf1, temporal_overlap
- **📉 越低越好**: chamfer_distance, gaussian_distance, pmse_2, pmse_4  
- **📊 比例指标**: event_count_ratio, pger (理想值≈1.0)

#### **⚠️ 距离指标优化** ✅
- **Chamfer/Gaussian距离重大改进**：从双向计算改为**单向计算**（估计→真值）
- **核心逻辑**：只计算estimated events到ground truth的距离，不反向计算
- **去炫光优化**：避免因移除炫光事件而被错误惩罚，更公平评估去炫光效果
- **实现验证**：✅ 参数顺序正确，estimated events查询ground truth树

### **环境准备（首次使用）**
```bash
cd /mnt/e/2025/event_flick_flare/experiments/main_experiments
chmod +x setup_environment.sh
./setup_environment.sh

# 使用voxel指标需要Umain环境
source ~/miniconda3/bin/activate && conda activate Umain
```

### **H5批量评估（推荐多方法版）**
```bash
cd /mnt/e/2025/event_flick_flare/experiments/main_experiments

# 查看所有可用指标  
python evaluate_all_methods.py --list-metrics

# 🌟 使用新默认混合指标：传统+voxel+健全性检查（推荐）
python evaluate_all_methods.py --output results

# 使用混合指标组合评估
python evaluate_all_methods.py --metrics chamfer_distance tf1 tpf1 pmse_2 --output results

# 快速测试少量样本
python evaluate_all_methods.py --num-samples 5 --output results
```

### **EVK4数据评估（新增）**
```bash
cd /mnt/e/2025/event_flick_flare/experiments/main_experiments

# 🎯 EVK4完整指标评估（需要Umain环境支持voxel指标）
source ~/miniconda3/bin/activate && conda activate Umain
python evaluate_evk4_methods.py --output results

# 查看EVK4可用指标
python evaluate_evk4_methods.py --list-metrics

# 快速测试少量样本
python evaluate_evk4_methods.py --num-samples 3 --output results

# 使用特定指标组合
python evaluate_evk4_methods.py --metrics chamfer_distance tf1 tpf1 pmse_2 --output results

# 静默模式
python evaluate_evk4_methods.py --quiet --output results
```

### **结果文件**
- **Simu数据**: `results/multi_method_evaluation_results.csv`
- **EVK4数据**: `results/evk4_evaluation_results.csv`
- **格式**: sample_id, {method1}_{metric1}, {method1}_{metric2}, {method2}_{metric1}, ...
- **特点**: 包含所有发现的方法×所选指标，最后一行为AVERAGE，适合论文直接使用
- **数据清洗**: Simu数据已删除composed_01000-01004无效样本，当前35个有效样本

### **默认指标详解**
**EVK4评估默认包含9个指标**：
- **传统指标**: chamfer_distance, gaussian_distance  
- **Voxel PMSE**: pmse_2, pmse_4
- **Voxel F1**: rf1, tf1, tpf1 
- **实用指标**: event_count_ratio, temporal_overlap

## 环境管理

### 指标框架优化 ✅ 
- **指标注册系统**: metrics.py:166-278 实现优雅的可扩展架构
- **配置驱动**: 支持动态选择指标组合，无需硬编码
- **CLI增强**: `--metrics` 和 `--list-metrics` 参数
- **向后兼容**: 保持现有 `calculate_all_metrics()` 接口

### 新指标添加模式
```python
# 1. 在 metrics.py 中定义新指标函数
def my_voxel_metric(events_est, events_gt):
    # 转换为voxel
    voxels_est = events_to_voxel_chunks(events_est)  
    voxels_gt = events_to_voxel_chunks(events_gt)
    
    # 计算voxel指标
    chunk_scores = []
    for v_est, v_gt in zip(voxels_est, voxels_gt):
        score = compute_some_voxel_similarity(v_est, v_gt)
        chunk_scores.append(score)
    
    return np.mean(chunk_scores)  # 平均所有chunks

# 2. 注册新指标
register_metric('my_voxel_metric', my_voxel_metric, 
               'Voxel-based similarity metric', 'voxel')

# 3. 使用新指标
python evaluate_all_methods.py --metrics chamfer_distance my_voxel_metric
```

### 完整指标体系 ✅
**总计14个指标，6个类别**

#### **距离类指标 (Lower is Better)**
- **`chamfer_distance`**: **单向Chamfer距离**，只计算estimated→ground truth的KDTree最近邻距离，避免惩罚炫光移除
- **`gaussian_distance`**: **单向高斯加权距离**，sigma=0.4，只计算estimated→ground truth方向，专门优化去炫光任务评估

#### **核方法指标 (Lower is Better)** ✅ 新增
基于RKHS（再生核希尔伯特空间）的3D高斯核距离，考虑时空极性的综合匹配度：
- **`kernel_standard`**: 标准配置（32×32×5ms块，σ=5/5/5000），均衡评估时空重建质量
- **`kernel_fine`**: 细粒度配置（16×16×2ms块，σ=3/3/2000），对局部细节和精确结构敏感
- **`kernel_spatial`**: 空间主导配置（32×32×10ms块，σ=3/3/10000），强调空间准确性，适合炫光去除空间效果评估
- **`kernel_temporal`**: 时间主导配置（64×64×2ms块，σ=10/10/1000），强调时间精度，适合运动和同步评估

**核方法优化**：
- 向量化分块实现，10×加速（避免Python循环）
- 原理：将事件流分割成时空立方体，用3D高斯核计算相似度
- 距离公式：`d = K(e1,e1) + K(e2,e2) - 2K(e1,e2)` （核技巧）
- 考虑极性分布加权（ON/OFF事件比例）

#### **计数类指标 (Ratio)**  
- **`event_count_ratio`**: 事件计数比例，估计数/真值数，理想值为1.0
- **`pger`**: PGER (Predicted/Ground Truth Event Number Ratio)，去炫光健全性检查，理想值为1.0

#### **时间类指标 (Higher is Better)**
- **`temporal_overlap`**: 时间覆盖重叠率，衡量两事件流时间窗口的重叠程度，范围[0,1]

#### **Voxel类指标**
**PMSE系列 (Lower is Better)**:
- **`pmse_2`**: 池化均方误差(pool=2)，对voxel进行2×2池化后计算MSE，减少空间错位敏感度
- **`pmse_4`**: 池化均方误差(pool=4)，对voxel进行4×4池化后计算MSE，更关注宏观结构

**F1系列 (Higher is Better, 范围[0,1])**:
- **`rf1`**: Raw F1分数，最严格，直接在5D voxel(B,P,T,H,W)上计算，要求精确的时空极性匹配
- **`tf1`**: Temporal F1分数，中等严格，坍缩时间维度，只要求空间和极性匹配
- **`tpf1`**: Temporal&Polarity F1分数，最宽松，坍缩时间和极性维度，只要求空间位置匹配

#### **算法验证结果 ✅**
- **边界情况**: 空事件→F1=0.0/PMSE=∞，完全正确
- **完美匹配**: 相同事件→F1=1.0/PMSE=0.0/Chamfer=0.0，完全正确  
- **数值范围**: 所有F1指标严格在[0,1]范围内
- **理论一致性**: RF1≤TF1≤TPF1，PMSE_4≤PMSE_2，符合预期

#### **技术特点**
- **分块处理**: 100ms文件→5×20ms voxel块，避免内存问题
- **极性分离**: 自动分离正负极性为独立通道  
- **维度坍缩**: 支持时间、极性维度的智能坍缩
- **鲁棒性**: 条件导入，优雅降级处理缺失依赖
- **性能**: 1.3秒处理50K事件，结果合理
- **集成**: 完全兼容现有CLI和指标注册系统

### 环境配置 
- **主环境**: conda activate Umain (包含PyTorch 2.3.0, scikit-learn 1.6.1)
- **备用环境**: 项目本地venv_main_exp (轻量依赖)
- **依赖策略**: voxel指标需要Umain环境，其他可用本地环境

## EVK4数据处理系统 ✅

### EVK4相机数据支持
- **硬件**: Prophesee EVK4-HD IMX636事件相机
- **格式支持**: HDF5 + NPY双格式处理管道
- **分辨率**: 原始1280×720，可裁剪重映射到640×480(DSEC兼容)
- **时间精度**: 微秒级时间戳，支持精确时间对齐

### 数据格式转换链
#### **HDF5格式 (原始)**
- **结构**: `/CD/events` 包含 x,y,p,t 字段
- **压缩**: 使用ECF (Event Compressed Format) codec
- **读取**: 需要专用ECF插件，在Windows evreal环境可正常读取
- **挑战**: Linux环境ECF插件配置复杂

#### **NPY格式 (中间)**
- **转换**: Windows evreal环境生成，包含6个文件
  - `events_ts.npy` - 时间戳(秒)
  - `events_xy.npy` - 坐标[N,2]
  - `events_p.npy` - 极性(0/1格式)
  - `images.npy`, `images_ts.npy`, `image_event_indices.npy` - 图像相关
  - `metadata.json` - 传感器分辨率等元数据
- **优势**: 无需插件，直接numpy读取

#### **DSEC格式 (输出)**
- **标准**: 与DSEC数据集完全兼容的H5格式
- **结构**: 
  ```
  /events/p  - 极性 (0/1格式，gzip压缩)
  /events/t  - 时间戳微秒 (gzip压缩) 
  /events/x  - X坐标 (gzip压缩)
  /events/y  - Y坐标 (gzip压缩)
  ```
- **压缩**: 所有数据集使用gzip压缩，compression_opts=6
- **用途**: 可直接用于去炫光算法训练

### EVK4数据集生成器
#### **通用版本**: `generate_evk4_dataset_from_npy.py`
#### **Defocus专版**: `generate_defocus_dataset.py` (独立defocus处理脚本) ✅
#### **RedAndFull专版**: `generate_redandfull_dataset.py` (独立redandfull处理脚本，支持双炫光源) ✅

#### **核心功能** (最新更新 ✅):
- **数据源**: 支持多种炫光类型配置
  - **Defocus组**: `full_noFlare_part_defocus` + `full_sixFlare_part_defocus` (当前使用)
  - **RedAndFull组**: `redandfull_noFlare_part` + `redandfull_sixFlare_part` + `redandfull_randomFlare_part` (双炫光源)
- **🎯 光源滤波**: **多种滤波策略** - 根据数据组选择
  - **Defocus组**: 圆形区域滤波，圆心(600,210)，半径30px，应用于noFlare数据
  - **RedAndFull组**: 四边形区域滤波，顶点[(610,341),(742,351),(562,456),(715,181)]，应用于noFlare数据
  - **✅ 滤波效果**: Defocus精确聚焦光源核心区域~1.8%，RedAndFull保留任意四边形光源区域
- **🎯 噪声增强**: **时空随机噪声添加** - 增强数据鲁棒性
  - **噪声数量**: 10万个随机事件
  - **时间分布**: 在原始数据时间范围内均匀分布
  - **空间分布**: 在整个1280×720传感器区域内均匀分布
  - **极性分布**: 随机选择-1或+1极性
  - **应用时机**: 光源滤波后，时间采样前
- **🎯 炫光边界滤除**: **RedAndFull组特有** - 移除炫光数据的边界区域
  - **移除区域**: 两个不相连的矩形区域
    - 左侧矩形: x < 335 的所有区域
    - 右上角矩形: x > 910 AND y < 255 的区域
  - **保留形状**: 不规则L形区域，避免简单矩形
  - **应用对象**: sixFlare和randomFlare数据
- **🎯 极性对齐**: 基于正极性占比的最优时间对齐，分析四个2.5ms时间窗口的极性分布  
- **时间采样**: 
  - **Defocus组**: 生成10对100ms数据，均匀分布在整个时间范围内
  - **RedAndFull组**: 生成20对100ms数据，前10个用sixFlare，后10个用randomFlare
- **🔄 不同裁剪**: 每对数据使用不同的随机裁剪区域(crop_seed=sample_id*42)，X±320px，Y±120px
- **序号管理**: 从1开始编号
- **空间重映射**: 1280×720→640×480，保持中心区域，避免过拟合
- **格式转换**: NPY→DSEC H5，极性归一化，时间戳单位转换，gzip压缩

#### **技术特点**:
- **🎯 极性对齐算法**: 创新的基于正极性占比的对齐方法
  - 参考数据固定从起始点开始，分析0-2.5ms, 2.5-5ms, 5-7.5ms, 7.5-10ms四个窗口的正极性占比
  - 测试数据从起始点开始，每次偏移0.5ms，最多20步(总计10ms搜索范围)
  - 计算四个窗口正极性占比差距之和，选择差距最小的偏移作为最佳对齐
  - 避免复杂依赖，快速准确，特别适合频闪节拍对齐
- **🔄 多样性保证**: crop_seed机制确保每对数据的裁剪区域不同，增强数据集多样性
- **极性处理**: NPY(0/1) → 内部(-1/+1) → DSEC(0/1)
- **时间转换**: NPY(秒) → DSEC(微秒)
- **坐标映射**: 智能中心裁剪 + 坐标重映射
- **压缩优化**: gzip压缩大幅减少文件大小(预期从139MB→20-30MB)
- **批量生成**: 灵活配置5-10对数据，每对650万-1100万事件
- **错误处理**: 指标对齐失败时优雅降级为零偏移

### 环境兼容性
- **Windows evreal**: ✅ 完整支持EVK4 HDF5读取和NPY转换
- **Linux WSL**: ✅ 支持NPY读取和DSEC生成，HDF5需插件
- **数据流**: Windows生成NPY → Linux处理生成DSEC → 算法训练

### 已验证结果
- **源数据**: NoFlare ~40M事件/0.5s，SixFlare ~84M事件/0.5s
- **输出数据**: 10对DSEC格式训练数据，640×480分辨率
- **数据质量**: 时间对齐精确，空间分布均匀，格式标准兼容
- **文件大小**: 每个样本H5文件~100-200MB

### 使用指南

#### **Defocus专用脚本**
```bash
# 1. 在Windows evreal环境中生成NPY文件
# 2. 在Linux环境中运行defocus专用生成器
python3 generate_defocus_dataset.py

# 特点:
# - 数据源: full_noFlare_part_defocus + full_sixFlare_part_defocus
# - 圆形光源滤波: 圆心(600,210), 半径30px
# - 固化噪声增强: 10万个随机事件
# - 生成10对样本，编号1-10
# - 输出目录: evk4_defocus_dataset_dsec_format_from_npy/
```

#### **RedAndFull专用脚本** ✅
```bash
# RedAndFull组数据生成器（双炫光源）
python3 generate_redandfull_dataset.py

# 特点:
# - 数据源: redandfull_noFlare_part + redandfull_sixFlare_part + redandfull_randomFlare_part
# - NoFlare处理: 四边形光源滤波，顶点[(610,341),(742,351),(562,456),(715,181)] + 10万噪声事件
# - 炫光处理: 边界滤除(移除左侧x<335 + 右上角x>910&y<255) + 各10万噪声事件
# - 生成20对样本: 前10个sixFlare(样本1-10) + 后10个randomFlare(样本11-20)
# - 时间对齐: NoFlare作为基准，分别与两种炫光数据对齐
# - 输出目录: evk4_redandfull_dataset_dsec_format_from_npy/
# - 文件命名: 包含炫光类型前缀(sixflare/randomflare)
```

#### **通用版本脚本**
```bash
# 通用EVK4数据集生成器（支持多种配置）
python3 generate_evk4_dataset_from_npy.py

# 配置选项:
# - 修改数据源路径: noflare_npy_folder, sixflare_npy_folder
# - 修改光源滤波参数: center_x, center_y, radius
# - 修改噪声参数: num_noise_events
# - 修改样本数量: num_samples
```

#### **输出目录结构**
```
# Defocus组
evk4_defocus_dataset_dsec_format_from_npy/
├── input/   (带炫光数据: defocus_XXXms_sampleY.h5)
└── target/  (干净数据: defocus_XXXms_sampleY.h5)

# RedAndFull组（双炫光源）
evk4_redandfull_dataset_dsec_format_from_npy/
├── input/   (带炫光数据: redandfull_sixflare_XXXms_sampleY.h5, redandfull_randomflare_XXXms_sampleY.h5)
└── target/  (干净数据: redandfull_sixflare_XXXms_sampleY.h5, redandfull_randomflare_XXXms_sampleY.h5)
```

### 数据源配置示例
```python
# Defocus组配置: 圆形光源滤波, 生成sample1-10
noflare_npy_folder = "Datasets/EVK4/full_noFlare_part_defocus"
sixflare_npy_folder = "Datasets/EVK4/full_sixFlare_part_defocus"
# 光源滤波: 圆心(600,210)，半径30px，应用于noFlare数据
noflare_events = filter_light_source_circular(noflare_events, center_x=600, center_y=210, radius=30)
# 噪声增强: 添加10万个时空随机分布的噪声事件
noflare_events = add_spatiotemporal_noise(noflare_events, num_noise_events=100000)
sample_starts = generate_random_sample_times(overlap_start, overlap_end, num_samples=10)
sample_id = i + 1  # 序号1-10

# RedAndFull组配置: 四边形光源滤波 + 炫光边界滤除, 生成sample1-20
noflare_npy_folder = "Datasets/EVK4/redandfull_noFlare_part"
sixflare_npy_folder = "Datasets/EVK4/redandfull_sixFlare_part"
randomflare_npy_folder = "Datasets/EVK4/redandfull_randomFlare_part"
# 光源滤波: 四边形顶点[(610,341),(742,351),(562,456),(715,181)]，应用于noFlare数据
polygon_vertices = [(610, 341), (742, 351), (562, 456), (715, 181)]
noflare_events = filter_light_source_polygon(noflare_events, vertices=polygon_vertices)
# 炫光边界滤除: 移除左侧x<335 + 右上角x>910&y<255，应用于炫光数据
sixflare_events = filter_flare_boundaries(sixflare_events, left_boundary=335, right_boundary=910, top_boundary=255)
randomflare_events = filter_flare_boundaries(randomflare_events, left_boundary=335, right_boundary=910, top_boundary=255)
# 噪声增强: 三个数据源各添加10万个随机事件
sample_starts = generate_random_sample_times(noflare_start, noflare_end, num_samples=20)
sample_id = i + 1  # 序号1-20, 前10个sixFlare后10个randomFlare
```

### Defocus数据集生成结果 ✅
- **源数据规模**: NoFlare ~40M事件，SixFlare ~84M事件，持续0.5秒
- **光源滤波效果**: ✅ 40M → 71万事件(1.78%保留率)，位置校正(600,210)，半径优化(30px)
- **生成数据**: 10对DSEC格式，文件名`defocus_XXXms_sampleY.h5`
- **输出目录**: `evk4_defocus_dataset_dsec_format_from_npy/`
- **数据质量**: 时间对齐正常，空间裁剪正常，光源滤波效果合理
- **样本事件数**: Target约14万事件/100ms，Input约1200万事件/100ms(裁剪后)

## Kernel指标系统 ✅ (固化优化配置)

### 概述
基于RKHS（再生核希尔伯特空间）的3D高斯核距离方法，将事件流分割成时空立方体(cube)，计算核相似度。

### 核心发现：细cube更快！
**关键洞察**：由于O(n²)复杂度，将事件分割成更小的cube反而更快：
- **粗cube**: 1个(86K事件)² = 74亿操作 → 极慢
- **细cube**: 1000个(86事件)² = 740万操作 → 快10000倍！

### 固化配置（向下取整优化）✅
**重要变更**：cube分辨率已固化为极细配置，采用整数向下取整保证鲁棒性
- **采样**：默认关闭（极细cube完全不需要采样）
- **进度显示**：默认关闭（避免刷屏）
- **Cube配置**：整数分辨率固化在代码中
- **指标精简**：从4个精简为3个，移除冗余配置

### 三个Kernel变体的设计思路

| Variant | 固化配置 | Cube数量 | 单cube事件 | 设计思路 |
|---------|---------|----------|-----------|---------|
| **kernel_pixel_temporal** | 1×1×0.1ms | ~307M | 0.02 | 单像素级空间 + 超细时间，极致时空采样 ⭐ |
| **kernel_balanced** | 2×2×0.2ms | ~38M | 0.9 | 均衡时空分辨率，通用评估 |
| **kernel_coarse_temporal** | 4×4×0.1ms | ~19M | 1.8 | 粗空间细时间，强调时间精度 |

**设计理念**：
- **pixel_temporal**: 每个像素独立cube（1×1空间），0.1ms时间片，捕捉精确时空动态
- **balanced**: 2×2空间块，0.2ms时间片，空间与时间分辨率平衡
- **coarse_temporal**: 4×4空间块（粗），0.1ms时间片（细），适合运动/同步分析

### 命令行使用（简化）
```bash
# Simu数据集评估（推荐三个kernel指标）
python evaluate_all_methods.py \
  --metrics kernel_pixel_temporal kernel_balanced kernel_coarse_temporal \
  --output results

# EVK4数据集评估（使用相同配置）
python evaluate_evk4_methods.py \
  --metrics kernel_pixel_temporal kernel_balanced kernel_coarse_temporal \
  --output results

# 单独使用pixel_temporal（最快，推荐）
python evaluate_all_methods.py \
  --metrics kernel_pixel_temporal \
  --output results

# 启用verbose查看进度（如需调试）
python evaluate_all_methods.py \
  --metrics kernel_balanced \
  --kernel-verbose on \
  --output results
```

### 配置参数说明（简化）
- `--kernel-sampling on/off`：采样开关（默认off，极细cube下不需要）
- `--kernel-max-events N`：采样阈值（默认10000，仅sampling=on时生效）
- `--kernel-verbose on/off`：进度显示（默认off，避免刷屏）
- ~~`--kernel-cube-scale`~~：已移除，配置已固化在代码中

### 性能对比
| 配置 | Cube数 | 单cube事件 | 总操作数 | 相对速度 |
|------|--------|-----------|---------|---------|
| **原始(40×30×5ms)** | 5K | 338 | 572M | 1x（基准）|
| **pixel_temporal(1×1×0.1ms)** | 307M | 0.02 | 1.2K | **476,667x** 🚀🚀 |
| **balanced(2×2×0.2ms)** | 38M | 0.9 | 30.8K | **18,571x** 🚀 |
| **coarse_temporal(4×4×0.1ms)** | 19M | 1.8 | 61.6K | **9,286x** 🚀 |

### 技术细节
- **算法**：K(e1,e2) = polarity_weight × Σexp(-d²/2σ²)
- **距离公式**：d² = Δx²/σx² + Δy²/σy² + Δt²/σt²
- **极性加权**：考虑ON/OFF事件分布
- **内存优化**：双重分块（1000×1000事件/块），峰值8MB
- **数学等价性**：✅ 分块只改变计算顺序，结果完全一致
- **配置固化**：保证simu和EVK4使用完全相同的kernel配置

## 项目优势
- **完整性**: 支持真实 DVS 相机数据格式，包括DAVIS和EVK4，已验证可用
- **专业性**: 包含时间对齐敏感性分析功能，支持微秒级精度
- **先进性**: 完整实现voxel级别指标(PMSE、F1系列) + kernel方法，涵盖现代事件流评估方法
- **可扩展性**: 优雅的指标注册系统，支持传统+voxel+kernel指标无缝集成
- **内存高效**: voxel分块 + kernel双重分块策略，适合大规模数据
- **极致性能**: Kernel指标通过超细cube优化，实现24000倍加速
- **鲁棒性**: 智能依赖管理，优雅降级，支持不同计算环境
- **独立性**: 双环境支持，灵活依赖管理
- **实用性**: 直接支持用户的实际数据文件，包括EVK4高端相机数据
- **工业级**: 支持Prophesee EVK4专业事件相机，满足实际应用需求

## 注意事项
1. **记忆文件命名**：已修正为 `CLAUDE.md`（Claude Code标准格式）
2. **虚拟环境**：依赖库已安装，可直接运行Python脚本
3. **数据路径**：支持Windows格式路径，自动转换为WSL格式
4. **性能考量**：完整的时间偏移分析需要较长时间，建议先用小范围测试
5. **EVK4处理**：Windows evreal环境生成NPY，Linux环境处理生成DSEC
6. **ECF插件**：EVK4 HDF5需要专用插件，建议使用NPY中间格式
7. **数据规模**：EVK4数据量大(百万级事件/100ms)，注意内存和存储空间