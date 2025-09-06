# 主实验框架 - Claude Code 项目记忆

## 项目概述
**事件流去炫光结果的定量评估框架**，专门用于评估**已经处理完成**的去炫光结果与真值数据的对比。

**重要澄清**：此框架 **不执行** 去炫光处理，仅负责评估外部方法的处理结果。

## 已完成的框架结构

### 核心模块
1. **data_loader.py** - 事件数据加载抽象层（支持多种格式）
2. **metrics.py** - 评估度量计算（用户提供的 Chamfer Distance 和 Gaussian Distance）
3. **methods.py** - 方法结果加载器（**已简化**，移除了处理逻辑）
4. **evaluator.py** - 实验流程协调器（**已更新**，直接比较结果）
5. **run_main_experiment.py** - 主执行脚本（**已重构**）
6. **__init__.py** - 包初始化文件
7. **requirements.txt** - 依赖列表
8. **README.md** - 详细文档（已更新）

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
- **文件**: `evaluate_simu_pairs_optimized.py` (单方法版)
- **文件**: `evaluate_all_methods.py` (多方法版)
- **功能**: 批量评估H5文件的去炫光效果
- **数据结构**: 
  - **真值**: `background_with_light_events_test/` (包含 `*_bg_light.h5`)
  - **各种方法**: 其他所有文件夹 (包含 `*_bg_flare.h5` 等各种方法的结果)
  - **评估原则**: 计算各种方法相对于真值的指标，方法间不互相比较
- **特点**:
  - 仅计算 chamfer_distance 和 gaussian_distance 
  - 结果统一保存到 `results/` 目录
  - 自动生成包含平均数行的CSV文件
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

**环境准备（首次使用）**：
```bash
cd /mnt/e/2025/event_flick_flare/experiments/main_experiments
chmod +x setup_environment.sh
./setup_environment.sh
```

**H5批量评估（推荐多方法版）**：
```bash
cd /mnt/e/2025/event_flick_flare/experiments/main_experiments

# 评估所有方法相对于真值 (推荐)
python evaluate_all_methods.py --output results

# 单个方法评估
python evaluate_simu_pairs_optimized.py --output results

# 快速测试少量样本
python evaluate_all_methods.py --num-samples 5 --output results
```

**结果文件**：
- **单方法版**: `results/h5_pairs_evaluation_results.csv`
  - **格式**: sample_id, chamfer_distance, gaussian_distance
- **多方法版**: `results/multi_method_evaluation_results.csv` (推荐)
  - **格式**: sample_id, {method1}_chamfer_distance, {method1}_gaussian_distance, {method2}_chamfer_distance, {method2}_gaussian_distance, ...
  - **特点**: 包含所有发现的方法，最后一行为AVERAGE，适合论文直接使用

## 项目优势
- **完整性**: 支持真实 DVS 相机数据格式，已验证可用
- **专业性**: 包含时间对齐敏感性分析功能
- **独立性**: 自包含虚拟环境，所需依赖已安装
- **实用性**: 直接支持用户的实际数据文件，已测试通过

## 注意事项
1. **记忆文件命名**：已修正为 `CLAUDE.md`（Claude Code标准格式）
2. **虚拟环境**：依赖库已安装，可直接运行Python脚本
3. **数据路径**：支持Windows格式路径，自动转换为WSL格式
4. **性能考量**：完整的时间偏移分析需要较长时间，建议先用小范围测试