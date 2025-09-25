# 通用事件相机重建Pipeline使用说明

现在的pipeline已经完全通用化，支持任意数据集！

## 🎯 支持的数据集
- lego (原有)
- ship (新增)
- hotdog (已有)
- 未来任何遵循 `xxx_flare + xxx_normal` 格式的数据集

## 🚀 使用方法

### 基本用法
```bash
# 处理默认数据集 (lego)
python run_full_pipeline.py

# 处理指定数据集
python run_full_pipeline.py ship
python run_full_pipeline.py hotdog
```

### 高级选项
```bash
# 自动合并数据集（如果尚未合并）
python run_full_pipeline.py ship --auto-merge

# 跳过特定步骤
python run_full_pipeline.py ship --skip-preprocess  # 跳过图像预处理
python run_full_pipeline.py ship --skip-dvs        # 跳过DVS仿真
python run_full_pipeline.py ship --skip-convert    # 跳过格式转换
python run_full_pipeline.py ship --skip-reconstruct # 跳过EVREAL重建

# 组合使用
python run_full_pipeline.py ship --auto-merge --skip-preprocess
```

### 查看帮助
```bash
python run_full_pipeline.py --help
```

## 📁 数据集结构要求

每个数据集需要遵循以下结构：
```
datasets/
├── xxx_flare/          # 炫光版本（作为训练集）
│   ├── train/
│   ├── transforms_train.json
│   └── points3d.ply
├── xxx_normal/         # 正常版本（作为测试集）
│   ├── train/
│   ├── transforms_train.json  
│   └── points3d.ply
└── xxx/                # 合并后的完整数据集
    ├── train/          # 来自xxx_flare
    ├── test/           # 来自xxx_normal
    ├── transforms_train.json
    ├── transforms_test.json
    └── points3d.ply
```

## 🔄 完整工作流程

1. **数据集检查**: 自动检查原始和合并数据集
2. **图像预处理**: 将训练图像转换为DVS输入格式
3. **DVS事件仿真**: 生成事件数据流
4. **格式转换**: DVS → EVREAL + H5格式
5. **图像重建**: 多方法重建 (E2VID, FireNet等)
6. **结果验证**: 自动统计和检查

## 📊 输出结果

每个数据集的完整输出：
```
datasets/xxx/
├── events_dvs/         # DVS事件数据
├── events_evreal/      # EVREAL格式数据
├── events_h5/          # H5格式数据
└── reconstruction/     # 重建结果
    ├── evreal_e2vid/
    ├── evreal_firenet/
    └── ...
```

## ⚡ 性能优化

- 使用 `--skip-*` 选项跳过已完成的步骤
- 在正确的conda环境下运行：`conda activate Umain2`
- 确保有足够的磁盘空间（每个数据集约需要几GB）

## 🛠️ 故障排除

1. **数据集不存在**: 确保有 `xxx_flare` 和 `xxx_normal` 目录
2. **合并数据集缺失**: 使用 `--auto-merge` 或手动运行 `python merge_datasets.py`
3. **环境问题**: 确保在 `Umain2` conda环境下运行
4. **DVS仿真失败**: 检查DVS-Voltmeter路径和依赖

这个通用pipeline完全不需要修改核心代码，只需要准备好数据集即可处理任意场景！