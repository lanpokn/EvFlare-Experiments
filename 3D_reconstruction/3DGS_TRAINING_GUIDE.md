# 3DGS自动训练脚本使用指南

## 🎯 **项目目标**
对比不同H5事件数据源对同一重建方法的3DGS训练效果，实现：
- **原始图像** vs **重建图像**的3DGS训练对比
- **三种H5数据源**：original, Unet, Unetsimple的同一重建方法对比
- **自动化训练、测试、评估**的完整workflow

## 📋 **使用步骤**

### **步骤1：准备JSON配置文件**
```bash
# 在WSL或Linux环境下执行
cd /mnt/e/2025/event_flick_flare/experiments/3D_reconstruction

# 为指定重建方法生成配置文件（推荐）
python generate_json_configs.py lego2 spade-e2vid

# 或为所有方法生成配置文件
python generate_json_configs.py lego2
```

**生成结果**：
- `transforms_train_original.json` - 原始训练图像配置
- `transforms_train_spade_e2vid_original.json` - spade-e2vid原始H5数据源
- `transforms_train_spade_e2vid_Unet.json` - spade-e2vid经过Unet处理的H5数据源  
- `transforms_train_spade_e2vid_Unetsimple.json` - spade-e2vid经过Unetsimple处理的H5数据源
- `training_methods_spade_e2vid.txt` - 训练方法列表文件

### **步骤2：切换到Windows 3DGS环境**
```cmd
# 打开Windows命令行，切换到3DGS目录
cd "E:\2025\event_flick_flare\experiments\3D_reconstruction\gaussian-splatting"

# 激活3DGS Python环境（根据实际环境调整）
conda activate 3dgs
# 或
activate 3dgs
```

### **步骤3：运行自动训练脚本**
```cmd
# 训练指定方法的所有H5数据源（推荐）
..\auto_train_3dgs.bat lego2 spade_e2vid

# 或使用默认参数（lego2 + spade_e2vid）
..\auto_train_3dgs.bat
```

### **步骤4：查看训练结果**
训练完成后，所有结果保存在：
```
datasets\lego2\3dgs_results\
├── weights\           # 训练权重文件
│   ├── original\      # 原始图像训练权重
│   ├── spade_e2vid_original\   # spade-e2vid原始H5
│   ├── spade_e2vid_Unet\       # spade-e2vid Unet H5
│   └── spade_e2vid_Unetsimple\ # spade-e2vid Unetsimple H5
├── renders\           # 测试渲染结果
│   ├── original\      # 原始图像的测试渲染
│   ├── spade_e2vid_original\   # spade-e2vid原始H5的测试渲染
│   ├── spade_e2vid_Unet\       # spade-e2vid Unet H5的测试渲染
│   └── spade_e2vid_Unetsimple\ # spade-e2vid Unetsimple H5的测试渲染
├── metrics\           # 评估指标文件
│   ├── original_metrics.txt
│   ├── spade_e2vid_original_metrics.txt
│   ├── spade_e2vid_Unet_metrics.txt
│   └── spade_e2vid_Unetsimple_metrics.txt
└── training_summary_spade_e2vid.txt  # 训练摘要报告
```

## 🔧 **训练流程详解**

### **自动化流程**
脚本会自动执行以下步骤：
1. **JSON切换**：`transforms_train_original.json` → `transforms_train.json`
2. **3DGS训练**：`python train.py -s ../datasets/lego2 -m output/lego2_original --iterations 7000 --grayscale`
3. **权重备份**：复制训练权重到结果目录
4. **测试渲染**：`python render.py -m output/lego2_original --grayscale`
5. **渲染备份**：复制渲染结果到结果目录
6. **指标计算**：`python metrics.py -m output/lego2_original`
7. **清理临时文件**：删除临时输出目录节省空间

### **训练参数**
- **迭代次数**：7000（可在脚本中调整）
- **模式**：灰度模式（`--grayscale`）
- **输出目录**：`output/lego2_{配置名称}/`

## 📊 **结果分析建议**

### **视觉对比**
1. 查看 `renders/` 目录中不同配置的渲染结果
2. 对比相同视角下不同H5数据源的视觉质量
3. 观察原始图像训练vs重建图像训练的差异

### **定量分析**
1. 查看 `metrics/` 目录中的指标文件
2. 对比PSNR、SSIM、LPIPS等评估指标
3. 分析不同H5处理方法对3DGS训练效果的影响

### **典型分析问题**
- **原始图像 vs 重建图像**：哪种训练数据质量更好？
- **Unet vs Unetsimple处理**：哪种H5处理方法保留更多有用信息？
- **重建方法影响**：同一H5处理下，不同重建方法的3DGS效果如何？

## 🚀 **扩展使用**

### **测试其他重建方法**
```bash
# 生成firenet方法的配置
python generate_json_configs.py lego2 firenet
```
```cmd
# Windows下训练firenet方法
..\auto_train_3dgs.bat lego2 firenet
```

### **测试其他数据集**
```bash
# 为ship数据集生成配置
python generate_json_configs.py ship spade-e2vid
```
```cmd
# Windows下训练ship数据集
..\auto_train_3dgs.bat ship spade_e2vid
```

## ⚠️ **注意事项**

### **环境要求**
- **Linux/WSL**：运行JSON生成脚本
- **Windows + GPU**：运行3DGS训练（必须有CUDA支持）
- **Python环境**：3DGS专用环境（避免环境冲突）

### **路径要求**
- 脚本假设在 `gaussian-splatting/` 目录下运行
- 数据集在 `../datasets/` 目录
- 确保Windows和WSL路径映射正确

### **存储空间**
- 每个配置的权重文件约100-200MB
- 每个配置的渲染结果约50MB
- 建议预留2-3GB存储空间

### **时间估计**
- 每个配置训练时间：20-40分钟（取决于GPU性能）
- 4个配置总时间：1.5-3小时
- 建议在GPU空闲时运行

## 🐛 **故障排除**

### **常见错误**
1. **配置文件不存在**：先运行JSON生成脚本
2. **CUDA内存不足**：降低迭代次数或清理其他GPU进程
3. **路径错误**：检查数据集目录和脚本路径
4. **Python环境错误**：确保3DGS环境正确激活

### **验证步骤**
```cmd
# 验证3DGS环境
python -c "import torch; print(torch.cuda.is_available())"

# 验证数据集路径
dir ..\datasets\lego2

# 验证配置文件
dir ..\datasets\lego2\transforms_train_*.json
```

## 📈 **成功案例**
根据lego2数据集的成功经验：
- **8/8种重建方法全部成功**
- **完美200:200图像对应**
- **最佳重建质量**：ET-Net (MSE=0.037)
- **数据集大小**：1.5GB

期望3DGS训练也能达到类似的成功率和质量水平！