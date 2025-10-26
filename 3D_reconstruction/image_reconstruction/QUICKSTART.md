# 快速上手指南 (5分钟入门)

## 🎯 核心功能

**输入**: H5事件数据文件
**输出**: 重建图像 (8种方法 × 200张图像)

---

## ⚡ 三种使用方式

### 方式1: 命令行 (最简单)
```bash
cd image_reconstruction

# 激活环境并运行
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && python h5_to_images.py lego2
```

### 方式2: Shell脚本 (一键启动)
```bash
cd image_reconstruction
./quick_start.sh lego2
```

### 方式3: Python代码 (可自定义)
```bash
cd image_reconstruction
python example_usage.py
# 然后选择选项 1
```

---

## 📂 输入数据位置

脚本会自动扫描：
```
../datasets/<dataset_name>/events_h5/*.h5
```

示例：`../datasets/lego2/events_h5/` 中的所有H5文件（除backup）

---

## 📦 输出结果位置

重建图像会保存到：
```
../datasets/<dataset_name>/reconstruction_<suffix>/evreal_<method>/
```

示例：
- `../datasets/lego2/reconstruction_original/evreal_spade_e2vid/0001.png`
- `../datasets/lego2/reconstruction_Unet/evreal_ssl_e2vid/0200.png`

---

## 🔍 检查结果

```bash
# 查看生成了多少重建图像
find ../datasets/lego2/reconstruction_* -name "*.png" | wc -l

# 查看某个方法的重建结果
ls ../datasets/lego2/reconstruction_original/evreal_spade_e2vid/

# 查看重建方法列表
ls ../datasets/lego2/reconstruction_original/
```

---

## ⏱️ 预计时间

| H5文件 | 事件数量 | 预计时间 | 输出图像 |
|--------|----------|----------|----------|
| ~100万事件 | 小型 | 5-8分钟 | 1600张 |
| ~200万事件 | 中型 | 8-12分钟 | 1600张 |
| ~500万事件 | 大型 | 12-20分钟 | 1600张 |

**注**: 时间基于8种重建方法，实际可能因系统性能有所不同

---

## ❓ 常见问题

### Q1: 环境激活失败？
```bash
# 确认conda已初始化
conda info

# 手动激活
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Umain2
conda info --envs  # 查看当前环境
```

### Q2: 找不到H5文件？
```bash
# 检查H5文件是否存在
ls -lh ../datasets/<dataset_name>/events_h5/*.h5

# 确认路径正确
pwd  # 应该在 image_reconstruction 目录下
```

### Q3: EVREAL重建失败？
```bash
# 检查EVREAL路径
ls /mnt/e/2025/event_flick_flare/EVREAL-main/EVREAL-main/eval.py

# 查看日志输出了解具体错误
```

### Q4: 部分方法失败是正常的吗？
是的，预期4-8种方法成功。E2VID+、FireNet+、HyperE2VID可能因路径兼容性或内存问题失败。

---

## 🎓 下一步

1. **查看重建图像**: 使用图像查看器查看 `reconstruction_*/evreal_*/` 中的PNG文件
2. **评估质量**: 使用主项目的 `calculate_reconstruction_metrics.py` 计算PSNR/SSIM/LPIPS
3. **自定义修改**: 参考 `README.md` 中的"独立修改指南"

---

**快速问题？** 查看 `README.md` 获取详细信息
