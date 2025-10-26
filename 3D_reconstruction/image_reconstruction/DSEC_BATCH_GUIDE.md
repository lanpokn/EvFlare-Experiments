# DSEC数据集批量重建指南 (支持断点续存)

## 🎯 功能说明

自动处理DSEC_data目录下所有方法的H5文件，进行图像重建。

### **核心特性**
- ✅ 自动扫描所有方法目录（排除visualize）
- ✅ 每5个H5文件采样1个进行重建
- ✅ 每个H5生成40张重建图像（可配置）
- ✅ 输出结构与DSEC_data一致（H5文件→文件夹）
- 🆕 **断点续存**: 中断后可继续，自动跳过已完成任务
- 🆕 **新处理顺序**: 先处理同名H5的所有方法，再处理下一个H5

### **断点续存原理**
- 进度文件: `DSEC_data_reconstructed/.batch_progress.json`
- 双重检查: 进度文件记录 + 输出目录检测
- 任务ID格式: `方法名:H5文件名`
- 可随时Ctrl+C中断，下次运行自动继续

---

## 🔄 处理顺序变化

### **旧版本处理顺序**
```
for 方法 in [input, inputpfda, output_physics_noRandom_method, ...]:
    for H5文件 in [file001, file006, file011, ...]:
        处理(方法, H5文件)
```
**问题**: 如果在处理input方法的第100个文件时中断，所有其他方法的前99个文件都未处理，难以恢复。

### **新版本处理顺序** (推荐)
```
for H5文件 in [file001, file006, file011, ...]:
    for 方法 in [input, inputpfda, output_physics_noRandom_method, ...]:
        处理(方法, H5文件)
```
**优势**:
- 同名H5文件（不同方法目录）一起处理完成
- 更合理的断点续存粒度
- 便于验证和调试某个具体H5文件的重建结果

---

## ⚡ 运行命令

### **快速启动** (推荐)
```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/image_reconstruction

# 使用Shell脚本（包含环境检查和进度提示）
./RUN_DSEC_BATCH.sh
```

### **直接运行** (40张图像)
```bash
cd /mnt/e/BaiduSyncdisk/2025/event_flick_flare/experiments/3D_reconstruction/image_reconstruction

source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

python batch_dsec_reconstruction.py \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data" \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data_reconstructed"
```

### **自定义图像数量**
```bash
# 生成80张图像
python batch_dsec_reconstruction.py \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data" \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data_reconstructed" \
  80
```

### **中断后继续**
```bash
# 中断后再次运行相同命令即可自动继续
# 脚本会自动加载 .batch_progress.json 并跳过已完成任务
./RUN_DSEC_BATCH.sh
```

---

## 📁 输入输出结构

### **输入结构** (DSEC_data)
```
DSEC_data/
├── input/                  # 方法1 (839个H5文件)
│   ├── file001.h5
│   ├── file002.h5
│   └── ...
├── output_physics_noRandom_method/  # 方法2 (839个H5文件)
│   ├── file001.h5
│   └── ...
├── inputpfda/             # 方法3
└── visualize/             # ← 自动跳过
```

### **输出结构** (DSEC_data_reconstructed)
```
DSEC_data_reconstructed/
├── input/                  # 方法1重建结果
│   ├── file001/           # H5文件→文件夹
│   │   ├── evreal_e2vid/          # 39张PNG
│   │   ├── evreal_firenet/        # 39张PNG
│   │   ├── evreal_ssl-e2vid/      # 39张PNG
│   │   └── ...
│   ├── file006/           # 每5个取1个
│   ├── file011/
│   └── ...
├── output_physics_noRandom_method/  # 方法2重建结果
│   ├── file001/
│   └── ...
└── inputpfda/             # 方法3重建结果
```

---

## 📊 预计处理量

### **采样计算**
- 每个方法: 839个H5文件
- 采样间隔: 每5个取1个
- 采样结果: 839 ÷ 5 ≈ 168个H5文件

### **假设有14个方法目录**
- 总任务数: 168个H5 × 14个方法 = 2,352个任务
- 预计总耗时: 2,352 × 12分钟 ≈ 28,224分钟 ≈ **20天**

### **建议**
1. **使用断点续存**: 可随时中断和恢复，无需一次性完成
2. **使用screen/tmux**: 避免SSH断开（或直接Ctrl+C中断，下次继续）
3. **监控磁盘空间**: 每个方法约50-100GB输出
4. **先测试小批量**: 可以手动编辑采样间隔测试

---

## 🔧 断点续存详细说明

### **进度文件格式** (.batch_progress.json)
```json
{
  "completed": [
    "input:real_flare_zurich_city_03_a_t0ms_20251021_210546",
    "inputpfda:real_flare_zurich_city_03_a_t0ms_20251021_210546",
    "output_physics_noRandom_method:real_flare_zurich_city_03_a_t0ms_20251021_210546"
  ],
  "total_completed": 3,
  "last_update": "2025-10-26 14:30:00"
}
```

### **双重检查机制**
脚本使用两种方式检查任务是否完成：
1. **进度文件检查**: 读取`.batch_progress.json`中的completed列表
2. **输出目录检查**: 检查输出目录是否存在且包含PNG文件

这样即使进度文件丢失，也能自动恢复已完成的任务。

### **手动清除进度** (重新开始)
```bash
# 删除进度文件即可重新开始
rm "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data_reconstructed/.batch_progress.json"
```

### **查看进度**
```bash
# 查看已完成任务数
cat DSEC_data_reconstructed/.batch_progress.json | grep "total_completed"

# 查看已完成任务列表
cat DSEC_data_reconstructed/.batch_progress.json | grep -A 999 "completed"
```

---

## 🔧 技术参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_interval` | 5 | 每5个H5取1个 |
| `num_images` | 40 | 每个H5生成40张图像 |
| `num_methods` | 8 | EVREAL重建方法数 |

### **修改采样间隔**
编辑 `batch_dsec_reconstruction.py` 第46行:
```python
self.sample_interval = 10  # 改为每10个取1个
```

---

## ⚠️ 重要提示

### **1. 环境要求**
- **必须**: Umain2 conda环境
- **禁止**: 安装新库或升级现有库

### **2. 运行建议**
```bash
# 使用screen避免SSH断开
screen -S dsec_reconstruction

# 进入环境
cd image_reconstruction
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

# 运行脚本
python batch_dsec_reconstruction.py \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data" \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data_reconstructed"

# 断开screen: Ctrl+A, D
# 重新连接: screen -r dsec_reconstruction
```

### **3. 监控进度**
```bash
# 查看已处理文件数
find DSEC_data_reconstructed -type d -name "evreal_*" | wc -l

# 查看输出大小
du -sh DSEC_data_reconstructed/*

# 查看Python进程
ps aux | grep batch_dsec_reconstruction
```

---

## 📝 输出日志示例

```
============================================================
DSEC数据集批量H5重建
============================================================
输入目录: /path/to/DSEC_data
输出目录: /path/to/DSEC_data_reconstructed
采样间隔: 每5个取1个
重建图像数: 40 张
============================================================

✓ 发现方法: input (839 个H5文件)
✓ 发现方法: output_physics_noRandom_method (839 个H5文件)
...

共找到 14 个方法目录

############################################################
方法进度: 1/14
############################################################

============================================================
处理方法: input
============================================================
  采样: 839 → 168 个文件 (间隔=5)

[1/168] 处理: file001
  ✅ 成功: 5 种方法

[2/168] 处理: file006
  ✅ 成功: 5 种方法

...
```

---

## 🎯 快速测试命令

### **测试单个H5文件**
```bash
cd image_reconstruction
source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2

# 测试40张图像
python single_h5_reconstruction.py \
  "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/DSEC_data/output_physics_noRandom_method/real_flare_zurich_city_03_a_t0ms_20251021_210546.h5" \
  "test_output_40imgs" \
  40
```

---

---

## 🆕 更新日志

### **2025-10-26 - 断点续存版本**
- ✅ **断点续存功能**: 使用`.batch_progress.json`记录已完成任务
- ✅ **处理顺序优化**: 外层循环改为H5文件，内层循环为方法
- ✅ **双重检查机制**: 进度文件 + 输出目录检测
- ✅ **Shell脚本增强**: 自动检测进度，显示中断/恢复提示
- ✅ **文档完善**: 详细的使用说明和断点续存原理

### **2025-10-25 - 初始版本**
- ✅ 基础批量处理功能
- ✅ 采样间隔配置
- ✅ 可配置图像数量

---

**作者**: Claude Code Assistant (Linus模式)
**最后更新**: 2025-10-26
**配置**: 每5个H5取1个，每个生成40张重建图像
