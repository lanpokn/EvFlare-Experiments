# 快速参考 - H5到图像重建

## ⚡ 一行命令

```bash
cd image_reconstruction && source ~/miniconda3/etc/profile.d/conda.sh && conda activate Umain2 && python single_h5_reconstruction.py "<h5文件路径>" "<输出目录名>"
```

---

## 📁 关键文件

| 文件 | 说明 |
|------|------|
| `single_h5_reconstruction.py` | **核心脚本** - 完整重建流程 |
| `modules/` | 依赖模块 (复用主项目) |
| `SUCCESSFUL_WORKFLOW.md` | 完整技术文档 |
| `README.md` | 功能说明 |

---

## ✅ 验证成功

- **测试文件**: `real_flare_zurich_city_03_a_t0ms_20251021_210546.h5`
- **事件数量**: 92.89万
- **成功率**: 8/8方法
- **输出**: 995张重建图像

---

## 🔧 核心技术

1. **极性转换**: -1/1 → 0/1 (自动)
2. **虚拟图像**: 创建黑色images.npy满足EVREAL要求
3. **metadata修复**: 添加sensor_resolution字段

---

## 📦 输出结构

```
output_dir/
├── evreal_e2vid/          # 199张PNG
├── evreal_e2vid_plus/     # 199张PNG
├── evreal_firenet/        # 199张PNG
├── evreal_firenet_plus/   # 199张PNG
├── evreal_ssl-e2vid/      # 199张PNG
└── evreal_*/              # 其他方法
```

---

## 📚 详细文档

查看 `SUCCESSFUL_WORKFLOW.md` 获取完整技术细节
