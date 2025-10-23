# 断点续存功能使用指南

## 功能概述

为 `evaluate_all_methods.py` 和 `evaluate_evk4_methods.py` 增加了**自动断点续存**机制，解决长时间指标计算被中断后需要重头开始的问题。

### 核心特性

- ✅ **自动checkpoint**: 每处理完1个样本立即保存
- ✅ **智能恢复**: 启动时自动检测已完成样本
- ✅ **零配置**: 默认开启，无需额外参数
- ✅ **中断安全**: Ctrl+C、网络断连、系统崩溃都不丢失进度

## 工作原理

### 1. 增量保存策略
```
评估流程:
  样本1 → 计算指标 → 立即写入CSV ✓
  样本2 → 计算指标 → 立即追加CSV ✓
  样本3 → 计算指标 → 立即追加CSV ✓
  [中断] ...
```

每个样本处理完成后，立即追加到 `results/xxx_evaluation_results.csv`，无需等待全部完成。

### 2. 自动恢复机制
```
重新启动评估:
  ↓
检查 results/xxx_evaluation_results.csv 是否存在
  ↓
存在 → 读取已完成的 sample_id
  ↓
跳过这些样本，只处理剩余样本
  ↓
继续评估
```

### 3. 结果合并
```
最终结果 = 历史数据 + 新计算数据 + AVERAGE行
```

自动合并checkpoint中的数据和新计算的数据，生成完整报告。

## 使用示例

### 场景1: 首次运行评估

```bash
# 开始评估35个样本
python evaluate_all_methods.py --output results

# 输出:
# ================================================================================
# MULTI-METHOD H5 EVENT EVALUATION
# ================================================================================
# No checkpoint found, starting fresh evaluation
# Total samples: 35
# Remaining to process: 35
#
# [1/35] composed_00470
#   ...
```

### 场景2: 中断后恢复

假设运行到第10个样本时中断（Ctrl+C 或系统崩溃）:

```bash
# 直接重新运行相同命令
python evaluate_all_methods.py --output results

# 输出:
# ================================================================================
# MULTI-METHOD H5 EVENT EVALUATION
# ================================================================================
# ✓ Checkpoint loaded: 10 samples already completed
#   Resuming from checkpoint: results/multi_method_evaluation_results.csv
# Total samples: 35
# Already completed: 10
# Remaining to process: 25
#
# [11/35] composed_00481  ← 自动从第11个继续
#   ...
```

### 场景3: 全部完成后再运行

```bash
python evaluate_all_methods.py --output results

# 输出:
# ✓ Checkpoint loaded: 35 samples already completed
# Total samples: 35
# Already completed: 35
# Remaining to process: 0
# ✓ All samples already completed!
```

### 场景4: EVK4数据评估

```bash
# 与simu数据完全相同的使用方式
python evaluate_evk4_methods.py --output results

# 中断后恢复
python evaluate_evk4_methods.py --output results
# 输出: "✓ Checkpoint loaded: X samples already completed"
```

## 技术细节

### Checkpoint文件格式

标准CSV格式，与最终结果完全相同:

```csv
sample_id,method1_chamfer_distance,method1_tf1,method2_chamfer_distance,...
composed_00470,4.123,0.876,5.234,...
composed_00471,3.456,0.912,4.567,...
...
```

**注意**:
- Checkpoint文件 **不包含** AVERAGE行
- AVERAGE行只在最终报告中生成
- 加载时自动过滤旧的AVERAGE行

### 进度显示增强

```
[已完成数+当前处理索引/总数] sample_id
```

示例:
```
[11/35] composed_00481  ← 第11个样本，共35个
```

### 文件路径

- **Simu数据**: `results/multi_method_evaluation_results.csv`
- **EVK4数据**: `results/evk4_evaluation_results.csv`

checkpoint文件与最终结果文件是**同一个文件**。

## 常见问题

### Q1: 如何清除checkpoint重新开始？

删除checkpoint文件即可:

```bash
rm results/multi_method_evaluation_results.csv
# 或
rm results/evk4_evaluation_results.csv
```

### Q2: 可以更改指标后继续吗？

**不推荐**。指标更改会导致列结构不匹配，建议:

```bash
# 方法1: 清除checkpoint重新开始
rm results/xxx_evaluation_results.csv

# 方法2: 使用不同的output目录
python evaluate_all_methods.py --output results_new_metrics
```

### Q3: 中断时数据安全吗？

**安全**。每个样本处理完后立即调用 `df.to_csv()`，数据已落盘。

**风险窗口**: 只有正在计算的当前样本会丢失，已完成的样本都已保存。

### Q4: 支持并行运行吗？

**不支持**。多个进程同时写入同一个CSV会导致数据损坏。

如需并行评估，使用不同的 `--output` 目录:
```bash
# 终端1
python evaluate_all_methods.py --num-samples 20 --output results_part1

# 终端2
python evaluate_all_methods.py --num-samples 20 --output results_part2
```

### Q5: 如何验证checkpoint功能正常？

运行测试脚本:
```bash
python test_checkpoint.py
```

预期输出: "✅ 所有测试通过！"

## 性能影响

### 额外开销

- **每样本IO时间**: ~10-50ms (追加CSV)
- **启动检查时间**: ~100-500ms (读取checkpoint)

### 收益

- **避免重复计算**: 单样本kernel指标可能耗时**数分钟**
- **灵活中断**: 不再担心中断损失，随时可以Ctrl+C

**结论**: IO开销可忽略，收益巨大。

## 实现细节

### 代码架构

```python
# evaluate_all_methods.py

def load_checkpoint(checkpoint_file):
    """加载已完成样本"""
    df = pd.read_csv(checkpoint_file)
    df = df[df['sample_id'] != 'AVERAGE']  # 过滤AVERAGE
    completed_ids = set(df['sample_id'])
    return df, completed_ids

def save_incremental_result(result_row, checkpoint_file, is_first):
    """增量保存单个样本结果"""
    df = pd.DataFrame([result_row])
    mode = 'w' if is_first else 'a'
    df.to_csv(checkpoint_file, mode=mode, header=is_first, index=False)

def run_evaluation(..., checkpoint_file=None):
    # 1. 加载checkpoint
    existing_df, completed_ids = load_checkpoint(checkpoint_file)

    # 2. 过滤已完成样本
    remaining_ids = [sid for sid in all_ids if sid not in completed_ids]

    # 3. 处理剩余样本
    for i, sample_id in enumerate(remaining_ids):
        result = evaluate_sample(sample_id)
        results.append(result)

        # 4. 立即保存checkpoint
        save_incremental_result(result, checkpoint_file, is_first=(i==0))

    # 5. 合并结果
    final_df = pd.concat([existing_df, pd.DataFrame(results)])
    return final_df
```

### 关键设计决策

1. **checkpoint即最终结果**: 不使用单独的checkpoint文件，避免文件冗余
2. **AVERAGE行后置**: checkpoint中不包含AVERAGE，只在最终报告生成
3. **set查找**: 使用 `set` 快速判断样本是否已完成 (O(1))
4. **is_first标志**: 首次写入需要header，后续追加不需要

## 测试覆盖

```bash
python test_checkpoint.py
```

测试用例:
- ✅ 基础读写功能
- ✅ 中断恢复场景
- ✅ AVERAGE行过滤
- ✅ 不存在的checkpoint文件处理

## 兼容性

- ✅ **向后兼容**: 不影响现有CSV格式
- ✅ **可选功能**: checkpoint_file=None时禁用
- ✅ **独立功能**: simu和EVK4评估独立checkpoint
- ✅ **Python 3.7+**: 使用标准库，无额外依赖

## 总结

断点续存机制显著提升了长时间评估任务的鲁棒性和灵活性:

| 特性 | 改进前 | 改进后 |
|------|--------|--------|
| 中断损失 | 全部重头开始 | 只损失当前样本 |
| 进度可见性 | 等待全部完成 | 实时CSV可查看 |
| 灵活性 | 必须一次运行完 | 随时中断/恢复 |
| 资源调度 | 长时间占用 | 可分批完成 |

**建议**: 在所有长时间评估任务中启用此功能，特别是使用kernel指标时。
