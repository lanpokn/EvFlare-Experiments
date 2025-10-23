#!/usr/bin/env python3
"""
Checkpoint功能测试脚本

验证断点续存机制的正确性：
1. 模拟中断场景
2. 测试自动恢复
3. 验证结果一致性
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Set

# 直接复制checkpoint函数，避免导入完整模块（避免PyTorch依赖问题）

def load_checkpoint(checkpoint_file: str, verbose: bool = True) -> Tuple[pd.DataFrame, Set]:
    """Load checkpoint data if exists."""
    checkpoint_path = Path(checkpoint_file)

    if not checkpoint_path.exists():
        if verbose:
            print("No checkpoint found, starting fresh evaluation")
        return None, set()

    try:
        existing_df = pd.read_csv(checkpoint_path)
        # Filter out AVERAGE row if present
        existing_df = existing_df[existing_df['sample_id'] != 'AVERAGE']
        completed_ids = set(existing_df['sample_id'].tolist())

        if verbose:
            print(f"✓ Checkpoint loaded: {len(completed_ids)} samples already completed")

        return existing_df, completed_ids
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to load checkpoint: {e}")
        return None, set()


def save_incremental_result(result_row: Dict, checkpoint_file: str, is_first: bool = False):
    """Save single result row incrementally to CSV."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame([result_row])

    # Write with or without header
    mode = 'w' if is_first else 'a'
    df.to_csv(checkpoint_path, mode=mode, header=is_first, index=False)


def test_checkpoint_basic():
    """测试基础checkpoint功能"""
    print("="*60)
    print("测试1: 基础checkpoint读写")
    print("="*60)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, "test_checkpoint.csv")

        # 测试首次写入
        print("✓ 写入第1个样本...")
        result1 = {
            'sample_id': 'sample_001',
            'method1_metric1': 1.23,
            'method1_metric2': 4.56
        }
        save_incremental_result(result1, checkpoint_file, is_first=True)

        # 测试追加写入
        print("✓ 追加第2个样本...")
        result2 = {
            'sample_id': 'sample_002',
            'method1_metric1': 2.34,
            'method1_metric2': 5.67
        }
        save_incremental_result(result2, checkpoint_file, is_first=False)

        # 测试加载checkpoint
        print("✓ 加载checkpoint...")
        df, completed_ids = load_checkpoint(checkpoint_file, verbose=False)

        # 验证结果
        assert df is not None, "❌ DataFrame为空"
        assert len(df) == 2, f"❌ 期望2行，实际{len(df)}行"
        assert completed_ids == {'sample_001', 'sample_002'}, f"❌ 完成ID不匹配: {completed_ids}"

        print(f"✅ 测试通过: 加载了{len(completed_ids)}个已完成样本")
        print()


def test_checkpoint_resume():
    """测试断点恢复场景"""
    print("="*60)
    print("测试2: 模拟中断恢复")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, "test_resume.csv")

        # 模拟首次运行（处理3个样本后中断）
        print("阶段1: 模拟处理前3个样本...")
        samples = [
            {'sample_id': f'sample_{i:03d}', 'metric': float(i)}
            for i in range(1, 11)
        ]

        # 只处理前3个
        for i, sample in enumerate(samples[:3]):
            save_incremental_result(sample, checkpoint_file, is_first=(i==0))
        print(f"✓ 已完成3个样本")

        # 模拟恢复运行
        print("\n阶段2: 恢复运行...")
        df, completed_ids = load_checkpoint(checkpoint_file, verbose=False)
        print(f"✓ 检测到{len(completed_ids)}个已完成样本")

        # 过滤出剩余样本
        remaining = [s for s in samples if s['sample_id'] not in completed_ids]
        print(f"✓ 剩余{len(remaining)}个样本待处理")

        # 继续处理剩余样本
        for sample in remaining:
            save_incremental_result(sample, checkpoint_file, is_first=False)

        # 验证最终结果
        final_df, final_ids = load_checkpoint(checkpoint_file, verbose=False)
        assert len(final_ids) == 10, f"❌ 期望10个样本，实际{len(final_ids)}个"
        assert final_ids == {f'sample_{i:03d}' for i in range(1, 11)}, "❌ 样本ID不完整"

        print(f"✅ 测试通过: 成功恢复并完成全部{len(final_ids)}个样本")
        print()


def test_checkpoint_average_filter():
    """测试AVERAGE行过滤"""
    print("="*60)
    print("测试3: AVERAGE行过滤")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, "test_average.csv")

        # 写入普通样本
        samples = [
            {'sample_id': 'sample_001', 'metric': 1.0},
            {'sample_id': 'sample_002', 'metric': 2.0},
            {'sample_id': 'AVERAGE', 'metric': 1.5},  # 模拟之前的AVERAGE行
        ]

        for i, sample in enumerate(samples):
            save_incremental_result(sample, checkpoint_file, is_first=(i==0))

        # 加载时应自动过滤AVERAGE
        df, completed_ids = load_checkpoint(checkpoint_file, verbose=False)

        assert 'AVERAGE' not in completed_ids, "❌ AVERAGE行未被过滤"
        assert len(completed_ids) == 2, f"❌ 期望2个样本，实际{len(completed_ids)}个"

        print(f"✅ 测试通过: AVERAGE行已正确过滤")
        print()


def test_checkpoint_nonexistent():
    """测试不存在的checkpoint文件"""
    print("="*60)
    print("测试4: 不存在的checkpoint文件")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = os.path.join(tmpdir, "nonexistent.csv")

        # 加载不存在的checkpoint
        df, completed_ids = load_checkpoint(checkpoint_file, verbose=False)

        assert df is None, "❌ DataFrame应为None"
        assert len(completed_ids) == 0, "❌ 完成ID应为空"

        print(f"✅ 测试通过: 正确处理不存在的文件")
        print()


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("CHECKPOINT功能测试套件")
    print("="*60 + "\n")

    try:
        test_checkpoint_basic()
        test_checkpoint_resume()
        test_checkpoint_average_filter()
        test_checkpoint_nonexistent()

        print("="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n断点续存功能已验证正常工作:")
        print("  ✓ 增量写入CSV")
        print("  ✓ 自动加载已完成样本")
        print("  ✓ 智能跳过重复计算")
        print("  ✓ AVERAGE行自动过滤")
        print("  ✓ 优雅处理不存在的checkpoint")
        print("\n可以放心在长时间评估任务中使用！\n")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
