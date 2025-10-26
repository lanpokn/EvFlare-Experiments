#!/usr/bin/env python3
"""
事件相机图像重建 - 使用示例

这是一个简单的示例，展示如何使用本模块从H5文件重建图像。
"""

import sys
from pathlib import Path

# 添加路径以便导入模块
sys.path.append('.')

from h5_to_images import AllH5Processor


def example_1_basic_usage():
    """示例1: 基础用法 - 处理单个数据集的所有H5文件"""
    print("=" * 60)
    print("示例1: 基础用法")
    print("=" * 60)

    # 创建处理器
    dataset_name = "lego2"
    processor = AllH5Processor(dataset_name)

    # 处理所有H5文件
    processor.process_all_h5_files()

    print("\n✅ 示例1完成!")


def example_2_single_h5_file():
    """示例2: 处理单个H5文件"""
    print("\n" + "=" * 60)
    print("示例2: 处理单个H5文件")
    print("=" * 60)

    # 创建处理器
    dataset_name = "lego2"
    processor = AllH5Processor(dataset_name)

    # 获取第一个H5文件
    h5_files = processor.get_all_h5_files()
    if h5_files:
        first_h5 = h5_files[0]
        print(f"处理文件: {first_h5.name}")

        # 处理单个文件
        result = processor.process_single_h5_file(first_h5)

        if result.get("success"):
            print(f"\n✅ 重建成功!")
            print(f"  成功方法: {result.get('methods', [])}")
            print(f"  输出目录: {result.get('reconstruction_dir')}")
        else:
            print(f"\n❌ 重建失败: {result.get('error')}")
    else:
        print("未找到H5文件")


def example_3_custom_output():
    """示例3: 自定义输出路径"""
    print("\n" + "=" * 60)
    print("示例3: 自定义输出路径（代码示例）")
    print("=" * 60)

    print("""
# 如果你想自定义输出路径，可以这样修改：

from h5_to_images import AllH5Processor

# 创建处理器
processor = AllH5Processor("lego2")

# 修改输出路径（在process_single_h5_file中）
# 编辑 h5_to_images.py:209 行的 config.reconstruction_dir
# 例如：
# config.reconstruction_dir = Path("custom_output") / f"reconstruction_{suffix}"

processor.process_all_h5_files()
    """)


def main():
    """主函数"""
    print("🎯 事件相机图像重建 - 使用示例")
    print("=" * 60)
    print("请选择运行的示例:")
    print("1. 基础用法 - 处理数据集的所有H5文件")
    print("2. 处理单个H5文件")
    print("3. 自定义输出路径（代码示例）")
    print("=" * 60)

    # 如果有命令行参数，直接运行对应示例
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("请输入选项 (1/2/3): ").strip()

    if choice == "1":
        example_1_basic_usage()
    elif choice == "2":
        example_2_single_h5_file()
    elif choice == "3":
        example_3_custom_output()
    else:
        print("❌ 无效选项")
        return

    print("\n" + "=" * 60)
    print("📚 更多信息请参考 README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
