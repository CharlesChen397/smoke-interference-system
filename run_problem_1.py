#!/usr/bin/env python3
"""
运行问题1的主脚本
"""

if __name__ == "__main__":
    try:
        from problem_1 import solve_problem_1, visualize_problem_1

        print("开始求解烟幕干扰弹投放策略问题1...")
        print("=" * 50)

        # 求解问题1
        result = solve_problem_1()

        print("=" * 50)
        print("求解完成！")

    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有必需的文件都在同一目录下")
    except Exception as e:
        print(f"运行错误: {e}")
