#!/bin/bash

# 设置要执行的 Python 文件路径
PYTHON_SCRIPT="data/code/generate_distribution_dataset.py"

# 循环执行 20 次
for i in {1..64}; do
    echo "Iteration $i/64"

    # 运行 Python 文件
    python3 $PYTHON_SCRIPT

    # # 检查是否出错
    # if [ $? -ne 0 ]; then
    #     echo "Error occurred at iteration $i"
    #     exit 1
    # fi

    echo "Iteration $i completed"
    echo ""
done

echo "All iterations completed!"