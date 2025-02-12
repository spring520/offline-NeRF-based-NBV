import os
import json
import random
from pathlib import Path
from typing import List, Dict
root_path = os.getenv('nbv_root_path', '/default/path')
shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')


def generate_model_status_json(shapenet_dir=root_path, percentage=0.3, output_file='data/test/model_status.json',num_viewpoints=48,num_rotations=8):
    """
    生成 ShapeNet 模型的状态文件，同时打印每个模型的统计信息。

    :param shapenet_dir: ShapeNet 数据集根目录
    :param percentage: 选择每个类别的模型的比例
    :param output_file: 输出 JSON 文件路径
    """
    model_status = {}
    total_model_number = 0

    for category in os.listdir(shapenet_dir):
        category_path = os.path.join(shapenet_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 获取所有模型列表
        models = [model for model in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, model))]

        # 按比例选择模型
        num_selected_models = max(1, int(len(models) * percentage))
        selected_models = random.sample(models, num_selected_models)

        # 初始化每个模型的 viewpoint 和 rotation 状态
        for model in selected_models:
            model_key = os.path.join(category, model)
            model_status[model_key] = {
                "finished": [[False for _ in range(num_rotations)] for _ in range(num_viewpoints)]
            }

            # 打印模型统计信息
        print(f"类别: {category}, 模型数量: {len(selected_models)}, 总模型数量: {len(models)}, 比例: {len(selected_models)/len(models):2f}")
        total_model_number += len(selected_models)
    
    print(f'\n总模型数量: {total_model_number}')
    # 将结果保存到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(model_status, f, indent=4)

    print(f"\nJSON 文件已保存到: {output_file}")


def read_model_status(json_file):
    """
    读取 ShapeNet 模型状态 JSON 文件，并打印模型的统计信息。

    :param json_file: JSON 文件路径
    """
    # 检查 JSON 文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: JSON 文件 {json_file} 不存在！")
        return

    # 读取 JSON 文件
    with open(json_file, "r") as f:
        model_status = json.load(f)
    num_finished = 0

    # 遍历并输出每个模型的状态信息
    for model_key, status in model_status.items():
        num_finished += sum(
            sum(rotation_finished for rotation_finished in viewpoint_finished)
            for viewpoint_finished in status["finished"]
        )

        # 输出每个模型的统计信息
        # print(f"模型: {model_key}, 已完成任务: {num_finished}/{total_tasks}")
    print(f'model num: {len(model_status.keys())},tasks:{num_finished}/{48*8*len(model_status.keys())}')

    print("\n读取完成")

# 示例调用
if __name__ == "__main__":
    shapenet_root = shapenet_path  # 修改为你的 ShapeNet 路径
    percentage = 20
    output_file = f'data/test/model_status_{percentage}.json'     # 输出文件路径
    # generate_model_status_json(shapenet_root, percentage=percentage/100,output_file=output_file)

    read_model_status(output_file)