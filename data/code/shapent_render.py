# 遍历所有模型，在progress.json中已经记录处理过的就直接跳过
# 尝试渲染，渲染失败超过四次，直接把原始模型从shapeet里删了
# 本质上是要实现一个垃圾模型删除的功能

import tyro
import sys
import mathutils
from tqdm import tqdm
sys.path.append("/home/zhengquan/04-fep-nbv")
import matplotlib
import subprocess
import traceback
import time
import shutil
matplotlib.use('Agg')  # 强制使用非交互式后端

from config import *
from fep_nbv.utils import *
from fep_nbv.env.shapenet_scene import ShapeNetScene
from fep_nbv.env.utils import *

import warnings
warnings.filterwarnings("ignore")

import psutil, os
process = psutil.Process(os.getpid())

def render_subprocess(model):
    command =[
        "python",
        "data/code/single_model_render.py",
        "--env.target_path",
        model
    ]

    failure_count = 0
    max_failure = 4
    print(f'\ndealing with {model}\n\n')
    while failure_count<max_failure:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"渲染成功: {model}")
            break
        else:
            failure_count+=1
            print(f"渲染失败第 {failure_count} 次")
    if failure_count==max_failure:
        print(f"模型 {model} 渲染失败超过 {max_failure} 次，正在删除...")
        try:
            shutil.rmtree(model)
            print(f"已删除模型目录: {model}")
        except Exception as e:
            print(f"删除模型目录失败: {e}")

if __name__=='__main__':
    # 所有模型的路径，一个dict
    model_dict = shapenet_model_path_dict()

    # 文件路径用于保存进度，加载已处理的模型记录
    progress_file = "/mnt/hdd/zhengquan/Shapenet/rendered_shapenet_sub_process_version/progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            processed_data = json.load(file)  # 格式：{"category1": ["model1", "model2"], ...}
    else:
        processed_data = {}

    for category, models in tqdm(model_dict.items(), desc="Processing Categories", unit="category"):
        # 跳过已完成的类别
        if category in processed_data and len(processed_data[category]) == len(models):
            continue

        # 确保当前类别的进度记录存在
        if category not in processed_data:
            processed_data[category] = []

        for model in tqdm(models, desc=f"Processing Models in {category}", unit="model", leave=False):
            if model in processed_data[category]:
                print(f'{model} already rendered, so jumped')
                continue  # 跳过已处理的模型
                
            render_subprocess(model)

            # 记录成功处理的模型
            processed_data[category].append(model)

            # 保存进度
            with open(progress_file, "w") as file:
                json.dump(processed_data, file)