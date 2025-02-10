import tyro
import sys
import mathutils
from tqdm import tqdm
sys.path.append("/home/zhengquan/04-fep-nbv")
import matplotlib
import multiprocessing
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

def render_shapenet_with_retry(cfg,model_path=None):
    try:
        offset = model_path.split('/')[-2]
        category = offset2word(offset)

        obj_file_path = model_path + '/models/model_normalized.obj'
        cfg.env.target_path = obj_file_path
        scene = eval(cfg.env.scene)(cfg=cfg.env)

        output_path = model_path.replace('/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2',
                                         '/mnt/hdd/zhengquan/Shapenet/rendered_shapenet')
        output_path = output_path.replace(offset, category)
        os.makedirs(output_path, exist_ok=True)

        # 渲染固定视角图像
        camera_position = mathutils.Vector((1, 1, 1))
        target_position = mathutils.Vector((0, 0, 0))
        direction = target_position - camera_position
        rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y'))  # wxyz
        fixed_pose = torch.concat((rot_quat[[1, 2, 3, 0]], torch.tensor(camera_position)))  # xyzwxyz
        img = scene.render_pose(fixed_pose)  # RGBA
        save_img(img[:, :, :3], output_path + '/scene_test.png')

        # 渲染多个视角图像
        poses = scene.gen_data_fn['full']()
        for i, pose in enumerate(poses):
            empty_cache()  # 清理显存
            print(f'处理视角 {i}')
            img = scene.render_pose(pose)
            save_img(img[:, :, :3], f'{output_path}/scene_gendata_init_{i}.jpg')

    except Exception as e:
        # 捕获异常并打印堆栈
        print(f"渲染失败: {e}")
        traceback.print_exc()
        raise RuntimeError("渲染函数内部异常")
        

if __name__=='__main__':
    multiprocessing.set_start_method("spawn", force=True)

    
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = 'ShapeNetScene'

    # 所有模型的路径，一个dict
    model_dict = shapenet_model_path_dict()

    # 文件路径用于保存进度，加载已处理的模型记录
    progress_file = "/mnt/hdd/zhengquan/Shapenet/rendered_shapenet/progress.json"
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
            # print(f'\n\n\n\ndealing with {model}')
            if model in processed_data[category]:
                pass  # 跳过已处理的模型
            
            failure_count = 0
            print(f'\ndealing with {category}:{model}\n\n')
            while failure_count<4:
                process = multiprocessing.Process(target=render_shapenet_with_retry, args=(cfg, model))
                process.start()
                process.join()

                if process.exitcode == 0:
                    print(f"模型 {model} 渲染成功")
                    break
                else:
                    failure_count += 1
                    print(f"模型 {model} 渲染失败 (第 {failure_count} 次)")

            # 如果失败次数超过限制，可以删除文件或记录日志
            if failure_count<4:
                pass
            else:
                print(f"模型 {model} 渲染失败超过 {m4} 次，正在删除...")
                try:
                    shutil.rmtree(model)
                    print(f"已删除模型目录: {model}")
                except Exception as e:
                    print(f"删除模型目录失败: {e}")
