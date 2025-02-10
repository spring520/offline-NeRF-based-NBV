import tyro
import sys
import mathutils
from tqdm import tqdm
sys.path.append("/home/zhengquan/04-fep-nbv")
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端

from config import *
from fep_nbv.utils import *
from fep_nbv.env.shapenet_scene import ShapeNetScene
from fep_nbv.env.utils import *

import warnings
warnings.filterwarnings("ignore")

import psutil, os
process = psutil.Process(os.getpid())



def render_shapenet_(cfg,model_path=None):
    offset = model_path.split('/')[-2]
    category = offset2word(offset)

    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path 
    scene = eval(cfg.env.scene)(cfg=cfg.env)

    output_path = model_path.replace('/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2','/mnt/hdd/zhengquan/Shapenet/rendered_shapenet')
    output_path = output_path.replace(offset,category)
    os.makedirs(output_path, exist_ok=True)

    # 渲染图像    
    camera_position = mathutils.Vector((1, 1, 1))
    target_position = mathutils.Vector((0, 0, 0))
    direction = target_position - camera_position
    rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y')) # wxyz
    fixed_pose = torch.concat((rot_quat[[1,2,3,0]], torch.tensor(camera_position))) # xyzwxyz
    img = scene.render_pose(fixed_pose) # RGBA
    save_img(img[:,:,:3], output_path+'/scene_test.png')

    # gen data test
    poses = scene.gen_data_fn['full']()
    # print(f'init mode poses: {poses}')
    for i,pose in enumerate(poses):
        # scene = eval(cfg.env.scene)(cfg=cfg.env)
        # print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")
        empty_cache()
        print(f'dealing with pose {i}')
        img = scene.render_pose(pose)
        print_cuda_allocated()
        save_img(img[:,:,:3], f'{output_path}/scene_gendata_init_{i}.jpg')
        

if __name__=='__main__':
    # 渲染shapenet数据集中所有模型的图像
    # 方位角8 俯仰角4 半径为1   
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = 'ShapeNetScene'

    model_dict = shapenet_model_path_dict()
    # for category in model_path.keys():
    #     for model in model_path[category]:
    #         print(model)
    #         render_shapenet_(cfg,model)

    # 文件路径用于保存进度
    progress_file = "/mnt/hdd/zhengquan/Shapenet/rendered_shapenet/progress.json"

    # 加载已处理的模型记录
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            processed_data = json.load(file)  # 格式：{"category1": ["model1", "model2"], ...}
    else:
        processed_data = {}

    # 假设 `model_dict` 是模型字典，其中 key 是类别，value 是模型列表
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
            else:
                print(f'{category}:{model}/n/n')

                empty_cache()
                render_shapenet_(cfg,model)

                # 记录成功处理的模型
                processed_data[category].append(model)

                # 保存进度
                with open(progress_file, "w") as file:
                    json.dump(processed_data, file)
                
            # try:
            #     # Your model processing code here
                
            # except Exception as e:
            #     print(f"Error processing model {model} in category {category}: {e}")
            #     break  # 遇到错误停止运行