# 为data/code/single_model_render.py作为子线程
# python data/code/single_model_render.py --env.target_path model_path
# 命令行输入模型路径，在'/mnt/hdd/zhengquan/Shapenet/rendered_shapenet_sub_process_version'下渲染对应的模型图像40张
# 渲染失败时，返回错误，并删除已经建立的文件夹

import sys
import tyro
import torch
import mathutils
import os
import traceback
import shutil
sys.path.append("/home/zhengquan/04-fep-nbv")

from config import *
from fep_nbv.utils import *
from fep_nbv.env.shapenet_scene import ShapeNetScene
from fep_nbv.env.utils import *

import warnings
warnings.filterwarnings("ignore")

def render_model(cfg,model_path=None):
    try:
        offset = model_path.split('/')[-2]
        category = offset2word(offset)

        obj_file_path = model_path + '/models/model_normalized.obj'
        cfg.env.target_path = obj_file_path
        scene = eval(cfg.env.scene)(cfg=cfg.env)

        output_path = model_path.replace('/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2',
                                         '/mnt/hdd/zhengquan/Shapenet/rendered_shapenet_sub_process_version')
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
        print(f"渲染失败: {e} 删除已经生成的文件")
        shutil.rmtree(output_path)
        sys.exit(1)
        # traceback.print_exc()
        # raise RuntimeError("渲染函数内部异常")

if __name__=='__main__':
    model = sys.argv[-1]
    print(model)
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = 'ShapeNetScene'

    render_model(cfg,model)