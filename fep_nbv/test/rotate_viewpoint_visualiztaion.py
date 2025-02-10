# 可视化一下视角的旋转过程，左边是视角，右边是三维球面上的视角分布旋转
import sys
from PIL import Image
import torch
import mathutils
import tyro
import numpy as np
sys.path.append('/home/zhengquan/04-fep-nbv')
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints
from fep_nbv.env.shapenet_env import set_env
from config import *
from fep_nbv.env.utils import *


import warnings
warnings.filterwarnings("ignore")

if __name__=='__main__':
    gif = GIFSaver()
    index = 5
    

    # 先产生所有的左边的视角图
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133'
    candidate_viewpoint_poses = generate_HEALPix_viewpoints(n_side=1)
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path
    env = set_env(cfg)

    left_figures = []
    for offset_phi_index,offset_phi in enumerate(np.arange(0, 2, 0.25)*np.pi):
        absolute_viewpoint_poses = generate_HEALPix_viewpoints(n_side=1,original_viewpoint=candidate_viewpoint_poses[index,4:],offset_phi=offset_phi)
        image = env.scene.render_pose(absolute_viewpoint_poses[0]) 
        left_figures.append(image)

    del env

    # 第一张图只有一个视角分布
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133'
    absolute_viewpoint_poses = generate_HEALPix_viewpoints(n_side=1,original_viewpoint=candidate_viewpoint_poses[index,4:])
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path
    env2 = set_env(cfg)

    # 摄像机视角
    # 计算相机位置，目标位置和视角方向的四元数
    camera_position = mathutils.Vector(cfg.camera_aabb[1])
    target_position = mathutils.Vector(cfg.camera_aabb[0])
    print(f'camera position:{camera_position}')
    print(f'camera position:{target_position}')
    direction = target_position - camera_position 
    rot_quat = torch.tensor(direction.to_track_quat('-Z', 'Y'))
    # 渲染固定斜上方视角下的目标图像并保存
    fixed_pose = torch.concat((rot_quat[[1,2,3,0]], torch.tensor(camera_position*2)))

    for i,pose in enumerate(absolute_viewpoint_poses):
        pose = torch.tensor(pose)
        env2.scene.add_cone_at_pose2(pose[-3:],pose[[3,0,1,2]],id=i,color=(1,1,1,0.2))
    image2 = env2.scene.render_pose(fixed_pose) # 固定视角
    image1 = left_figures[0]
    width, height = Image.fromarray(image2).size

    canvas = Image.new("RGB", (2 * width, height))
    canvas.paste(Image.fromarray(image2), (width, 0))  # 右上角
    canvas.paste(Image.fromarray(image1), (0, 0))  # 左上角
    gif.add(np.array(canvas))
    del env2
        
    for offset_phi_index,offset_phi in enumerate(np.arange(0.25, 2, 0.25)*np.pi):
        absolute_viewpoint_poses = generate_HEALPix_viewpoints(n_side=1,original_viewpoint=candidate_viewpoint_poses[index,4:],offset_phi=offset_phi)
        env2 = set_env(cfg)
        image1 = left_figures[offset_phi_index+1]
        # for i,pose in enumerate(candidate_viewpoint_poses):
        #     pose = torch.tensor(pose)
        #     env2.scene.add_cone_at_pose2(pose[-3:],pose[[3,0,1,2]],id=i,color=(1,1,1,0.2))
        for i,pose in enumerate(absolute_viewpoint_poses):
            if i==1:
                color=(1,0,0,0.6)
            else:
                color = (1,1,1,0.2)
            pose = torch.tensor(pose)
            env2.scene.add_cone_at_pose2(pose[-3:],pose[[3,0,1,2]],id=i,color=color)
        image2 = env2.scene.render_pose(fixed_pose) # 固定视角
        canvas = Image.new("RGB", (2 * width, height))
        canvas.paste(Image.fromarray(image2), (width, 0))  # 右上角
        canvas.paste(Image.fromarray(image1), (0, 0))  # 左上角
        gif.add(np.array(canvas))
        del env2
    gif.save('/home/zhengquan/04-fep-nbv/data/test/test.gif')