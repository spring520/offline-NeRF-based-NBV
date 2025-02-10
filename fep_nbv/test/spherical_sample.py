import sys
import healpy as hp
import numpy as np
import tyro
import torch
import mathutils
sys.path.append("/home/zhengquan/04-fep-nbv")

from config import *
from fep_nbv.utils.utils import *
from fep_nbv.env.utils import *
from fep_nbv.env.shapenet_env import set_env
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints,generate_fibonacci_viewpoints,generate_polar_viewpoints



if __name__=='__main__':
    # 生成一套视角，然后可视化这些视角
    # finabocci方式生成一组视角
    poses = generate_HEALPix_viewpoints(n_side=4,original_viewpoint=np.array([1,0,0]))
    print(poses[0])
    print(poses.shape)
    poses = generate_polar_viewpoints()
    print(poses[0])
    print(poses.shape)
    poses = generate_fibonacci_viewpoints(num_points=100)
    print(poses[0])
    print(poses.shape)
    

    # 环境闯将
    test_model_path = random_shapenet_model_path()
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    obj_file_path = test_model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path
    env = set_env(cfg)

    # 摄像机视角
    # 计算相机位置，目标位置和视角方向的四元数
    camera_position = mathutils.Vector(cfg.camera_aabb[1])
    target_position = mathutils.Vector(cfg.camera_aabb[0])
    print(f'camera position:{camera_position}')
    print(f'camera position:{target_position}')
    direction = target_position - camera_position 
    rot_quat = torch.tensor(direction.to_track_quat('-Z', 'Y'))
    # 渲染固定斜上方视角下的目标图像并保存
    fixed_pose = torch.concat((rot_quat[[1,2,3,0]], torch.tensor(camera_position*3)))

    gif = GIFSaver()
    for i,pose in enumerate(poses):
        pose = torch.tensor(pose)
        env.scene.add_cone_at_pose(pose[-3:],pose[[3,0,1,2]],id=i)
        image = env.scene.render_pose(fixed_pose)
        gif.add(np.asarray(image))
    save_img(image, '/home/zhengquan/04-fep-nbv/data/test/visualizaton/polar_viewpoints.png')
    gif.save('/home/zhengquan/04-fep-nbv/data/test/visualizaton/polar_viewpoints.gif')
