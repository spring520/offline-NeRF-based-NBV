# 这个代码是为了画一些图，说明我的问题
import sys
import tyro
import os
import torch
import mathutils
import numpy as np
sys.path.append("/home/zhengquan/04-fep-nbv")

from config import *
from fep_nbv.env.utils import offset2word,save_img,empty_cache,GIFSaver
from fep_nbv.env.shapenet_scene import *
from fep_nbv.utils.transform_viewpoints import xyz2pose
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints
# from fep_nbv.utils.utils import 

if __name__=='__main__':
    # 渲染一辆车的图像，车头分别指向x轴和y轴
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = 'ShapeNetScene'
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02958343/bf7efe509caf42cb7481cee66aa2b2f4'

    offset = model_path.split('/')[-2]
    category = offset2word(offset)

    obj_file_path = model_path + '/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path
    scene = eval(cfg.env.scene)(cfg=cfg.env)
    # scene.set_white_background()
    scene.add_light()
    scene.add_axes()
    # scene.rotate_objects_z()

    output_path = '/home/zhengquan/04-fep-nbv/data/test/visualizaton'
    os.makedirs(output_path, exist_ok=True)

    # 渲染固定视角图像
    camera_position = mathutils.Vector((1, 1, 1))*3
    target_position = mathutils.Vector((0, 0, 0))
    direction = target_position - camera_position
    rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y'))  # wxyz
    fixed_pose = torch.concat((rot_quat[[1, 2, 3, 0]], torch.tensor(camera_position)))  # xyzwxyz
    
    poses = generate_HEALPix_viewpoints(n_side=1,original_viewpoint=np.array([1,0,0]))
    # poses = generate_HEALPix_viewpoints(n_side=1,original_viewpoint=np.array(poses[1,4:]))
    gif = GIFSaver()
    for i,pose in enumerate(poses):
        pose = torch.tensor(pose)
        scene.add_cone_at_pose2(pose[-3:],pose[[3,0,1,2]],id=i)
        image = scene.render_pose(fixed_pose)
        gif.add(np.asarray(image))
    # gif.save(os.path.join(output_path,'distribution_3.gif'))
    save_img(image[:, :, :3], output_path + '/distribution_1_final.png')

    # point = np.array([[0,2,0],[1,0,0]])
    # pose = xyz2pose(point[:,0],point[:,1],point[:,2])
    # pose = pose[0]
    # # pose = torch.tensor(pose)
    # scene.add_cone_at_pose(pose[4:],pose[[3,0,1,2]])
    # img = scene.render_pose(fixed_pose)  # RGBA
    # save_img(img[:, :, :3], output_path + '/origin_with_coordinate.png')


