import gym
import numpy as np
import os
import tyro
import sys
import json
import mathutils
sys.path.append("/attached/data/remote-home2/zzq/04-fep-nbv")

from config import *
from fep_nbv.env.shapenet_scene import ShapeNetScene
from fep_nbv.env.utils import *
from fep_nbv.utils import *

def set_env(cfg):
    if cfg.scene.name == 'hubble':
        # breakpoint()
        #aabb = ([[-0.92220873, -1.00288355, -1.03578806],
    #    [ 0.92220724,  1.05716348,  1.75192416]])
        cfg.object_aabb = torch.tensor([[-1, -1.1, -1.1], [1, 1.1, 1.8]])#*1.1
        factor = 2.5
        cfg.target_aabb = cfg.object_aabb*factor
        cfg.camera_aabb = cfg.object_aabb*factor
        cfg.env.scale = 0.3333 * 0.5
        cfg.density_threshold = 1e-3
    elif cfg.scene.name =='shapenet':
        cfg.object_aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]])#*1.1
        factor = 2
        cfg.target_aabb = cfg.object_aabb
        cfg.camera_aabb = cfg.object_aabb*1.5
        # cfg.env.scale = 0.3333 * 0.5
        cfg.density_threshold = 1e-3
    else:
        raise NotImplementedError
    env = ShapeNetEnviroment(cfg.env)
    # breakpoint()
    return env


class ShapeNetEnviroment(gym.Env):
    # init_images = 10
    # horizon = 20
    """Custom Environment that follows gym interface"""
    def __init__(self, cfg,):
        super(ShapeNetEnviroment, self).__init__()

        self.horizon = cfg.horizon
        self.max_steps = cfg.horizon
        
        self.pose_history = []
        self.obs_history = []

        # set env
        cfg.object_aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]])#*1.1
        factor = 2
        cfg.target_aabb = cfg.object_aabb
        cfg.camera_aabb = cfg.object_aabb*factor
        cfg.density_threshold = 1e-3

        self.cfg = cfg

        self.scene = eval(self.cfg.scene)(self.cfg)

        # Initialize state
        self.reset()

    def step(self, action):
        if action.ndim==1:
            action = action.unsqueeze(0)
        position = self.state

        # new_poses = []
        new_images = []
        for pose in action:
            img = self.scene.render_pose(pose)

            # self.scene.set_camera_pose(pose)
            # result = self.scene.render()
            # # img = result['mask']*result['image']
            # img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
            # input(img.shape)
            # new_poses.append(pose)
            new_images.append(img)
        

        
        # self.pipeline.add_image(images=new_images, poses=action)

        self.pose_history += action
        self.obs_history += new_images

        done = False
        if self.steps >= self.max_steps:
            done = True

        reward = 0. # TODO

        self.state = action[-1]
        self.steps += 1


        return new_images, reward, done, {}

    def reset(self):
        # transforms = get_transforms(idx_start=0, idx_end=100)

        # np_images = []
        # for pose in transforms:
        #     self.scene.set_camera_pose(pose)
        #     result = self.scene.render()
        #     img = result['mask']*result['image']
        #     img = rgb_to_rgba(img)
        #     # input(img.shape)
        #     # new_poses.append(pose)
        #     np_images.append(img)
        
        np_images, transforms = self.get_images(mode='init')

        self.pose_history = transforms
        self.obs_history = np_images
        self.state = np.array(self.pose_history[-1])  # Reset position to the center

        # self.pipeline.add_image(images=np_images, poses=transforms)
        
        self.steps = 0
        return np_images  # reward, done, info can't be included
    
    def get_images(self, mode, return_quat=True):
        return self.gen_data(mode, return_quat=return_quat)


        # file = f'data/{self.cfg.scene.name}/{mode}/transforms.json'
        # if not os.path.exists(file) or (hasattr(self.cfg, f'gen_{mode}') and getattr(self.cfg, f'gen_{mode}')):
        #     return self.gen_data(mode, return_quat=return_quat)
        # else:
        #     img, transforms = self.scene.load_data(file=file, return_quat=return_quat)
        #     if mode == 'init':
        #         if len(img) != self.cfg.n_init_views:
        #             return self.gen_data(mode, return_quat=return_quat)
        #     return img, transforms
          
    def gen_data(self, mode, return_quat=False):
        file = f'data/{self.cfg.scene.name}/{mode}/transforms.json'
        print(f'Generating data for {mode}')
        poses = self.scene.gen_data_fn[mode]()

        images = []
        for pose in poses:
            image = self.scene.render_pose(pose)
            images.append(image)
        
        if getattr(self.cfg, f'save_data'):
            print(f'saving data to {file}')
            self.scene.save_data(file, poses, images)
        
        if return_quat:
            poses = [pose2tensor(pose) for pose in poses]
        else:
            poses = [torch.FloatTensor(pose) for pose in poses]
        return images, poses
 


if __name__=='__main__':
    # 环境创建初始化
    # 函数：step, reset, render, close
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet

    target_path = random_shapenet_model_path()
    obj_file_path = target_path+'/models/model_normalized.obj'
    json_path = obj_file_path[:-4]+'.json'
    with open(json_path, 'r') as file:
        data = json.load(file)
        centroid = data.get("centroid", None)
    cfg.env.target_path = obj_file_path # type: ignore

    env = ShapeNetEnviroment(cfg.env)

    # 渲染图像
    camera_position = mathutils.Vector((1, 1, 1))
    target_position = mathutils.Vector((0, 0, 0))
    direction = target_position - camera_position
    rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y')) # wxyz
    fixed_pose = torch.concat((rot_quat[[1,2,3,0]], torch.tensor(camera_position))) # xyzwxyz
    obs,_,_,_ = env.step(fixed_pose)

    print(obs[0].shape)
    print('env test success')


    pass 