import gym
import numpy as np
import os
import tyro
import sys
import json
import mathutils
root_path = os.getenv('nbv_root_path', '/default/path')
shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')
distribution_dataset_path = os.getenv('distribution_dataset_path', '/default/distribution/dataset/path')
if not os.path.exists(root_path):
    root_path=root_path.replace('/attached/data','/attached')
    shapenet_path=shapenet_path.replace('/attached/data','/attached')
    distribution_dataset_path=distribution_dataset_path.replace('/attached/data','/attached')
sys.path.append(root_path)

from config import *
from fep_nbv.env.shapenet_scene import ShapeNetScene
from fep_nbv.env.utils import *
from fep_nbv.utils import *
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints

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
        self.action_space_mode = cfg.action_space_mode  
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

        self.initialize_action_space()

    def initialize_action_space(self):
        """初始化不同模式的动作空间"""
        if self.action_space_mode == 'discrete':
            self.discrete_actions = generate_HEALPix_viewpoints(n_side=2)
            self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        elif self.action_space_mode == 'continuous_sphere':
            self.action_space = gym.spaces.Box(
                low=np.array([0, 0]), 
                high=np.array([2*np.pi, np.pi]), 
                dtype=np.float32
            )  # 方位角俯仰角
        elif self.action_space_mode == 'continuous_aabb':
            lower, upper = self.camera_aabb.numpy()
            self.action_space = gym.spaces.Box(
                low=lower, 
                high=upper, 
                dtype=np.float32
            )  # 相机 AABB 内的 3D 位置
        else:
            raise ValueError(f"Unsupported action space mode: {self.action_space_mode}")


    def step(self, action):
        if self.action_space_mode == 'discrete':
            # 动作为索引，选择离散动作列表中的视角
            pose = self.discrete_actions[action]

        elif self.action_space_mode == 'continuous_sphere':
            # 动作是 [方位角, 俯仰角]，转换为 3D 位置 + 旋转四元数
            azimuth, elevation = action[0], action[1]
            radius = 2  # 固定半径
            x = radius * np.sin(elevation) * np.cos(azimuth)
            y = radius * np.cos(elevation)
            z = radius * np.sin(elevation) * np.sin(azimuth)
            position = torch.tensor([x, y, z])
            quat = quaternion_from_euler(azimuth, elevation)
            pose = pose_to_7d(quat, position)

        elif self.action_space_mode == 'continuous_aabb':
            # 动作为3D位置，从AABB内随机采样
            position = sample_pose_from_aabb(self.camera_aabb.numpy())
            target = torch.tensor([0.0, 0.0, 0.0])  # 目标位置，假设是世界中心
            quat = look_at_rotation(position.numpy(), target.numpy())
            pose = pose_to_7d(quat, position)

        else:
            raise ValueError(f"Unsupported action space mode: {self.action_space_mode}")

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
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    model_path = random_shapenet_model_path()
    # model_path = cfg.env.target_path
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path

    env = set_env(cfg)

    print(env.action_space)

    