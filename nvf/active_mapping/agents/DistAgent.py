
import gtsam
import numpy as np
import torch
import nerfacc
from nerfstudio.cameras.cameras import Cameras, CameraType
from nvf.env.utils import GIFSaver
from PIL import ImageDraw

from nerfstudio.model_components.renderers import RGBVarianceRenderer
from nvf.active_mapping.agents.Sampler import *
from nvf.active_mapping.active_mapping import ActiveMapper
from nvf.active_mapping.mapping_utils import to_transform
from nvf.env.utils import pose2tensor, tensor2pose

from nvf.env.utils import empty_cache
from nvf.metric.mesh_metrics import VisibilityMeshMetric, FaceIndexShader

from dataclasses import dataclass, field

from nvf.env.Scene import *
from nvf.env.utils import get_conf
from nvf.env.utils import get_images

# from eval import set_env

from torch.cuda.amp import GradScaler, autocast
import time
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nvf.active_mapping.mapping_utils import *

class DistAgent():
    use_ckpts:bool = False

    def __init__(self, config):
        # super().__init__()
        self.config = config
        # utils.set_seed(0)

        # init nerf pipeline
        self.pipeline = ActiveMapper()
        self.pipeline.fov = config.env.fov
        self.pipeline.train_img_size = config.env.resolution
        config_path = self.pipeline.initialize_config(config_home = "cfg/", dataset_path = "outputs/pipeline/dataset", model=config.model)
        
        # self.pipeline.clear_dataset()
        # self.pipeline.toggle_config_model_checkpoint(True)
        # self.pipeline.toggle_config_model_checkpoint(True)
        # self.sampler = AABBSampler()
        self.sampler = eval(self.config.sampler)(config)
        self.sampler.pipeline = self.pipeline
        self.n_sample = self.config.n_sample
        self.weight_dist = self.config.weight_dist
        

        self.step = 0

        self.plan_hist = []
        self.obs_hist = []
        self.pose_hist = []
        self.entropy = [0,0,0,0]
        self.rendered_entropy_gif = None

        self.time_record = {'plan_time':[], 'train_time':[]}
        print('Using Dist Agent')

        self.k = 360 # 1
        self.k_b = 1.38*(10**(-23)) # 1
        self.T = 300 # 

    def process_obs(self, obs, prev_poses):
        """
        Process the given observations and previous poses.
        Args:
            obs (list): List of observations.
            prev_poses (list): List of previous poses.
        Returns:
            None
        """
        self.obs_hist += obs
        self.pose_hist += prev_poses
        if self.step ==0:
            add_image_option = None
            self.start_step = len(obs)
        else:
            add_image_option = None #'reinit'
        if self.step == self.config.horizon-1:
            print('agent last step')
        
        self.pipeline.add_image(images=obs, poses=prev_poses, model_option=add_image_option)
        # obs 3 512 512 4 max 255
        # poses 3 7
        # model_option: reinit
        self.current_pose = tensor2pose(prev_poses[-1])
    
    def get_reward(self, poses):
        enren  = self.pipeline.trainer.pipeline.model.renderer_entropy
        d0 = '' if not hasattr(enren, 'depth_threshold') else f'd0: {enren.depth_threshold}'
        print('Entropy Type:',type(enren), d0)
        
        plan_result = {"pose":poses.detach().cpu().numpy()}

        with torch.no_grad():
            cost = self.pipeline.get_cost(poses=poses[:,None,:], return_image=True)
            # plan_result["entropy"]= cost.detach().cpu().numpy()
            cost = cost.mean(dim=(-1, -2))
            plan_result["entropy"]= cost.detach().cpu().numpy()
            # print(cost.shape)
        self.plan_hist.append(plan_result)

        return cost
       

    def act(self, obs,prev_poses):
        """
        Takes in the current observation and previous poses and performs the following steps:
        1. Updates the internal NeRF model based on the new observation and previous poses.
        2. Performs planning by sampling poses using a sampler.
        3. Calculates the reward for each sampled pose.
        4. Selects the pose with the highest reward as the best pose.
        5. Updates the step count and records the training and planning times.
        Args:
            obs (object): The current observation.
            prev_poses (list): The list of previous poses.
        Returns:
            list: A list containing the best pose.
        """
        t0 = time.time()
        empty_cache()
        print('Start Training NeRF')
        self.process_obs(obs, prev_poses) # 根据新的观察和之前的pose更新内部的NeRF
        t1 = time.time()
        print('Start Planning')
        empty_cache()

        poses = self.sampler(self.n_sample, pose=self.current_pose) # n_sample x 7
        poses = [pose for pose in poses if not any((pose == t).all() for t in self.pose_hist)]
        poses = torch.stack(poses,dim=0)
        cost = self.get_reward(poses)

        # 加入移动距离
        ## 计算与上一个pose的移动距离
        current_poses = poses.cpu().numpy()
        prev_pose = self.pose_hist[-1].cpu().numpy()
        dist = np.linalg.norm(current_poses[:, -3:] - prev_pose[-3:], axis=1) # 欧式距离
        dist = self.k/self.k_b/self.T*dist
        # dist = np.abs(current_poses[:, -3:] - prev_pose[-3:]).sum(axis=1) # 曼哈顿距离
        # print('Distance:', dist)
        cost = cost - self.weight_dist*dist[:,None]
        print(f'距离前的系数：{self.k/self.k_b/self.T*self.weight_dist}')

        best_idx = cost.argmax().item()
        best_cost = cost[best_idx].item()
        best_pose = poses[best_idx, ...]
        empty_cache()
        self.step +=1
        t2 = time.time()
        
        
        self.entropy = self.entropy_calc_uniform_sample()
        t3 = time.time()
        print(f'enropy calculation spent: {t3-t2}s')

        self.time_record['train_time'].append(t1-t0)
        self.time_record['plan_time'].append(t2-t1)
        return [best_pose, best_cost]
    

    def entropy_calc_uniform_sample(self): # type: ignore
        # 从object_aabb的6个方向进行均匀采样，计算几种不同的熵函数
        ray_bundle_list = []
        num_grid = 16

        # z轴
        x = torch.linspace(-1, 1, num_grid)
        y = torch.linspace(-1.1, 1.1, num_grid)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        # 负方向
        z = torch.ones(num_grid * num_grid) * 2.8
        origins = torch.stack([grid_x.flatten(), grid_y.flatten(), z], dim=-1).to('cuda')
        directions = torch.stack([torch.zeros(num_grid * num_grid), torch.zeros(num_grid * num_grid), torch.ones(num_grid * num_grid) * -1], dim=-1).to('cuda')
        pixel_area = torch.ones(num_grid * num_grid,1).to('cuda')*0.1
        ray_bundle_uniform = RayBundle(origins,directions,pixel_area)
        ray_bundle_uniform.camera_indices = torch.arange(num_grid * num_grid).unsqueeze(dim=1).to('cuda')
        ray_bundle_list.append(ray_bundle_uniform)
        # 正方向
        z = torch.ones(num_grid * num_grid) * -2.1
        origins = torch.stack([grid_x.flatten(), grid_y.flatten(), z], dim=-1).to('cuda')
        directions = torch.stack([torch.zeros(num_grid * num_grid), torch.zeros(num_grid * num_grid), torch.ones(num_grid * num_grid) * 1], dim=-1).to('cuda')
        pixel_area = torch.ones(num_grid * num_grid,1).to('cuda')*0.1
        ray_bundle_uniform = RayBundle(origins,directions,pixel_area)
        ray_bundle_uniform.camera_indices = torch.arange(num_grid * num_grid).unsqueeze(dim=1).to('cuda')
        ray_bundle_list.append(ray_bundle_uniform)

        # x轴   
        y = torch.linspace(-1.1, 1.1, num_grid)
        z = torch.linspace(-1, 1.8, num_grid)
        grid_y, grid_z = torch.meshgrid(y, z, indexing='ij')
        # 负方向
        x = torch.ones(num_grid * num_grid) * 2
        origins = torch.stack([x, grid_y.flatten(), grid_z.flatten()], dim=-1).to('cuda')
        directions = torch.stack([torch.ones(num_grid * num_grid)*-1, torch.zeros(num_grid * num_grid), torch.zeros(num_grid * num_grid)], dim=-1).to('cuda')
        pixel_area = torch.ones(num_grid * num_grid,1).to('cuda')*0.1
        ray_bundle_uniform = RayBundle(origins,directions,pixel_area)
        ray_bundle_uniform.camera_indices = torch.arange(num_grid * num_grid).unsqueeze(dim=1).to('cuda')
        ray_bundle_list.append(ray_bundle_uniform)
        # 正方向
        x = torch.ones(num_grid * num_grid) * -2
        origins = torch.stack([x, grid_y.flatten(), grid_z.flatten()], dim=-1).to('cuda')
        directions = torch.stack([torch.ones(num_grid * num_grid), torch.zeros(num_grid * num_grid), torch.zeros(num_grid * num_grid)], dim=-1).to('cuda')
        pixel_area = torch.ones(num_grid * num_grid,1).to('cuda')*0.1
        ray_bundle_uniform = RayBundle(origins,directions,pixel_area)
        ray_bundle_uniform.camera_indices = torch.arange(num_grid * num_grid).unsqueeze(dim=1).to('cuda')
        ray_bundle_list.append(ray_bundle_uniform)

        # y轴   
        x = torch.linspace(-1, 1, num_grid)
        z = torch.linspace(-1, 1.8, num_grid)
        grid_x, grid_z = torch.meshgrid(x, z, indexing='ij')
        # 负方向
        y = torch.ones(num_grid * num_grid) * 2.1
        origins = torch.stack([grid_x.flatten(), y, grid_z.flatten()], dim=-1).to('cuda')
        directions = torch.stack([torch.zeros(num_grid * num_grid), torch.ones(num_grid * num_grid)*-1, torch.zeros(num_grid * num_grid)], dim=-1).to('cuda')
        pixel_area = torch.ones(num_grid * num_grid,1).to('cuda')*0.1
        ray_bundle_uniform = RayBundle(origins,directions,pixel_area)
        ray_bundle_uniform.camera_indices = torch.arange(num_grid * num_grid).unsqueeze(dim=1).to('cuda')
        ray_bundle_list.append(ray_bundle_uniform)
        # 正方向
        y = torch.ones(num_grid * num_grid) * -2.1
        origins = torch.stack([grid_x.flatten(), y, grid_z.flatten()], dim=-1).to('cuda')
        directions = torch.stack([torch.zeros(num_grid * num_grid), torch.ones(num_grid * num_grid), torch.zeros(num_grid * num_grid)], dim=-1).to('cuda')
        pixel_area = torch.ones(num_grid * num_grid,1).to('cuda')*0.1
        ray_bundle_uniform = RayBundle(origins,directions,pixel_area)
        ray_bundle_uniform.camera_indices = torch.arange(num_grid * num_grid).unsqueeze(dim=1).to('cuda')
        ray_bundle_list.append(ray_bundle_uniform)

        entropy = [0,0,0,0]
        for ray_bundle in ray_bundle_list:
            ray_samples, ray_indices = self.pipeline.trainer.pipeline.model.sampler(
                ray_bundle=ray_bundle, # 256
                near_plane=self.pipeline.trainer.pipeline.model.config.near_plane, # 0.05 
                far_plane=self.pipeline.trainer.pipeline.model.config.far_plane,  # 1000
                render_step_size=self.pipeline.trainer.pipeline.model.config.render_step_size, # 0.006928203105926514
                alpha_thre=self.pipeline.trainer.pipeline.model.config.alpha_thre, # 0.01
                cone_angle=self.pipeline.trainer.pipeline.model.config.cone_angle, # 0.004
                )

            field_outputs = self.pipeline.trainer.pipeline.model.field(ray_samples) # type: ignore
            packed_info = nerfacc.pack_info(ray_indices, len(ray_bundle))
            weights = nerfacc.render_weight_from_density( # 维度
                t_starts=ray_samples.frustums.starts[..., 0],
                t_ends=ray_samples.frustums.ends[..., 0],
                sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
                packed_info=packed_info,
            )[0]
            weights = weights[..., None]
            entropy[1] += ((weights*field_outputs[FieldHeadNames.RGB_VARIANCE]*(1-field_outputs[FieldHeadNames.VISIBILITY])).sum()/len(field_outputs[FieldHeadNames.RGB_VARIANCE])).item()

            # 计算一个方向上射线的熵
            entropy[2] += get_entropy_for_ray_bundle_uniform(self.pipeline.trainer.pipeline.model, ray_bundle_uniform).mean().item() # type: ignore

            entropy[3] += self.pipeline.trainer.pipeline.model.renderer_entropy(field_outputs,ray_samples).item() # type: ignore

            entropy[1] += (weights*field_outputs[FieldHeadNames.RGB_VARIANCE]).mean().item()
        
        entropy = [e/len(ray_bundle_list) for e in entropy]

        return entropy
    
    def visualize_entropy(self):
        # 沿着一个圆形轨道对相机进行采样，画出每个位置熵的图像
        # 生成一系列相机位姿，以原点为中心，半径为2的园
        num_cameras = 100
        radius = 3
        theta = np.linspace(0, 2*np.pi, num_cameras)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros(num_cameras)

        cameras = np.stack([x, y, z], axis=1)
        cameras = torch.tensor(cameras, dtype=torch.float32).to('cuda')
        cameras = cameras.unsqueeze(1)
        # 生成相机的方向，从相机位置指向原点
        directions = -cameras
        directions[:,:,1] = -directions[:,:,1]
        directions = directions.squeeze(1)
        directions = [mathutils.Vector(row) for row in directions.cpu().numpy()]
        for direction in directions:
            direction.normalize()
        rot_quat = torch.tensor([direction.to_track_quat('-Z', 'Y') for direction in directions]).to('cuda')
        poses = torch.concat((rot_quat[:,[3,0,1,2]], torch.tensor(cameras).squeeze()),dim=1)
        transforms = [to_transform(pose) for pose in poses]
        
        gif = []

        fov = self.config.env.fov /180 *np.pi
        width = self.config.env.resolution[1] * torch.ones(num_cameras,1, dtype=torch.float32)
        height = self.config.env.resolution[0] * torch.ones(num_cameras,1, dtype=torch.float32)
        fx = 0.5*width/np.tan(fov/2)
        fy = fx
        cx = width//2
        cy = height//2
        
        for i in range(num_cameras):
            cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=transforms[i][:-1,:] # 3x4
            ).to(self.pipeline.trainer.device)


            camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=None)
            entropy = get_entropy_for_camera_ray_bundle(self.pipeline.trainer.pipeline.model, camera_ray_bundle) # 512 512 1
            gif.append(entropy)

        # gif后处理
        # 归一化到0-1之间
        gif = torch.stack(gif).squeeze()
        gif = gif - gif.min()
        gif = gif / gif.max() * 255
        # 将这个图像序列存成gif对象
        gif = gif.cpu().numpy().astype(np.uint8)
        gif = [Image.fromarray(gif[i, :, :], mode='L') for i in range(gif.shape[0])]

        # entropy_gif = GIFSaver()
        # for i in range(len(gif)):
        #     entropy_gif.add(gif[i], fn=lambda img: ImageDraw.Draw(img).text((3, 3), f"T={i}", fill=(255)))
        

        return gif
            




       

class RandomAgent(DistAgent):
    train_each_iter = True

    def __init__(self, config):
        super().__init__(config)
        self.sampler.pipeline = None
    
    def process_obs(self, obs, prev_poses):
        self.obs_hist += obs
        self.pose_hist += prev_poses
        if self.train_each_iter:
            if self.step ==0:
                add_image_option = None
            else:
                add_image_option = 'reinit'
            self.pipeline.add_image(images=obs, poses=prev_poses, model_option=add_image_option)  
                  
        elif self.step == self.config.horizon-1:
            self.pipeline.add_image(images=self.obs_hist, poses=self.pose_hist)
        
        self.current_pose = tensor2pose(prev_poses[-1])

    def get_reward(self, poses):
        # breakpoint()
        return torch.rand(poses.shape[0], 1, device=poses.device)
    
class OptAgent(DistAgent):
    use_ckpts:bool = False
    
    def get_reward(self, poses):
        plan_result = {"pose":poses.detach().cpu().numpy()}

        with torch.no_grad():
            cost = self.pipeline.get_cost(poses=poses[:,None,:], return_image=True)
            plan_result["entropy"]= cost.detach().cpu().numpy()
            cost = cost.mean(dim=(-1, -2))
        
        self.plan_hist.append(plan_result)

        return cost
       
    def act(self, obs,prev_poses):
        t0 = time.time()
        empty_cache()
        print('Start Training NeRF')
        self.process_obs(obs, prev_poses)
        t1 = time.time()
        print('Start Planning')
        empty_cache()        

        device = self.pipeline.trainer.device
        cpu_or_cuda_str: str = self.pipeline.trainer.device.split(":")[0]
        mixed_precision = self.pipeline.trainer.mixed_precision
        aabb = self.config.camera_aabb.to(device)

        # Specify top-k value
        k=self.config.n_opt

        poses_ = self.sampler(self.n_sample, pose=self.current_pose)
        cost = self.pipeline.get_cost(poses=poses_[:,None,:], return_image=True)
        cost = cost.mean(dim=(-1, -2))
        # Get top-k costs
        k_costs,k_idxs = torch.topk(cost.squeeze(1),k)
        topk_poses = poses_[k_idxs,...]
        topk_cost = cost[k_idxs,...]

        poses = torch.tensor(poses_[k_idxs,...], device=device, requires_grad=True) # getTopKPoses  
        # poses.requires_grad = True
        scaler = GradScaler()
        optimizer = torch.optim.Adam([poses] ,lr=self.config.opt_lr)
        print(f"Top {k} Poses Pre Optimization:{poses_[k_idxs,...]}")
        print(f"Cost Pre Optimization:{cost[k_idxs,...].view(-1)}")
        for iter in range(self.config.opt_iter):
            optimizer.zero_grad()
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=mixed_precision):
                cost = -self.pipeline.get_cost(poses[:,None,:], return_image=False)
                

            if cost.requires_grad:
                try: 
                    scaler.scale(cost.sum()).backward(retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as error:
                    # likely cased by no samples within instant-ngp
                    print('error occurs in pose optimization!!!')
                    print(error)
            
            # Normalize quaternion and clip pose according to camera using aabb 
            with torch.no_grad():
                for i in range(0,k):
                    # print("Pose Pre Quaternion Normalization",poses[i])
                    quat_norm = torch.norm(poses[i][0:4], p=2).clone()
                    quat = (poses[i][0:4]/quat_norm)
                    poses[i][0:4]=quat.clone()

                    
                    poses[i][4] = torch.clip(poses[i][4],aabb[0][0],aabb[1][0])
                    poses[i][5] = torch.clip(poses[i][5],aabb[0][1],aabb[1][1])
                    poses[i][6] = torch.clip(poses[i][6],aabb[0][2],aabb[1][2])
                    # print("Pose Post Quaternion Normalization",poses[i])

        best_poses = poses.detach().cpu().clone()

        mask = ~torch.isnan(best_poses).any(dim=1)
        best_poses = best_poses[mask]

        plan_result = {"pose":poses.detach().cpu().numpy()}
        
        use_init_poses = True # use init (topk) pose for selection
        if use_init_poses:
            with torch.no_grad():
                best_poses = torch.cat([best_poses, topk_poses], dim=0)
                cost = self.pipeline.get_cost(poses=best_poses[:,None,:], return_image=True)
        else:
            if best_poses.shape[0] == 0:
                print('No valid poses found, using topk poses')
                best_poses = topk_poses
                cost = topk_cost.unsqueeze(-1).unsqueeze(-1)
            else:
                cost = self.pipeline.get_cost(poses=best_poses[:,None,:], return_image=True)

        
        plan_result["entropy"]= cost.detach().cpu().numpy()
        cost = cost.mean(dim=(-1, -2))
        self.plan_hist.append(plan_result)
        
        best_pose = best_poses[cost.argmax().item(),...]
        print(f"Pose Post Optimization:{best_pose}")
        print(f"Cost Post Optimization:{cost.view(-1)}")
        self.step +=1
        del poses
        del poses_
        del best_poses
        del cost

        t2 = time.time()
        self.time_record['train_time'].append(t1-t0)
        self.time_record['plan_time'].append(t2-t1)
        return [best_pose]

if __name__ == "__main__":
    DistAgent()

    