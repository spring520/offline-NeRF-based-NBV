import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tyro
from config import ExpConfig
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.cameras.rays import RayBundle
from nvf.active_mapping.active_mapping import ActiveMapper
from eval import set_env, set_agent, update_params

if __name__=="__main__":
    sys.argv = ["eval.py", "--scene", "hubble","--method","nvf","--agent","dist","--weight_dist","0.5"]
    cfg = tyro.cli(ExpConfig)
    env = set_env(cfg)
    agent = set_agent(cfg)
    if cfg.agent == "VisibilityAgent":
        agent.init_visibility(env)
    agent.pipeline.reset()
    update_params(cfg, agent, 0)

    # 直接创建ray_bundle
    # origin从 z = 2.8的平面上进行采样，x的范围-1到1 y的范围-1.1到1.1
    # direction指向z轴负方向
    # camera_indices 从0 到512*512-1
    # pixel_area全是1

    num_grid = 16
    x = torch.linspace(-1, 1, num_grid)
    y = torch.linspace(-1.1, 1.1, num_grid)
    z = torch.ones(num_grid * num_grid) * 1.81
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    origins = torch.stack([grid_x.flatten(), grid_y.flatten(), z], dim=-1).to('cuda:0')
    directions = torch.stack([torch.zeros(num_grid * num_grid), torch.zeros(num_grid * num_grid), torch.ones(num_grid * num_grid) * -1], dim=-1).to('cuda:0')
    pixel_area = torch.ones(num_grid * num_grid,1).to('cuda:0')*0.1

    ray_bundle_uniform = RayBundle(origins,directions,pixel_area)
    ray_bundle_uniform.camera_indices = torch.arange(num_grid * num_grid).unsqueeze(dim=1).to('cuda:0')

    ray_samples, ray_indices = agent.pipeline.trainer.pipeline.model.sampler(
            ray_bundle=ray_bundle, # 256
            near_plane=agent.pipeline.trainer.pipeline.model.config.near_plane, # 0.05
            far_plane=agent.pipeline.trainer.pipeline.model.config.far_plane, # 1000
            render_step_size=agent.pipeline.trainer.pipeline.model.config.render_step_size, # 0.006928203105926514
            alpha_thre=agent.pipeline.trainer.pipeline.model.config.alpha_thre, # 0.01
            cone_angle=agent.pipeline.trainer.pipeline.model.config.cone_angle, # 0.004
        )





    aabb = torch.tensor([[-2,-2,-2],[2,2,2]])
    num_images=401
    log2_hashmap_size = 19
    max_res = 512

    # init Field
    field = NerfactoField(
            aabb=aabb,
            num_images=num_images,
            log2_hashmap_size=log2_hashmap_size,
            max_res=max_res,
            spatial_distortion=None,
            use_rgb_variance=False, # Change to enable only when NeurAR or ActiveNeRF being used
        ) 


    pipeline = ActiveMapper()
    pipeline.fov = cfg.env.fov
    pipeline.train_img_size = config.env.resolution
    config_path = pipeline.initialize_config(config_home = "cfg/", dataset_path = "outputs/pipeline/dataset", model=config.model)

    # 产生普通的raysample 参考datamanager next_train，输入图片输出ray_bundle
    # 创建 3x512x512x4的tensor数组
    # image = torch.rand((3,512,512,4))
    # image_idx = torch.tensor([0,1,2])
    # image_batch = {'image':image,'image_idx':image_idx}


    # 直接创建ray_bundle
    # origin从 z = 2.8的平面上进行采样，x的范围-1到1 y的范围-1.1到1.1
    # direction指向z轴负方向
    # camera_indices 从0 到512*512-1
    ray_bundle = RayBundle()
    ray_bundle.origin = torch.stack([torch.linspace(-1,1,512),torch.linspace(-1.1,1.1,512),torch.ones(512)*2.8],dim=-1)
    ray_bundle.direction = torch.stack([torch.zeros(512),torch.zeros(512),torch.ones(512)*-1],dim=-1)
    ray_bundle.camera_indices = torch.arange(512*512)

    ray_samples, ray_indices = field.sampler(
            ray_bundle=ray_bundle, # 256
            near_plane=self.config.near_plane, # 0.05
            far_plane=self.config.far_plane, # 1000
            render_step_size=self.config.render_step_size, # 0.006928203105926514
            alpha_thre=self.config.alpha_thre, # 0.01
            cone_angle=self.config.cone_angle, # 0.004
        )


    