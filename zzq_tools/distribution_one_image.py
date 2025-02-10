# 当NeRF只使用一张图片进行训练时，不确定性的分布
# 先计算只初始化之后的不确定性分布
# PSNR SSIM 和 Uncertainty
import sys
import os
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import random
import mathutils
import tyro
import time
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure
from PIL import Image
import gc
from mathutils import Vector, Matrix, Quaternion
import matplotlib
matplotlib.use('Agg')  # 设置为无界面后端
from scipy.interpolate import griddata



from config import *
from nvf.env.Enviroment import Enviroment
from nvf.active_mapping.active_mapping import ActiveMapper,get_entropy_for_camera_ray_bundle
from nvf.uncertainty.entropy_renderers import VisibilityEntropyRenderer, WeightDistributionEntropyRenderer
from nvf.active_mapping.mapping_utils import to_transform
from nerfstudio.cameras.cameras import Cameras, CameraType
from nvf.env.utils import save_img

'''
# for test the opengl problem
# import pygame
# from OpenGL.GL import glGetString, GL_VERSION, GL_VENDOR, GL_RENDERER

# def get_opengl_info_with_context():
#     # 初始化 pygame 并创建 OpenGL 上下文
#     pygame.init()
#     pygame.display.set_mode((640, 480), pygame.OPENGL | pygame.DOUBLEBUF)

#     # 获取 OpenGL 信息
#     version = glGetString(GL_VERSION)
#     vendor = glGetString(GL_VENDOR)
#     renderer = glGetString(GL_RENDERER)

#     # 打印信息
#     print("OpenGL Version:", version.decode('utf-8'))
#     print("Vendor:", vendor.decode('utf-8'))
#     print("Renderer:", renderer.decode('utf-8'))

#     # 退出 pygame
#     pygame.quit()

# # 调用函数获取 OpenGL 信息
# get_opengl_info_with_context()
'''

def empty_cache():
    torch.cuda.empty_cache(); gc.collect()

def set_aabb(pipeline, aabb):
    pipeline.trainer.pipeline.datamanager.train_dataset.scene_box.aabb[...] = aabb
    pipeline.trainer.pipeline.model.field.aabb[...] = aabb
    
def set_params(cfg, pipeline, planning=True):
    # print('Exp Name: ', exp_name)
    if cfg.method =='WeightDist':
        pipeline.trainer.pipeline.model.renderer_entropy = WeightDistributionEntropyRenderer()

    elif cfg.method == 'NVF':
        
        pipeline.trainer.pipeline.model.field.use_visibility = True
        pipeline.trainer.pipeline.model.field.use_rgb_variance = True
        
        pipeline.trainer.pipeline.model.renderer_entropy = VisibilityEntropyRenderer()
        if planning:
            pipeline.trainer.pipeline.model.renderer_entropy.d0 = cfg.d0
        else:
            pipeline.trainer.pipeline.model.renderer_entropy.d0 = 0.

        pipeline.trainer.pipeline.model.renderer_entropy.use_huber=cfg.use_huber
        pipeline.trainer.pipeline.model.renderer_entropy.use_var=cfg.use_var
        pipeline.trainer.pipeline.model.renderer_entropy.use_visibility = cfg.use_vis
        pipeline.trainer.pipeline.model.renderer_entropy.mu = cfg.mu
        pipeline.use_visibility = True
        pipeline.trainer.pipeline.model.use_nvf = True

        
    else:
        raise NotImplementedError
    
    pipeline.trainer.pipeline.model.use_uniform_sampler = cfg.use_uniform
    print('entropy use uniform sampler:',pipeline.trainer.pipeline.model.use_uniform_sampler)
    pipeline.trainer.pipeline.model.populate_entropy_modules()
    
    pipeline.use_tensorboard = cfg.train_use_tensorboard

    # set_nvf_params_local(cfg, pipeline)
    # set_nvf_params(cfg, pipeline)
    # pipeline.fov = 60.

    set_aabb(pipeline, cfg.object_aabb)

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
    
    elif cfg.scene.name == 'shapenet':
        print('快想办法写写吧')
        pass

    elif cfg.scene.name =='lego':
        # array([[-0.6377874 , -1.14001584, -0.34465557],
    #    [ 0.63374418,  1.14873755,  1.00220573]])
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-0.7, -1.2, -0.345], [0.7, 1.2, 1.1]])
        
        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb
    elif cfg.scene.name =='drums':
        # array([[-1.12553668, -0.74590737, -0.49164271],
        #[ 1.1216414 ,  0.96219957,  0.93831432]])
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-1.2, -0.8, -0.49164271], [1.2, 1.0, 1.0]])
        
        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb

        cfg.cycles_samples = 50000
        # cfg.env.n_init_views = 5
    elif cfg.scene.name =='hotdog':
        # wrong aabb [[-1.22326267 -1.31131911 -0.19066653]
        # [ 1.22326279  1.13520646  0.32130781]]
        
        # correct aabb [[-1.19797897 -1.28603494 -0.18987501]
        # [ 1.19797897  1.10992301  0.31179601]]
        # factor = 3
        cfg.object_aabb = torch.tensor([[-1.3, -1.4, -0.18987501], [1.3, 1.2, 0.5]])

        diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        cfg.env.n_init_views = 5
        # cfg.check_density = True

    elif cfg.scene.name =='room':
        factor = 1
        cfg.object_aabb = torch.tensor([[-12.4, -4.5,-0.22], [4.1, 6.6, 5.2]])
        cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb
        cfg.env.scale = 0.3333 * 0.5
    elif cfg.scene.name =='ship':
        # [[-1.27687299 -1.29963005 -0.54935801]
        # [ 1.37087297  1.34811497  0.728508  ]]
        cfg.object_aabb = torch.tensor([[-1.35, -1.35,-0.54935801], [1.45, 1.45, 0.73]])
        
        diff_box = torch.tensor([[-1.7,-1.7,0.43], [1.7,1.7,3.3]])
        
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        # cfg.env.n_init_views = 3

        # if cfg.d0 > 0.: cfg.d0=0.8

    elif cfg.scene.name =='chair':
        # [[-0.72080803 -0.69497311 -0.99407679]
        # [ 0.65813684  0.70561057  1.050102  ]]

        cfg.object_aabb = torch.tensor([[-0.8, -0.8,-0.99407679], [0.8, 0.8, 1.1]])
        
        diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

    elif cfg.scene.name =='mic':
    #     array([[-1.25128937, -0.90944701, -0.7413525 ],
    #    [ 0.76676297,  1.08231235,  1.15091646]])
        # factor = 2.5
        cfg.object_aabb = torch.tensor([[-1.3, -1.0,-0.7413525], [0.8, 1.2, 1.2]])
        diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        
        # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
        

        cfg.target_aabb = cfg.camera_aabb
        # cfg.env.n_init_views = 5
        # breakpoint()

    elif cfg.scene.name =='materials':
        # [[-1.12267101 -0.75898403 -0.23194399]
        # [ 1.07156599  0.98509198  0.199104  ]]
        # factor = torch.tensor([2.5, 2.5, 3.5]).reshape(-1,3)
        cfg.object_aabb = torch.tensor([[-1.2, -0.8,-0.23194399], [1.2, 1.0, 0.3]])
        # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        cfg.target_aabb = cfg.camera_aabb
        # breakpoint()
    elif cfg.scene.name =='ficus':
        #[[-0.37773791 -0.85790569 -1.03353798]
        #[ 0.55573422  0.57775307  1.14006007]]
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-0.4, -0.9, -1.03353798], [0.6, 0.6, 1.2]])

        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
        cfg.target_aabb = cfg.camera_aabb

        # cfg.env.n_init_views = 5
    else:
        raise NotImplementedError
    env = Enviroment(cfg.env)
    # breakpoint()
    return env

def generate_candidate_viewpoint(num_azimuth=8,num_elevation=8,radius=[2]):
    azimuths = np.linspace(0, 2 * np.pi, num=num_azimuth, endpoint=False)
    elevations = np.linspace(0, np.pi, num=num_elevation)
    radius = np.asarray(radius)
    azimuths,elevations,radius = np.meshgrid(azimuths, elevations, radius)
    azimuths = azimuths.ravel()
    elevations = elevations.ravel()
    radius = radius.ravel()

    return azimuths,elevations,radius

def xyz2pose(x,y,z):
    cameras = np.stack([x, y, z], axis=1)
    cameras = torch.tensor(cameras, dtype=torch.float32)
    directions = -cameras # 生成相机的方向，从相机位置指向原点
    
    directions = [mathutils.Vector(row) for row in directions.cpu().numpy()]
    for direction in directions:
        direction = direction.normalize()
    rot_quat = torch.tensor([direction.to_track_quat('-Z', 'Y') for direction in directions])

    pose_hist = torch.concat((rot_quat[:,[1,2,3,0]], cameras),dim=1)
    return pose_hist

def polar2pose(azimuths,elevations,radius):
    # 输入在方位角、俯仰角的采样点数，以及半径的采样点
    # 输出采样结果，其中方位角和俯仰角均匀采样
    # 按照 xyzw xyz的顺序排列
    x = radius * np.sin(elevations) * np.cos(azimuths)
    y = radius * np.sin(elevations) * np.sin(azimuths)
    z = radius * np.cos(elevations)

    pose = xyz2pose(x,y,z)
    return pose


def NeRF_init(cfg):
    NeRF_pipeline = ActiveMapper()
    NeRF_pipeline.fov = cfg.env.fov
    NeRF_pipeline.train_img_size = cfg.env.resolution
    config_path = NeRF_pipeline.initialize_config(config_home = "cfg/", dataset_path = "outputs/pipeline/dataset", model=cfg.model)
    NeRF_pipeline.reset()
    NeRF_pipeline.config.max_num_iterations = cfg.train_iter
    set_params(cfg,NeRF_pipeline)
    NeRF_pipeline.trainer.pipeline.model.renderer_entropy.set_iteration(0)

    return NeRF_pipeline

def Uncertainty_distribution_calculation(cfg, NeRF_pipeline, poses, gt_images, input_viewpoint_index):
    Uncertainty = {'PSNR':[],"uncertainty":[],'SSIM':[],"MSE":[]}
    
    # transforms = [to_transform(pose[[3,0,1,2,4,5,6]]) for pose in poses]
    transforms = [to_transform(pose) for pose in poses]
    fov = cfg.env.fov /180 *np.pi
    width = cfg.env.resolution[1] * torch.ones(len(transforms),1, dtype=torch.float32)
    height = cfg.env.resolution[0] * torch.ones(len(transforms),1, dtype=torch.float32)
    fx = 0.5*width/np.tan(fov/2)
    fy = fx
    cx = width//2
    cy = height//2

    for i in range(len(transforms)):
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=transforms[i][:-1,:] # 3x4
            ).to(NeRF_pipeline.trainer.device)
        camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=None)
        outputs = NeRF_pipeline.trainer.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        gt_images[i] = NeRF_pipeline.trainer.pipeline.model.renderer_rgb.blend_background(gt_images[i])
        predicted_rgb, gt_rgb = NeRF_pipeline.trainer.pipeline.model.renderer_rgb.blend_background_for_loss_computation(
            gt_image=gt_images[i], pred_image=outputs["rgb"], pred_accumulation=outputs["accumulation"]
        ) # 3 512 512; 512 512 3;
        # print(f'predicted_rgb max: {predicted_rgb.max()}')
        # print(f'gt_rgb max: {gt_rgb.max()}')
        # print(f'gt_images[i] max {gt_images[i].max()}')

        psnr = NeRF_pipeline.trainer.pipeline.model.psnr(gt_rgb/255,predicted_rgb)
        ssim = structural_similarity_index_measure(gt_rgb.permute(2, 0, 1).unsqueeze(0)/255,predicted_rgb.permute(2, 0, 1).unsqueeze(0))
        mse = torch.mean((gt_rgb/255 - predicted_rgb) ** 2)
        
        Uncertainty['PSNR'].append(psnr.item())
        Uncertainty['SSIM'].append(ssim.item())
        Uncertainty['MSE'].append(mse.item())
        Uncertainty['uncertainty'].append(outputs['entropy'].mean().item())

        if i==input_viewpoint_index:
            image1 = Image.fromarray(gt_rgb.cpu().detach().numpy().astype(np.uint8))
            image2 = Image.fromarray((predicted_rgb*255).cpu().detach().numpy().astype(np.uint8))
        if i==0:
            gt_rgb_0 = gt_rgb
            predicted_rgb_0=predicted_rgb*255
        if i==32:
            gt_rgb_32 = gt_rgb
            predicted_rgb_32=predicted_rgb*255

    if input_viewpoint_index<(len(transforms)/2-input_viewpoint_index):
        image3 = Image.fromarray(gt_rgb_32.cpu().detach().numpy().astype(np.uint8))
        image4 = Image.fromarray(predicted_rgb_32.cpu().detach().numpy().astype(np.uint8))
    else:
        image3 = Image.fromarray(gt_rgb_0.cpu().detach().numpy().astype(np.uint8))
        image4 = Image.fromarray(predicted_rgb_0.cpu().detach().numpy().astype(np.uint8))
    width, height = image1.size
    canvas = Image.new("RGB", (2 * width, 2 * height))
    canvas.paste(image1, (0, 0))  # 左上角
    canvas.paste(image2, (width, 0))  # 右上角
    canvas.paste(image3, (0, height))  # 左下角
    canvas.paste(image4, (width, height))  # 右下角
        
    return Uncertainty,canvas

def visualize_distribution_one_image(cfg, input_viewpoint_index, poses, Uncertainty_before, Uncertainty_after,gt_uncertainty_path):
    camera_position = mathutils.Vector(cfg.camera_aabb[0])
    target_position = mathutils.Vector(cfg.camera_aabb[1])
    direction = target_position - camera_position 
    rot_quat = torch.tensor(direction.to_track_quat('-Z', 'Y'))

    os.makedirs('zzq_tools/distribution_after_single_image', exist_ok=True)
    os.makedirs('zzq_tools/obs', exist_ok=True)
    with open(f'zzq_tools/distribution_after_single_image/{input_viewpoint_index}_uncertainty.txt','w') as f:
        pass
    
    for key in Uncertainty_before.keys():
        with open(f'zzq_tools/distribution_after_single_image/{input_viewpoint_index}_uncertainty.txt','a') as f:
            f.write(f'{key}:\n')
            value_str = ','.join(map(str,Uncertainty_before[key]))
            f.write(f'before train:{value_str}\n')
            value_str = ','.join(map(str,Uncertainty_after[key]))
            f.write(f'after train: {value_str}\n')

        min_ = min(min(Uncertainty_before[key]),min(Uncertainty_after[key]))
        max_ = max(max(Uncertainty_before[key]),max(Uncertainty_after[key]))

        norm = Normalize(vmin=min_ + (max_ - min_) * 0.5, vmax=max_)
        cmap = cm.get_cmap('plasma')

        colors = cmap(norm(Uncertainty_before[key]))
        env = set_env(cfg)
        for i,pose in enumerate(poses):
            # pose = torch.tensor(pose)
            direction = Quaternion(pose[[3,0,1,2]]) @ Vector((0,0,1))
            direction[1] = - direction[1]
            direction = mathutils.Vector(direction).normalized().to_track_quat('-Z', 'Y')
            direction = torch.tensor(direction)
            env.scene.add_cone_at_pose2(pose[-3:],pose[[3,0,1,2]],id=i,color=colors[i])
        # 渲染固定斜上方视角下的目标图像并保存
        fixed_pose = torch.concat((rot_quat, torch.tensor(camera_position)))
        image = env.scene.render_pose(fixed_pose)
        # 图像存在log_path的views目录下，需要创建views并保存
        save_img(image, f'{gt_uncertainty_path}{input_viewpoint_index}/{key}_{input_viewpoint_index}_before_train.png')

        colors = cmap(norm(Uncertainty_after[key]))
        env = set_env(cfg)
        for i,pose in enumerate(poses):
            # pose = torch.tensor(pose)
            direction = Quaternion(pose[[3,0,1,2]]) @ Vector((0,0,1))
            direction[1] = - direction[1]
            direction = mathutils.Vector(direction).normalized().to_track_quat('-Z', 'Y')
            direction = torch.tensor(direction)
            env.scene.add_cone_at_pose2(pose[-3:],pose[[3,0,1,2]],id=i,color=colors[i])
        # 渲染固定斜上方视角下的目标图像并保存
        fixed_pose = torch.concat((rot_quat, torch.tensor(camera_position)))
        image = env.scene.render_pose(fixed_pose)
        print('render_pose success')
        # 图像存在log_path的views目录下，需要创建views并保存
        save_img(image, f'{gt_uncertainty_path}{input_viewpoint_index}/{key}_{input_viewpoint_index}_after_train.png')


def visualize_distribution_ball_version(cfg, input_viewpoint_index, poses, Uncertainty_before, Uncertainty_after,gt_uncertainty_path):
    # 把不确定性画到一个球面上
    camera_position = mathutils.Vector(cfg.camera_aabb[0])
    target_position = mathutils.Vector(cfg.camera_aabb[1])
    direction = target_position - camera_position 
    rot_quat = torch.tensor(direction.to_track_quat('-Z', 'Y'))

    os.makedirs('zzq_tools/distribution_after_single_image', exist_ok=True)
    os.makedirs('zzq_tools/obs', exist_ok=True)
    with open(f'zzq_tools/distribution_after_single_image/{input_viewpoint_index}_uncertainty.txt','w') as f:
        pass

    for key in Uncertainty_before.keys():
        with open(f'zzq_tools/distribution_after_single_image/{input_viewpoint_index}_uncertainty.txt','a') as f:
            f.write(f'{key}:\n')
            value_str = ','.join(map(str,Uncertainty_before[key]))
            f.write(f'before train:{value_str}\n')
            value_str = ','.join(map(str,Uncertainty_after[key]))
            f.write(f'after train: {value_str}\n')

        min_ = min(min(Uncertainty_before[key]),min(Uncertainty_after[key]))
        max_ = max(max(Uncertainty_before[key]),max(Uncertainty_after[key]))

        norm = Normalize(vmin=min_ + (max_ - min_) * 0.5, vmax=max_)
        cmap = cm.get_cmap('plasma')

        phi_grid, theta_grid = np.mgrid[0:2*np.pi:200j, 0:np.pi:200j]
        radius_ = 2
        x_grid = radius_ * np.sin(phi_grid) * np.cos(theta_grid)
        y_grid = radius_ * np.sin(phi_grid) * np.sin(theta_grid)
        z_grid = radius_ * np.cos(phi_grid)

        points = poses
        grid_points = np.array([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T  # 网格点
        grid_values = griddata(points[:,4:].numpy(), norm(Uncertainty_after[key]), grid_points, method='nearest', fill_value=0)
        grid_values = grid_values.reshape(x_grid.shape)

        # 绘制球面热力图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制球面，颜色表示值
        surf = ax.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(grid_values), rstride=1, cstride=1, alpha=0.9)

        # 添加颜色条
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(norm(Uncertainty_after[key]))
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Value')
        # 设置标题和显示
        ax.set_box_aspect([1, 1, 1])  # 等比例缩放
        ax.set_title("3D Spherical Heatmap")
        ax.view_init(elev=45, azim=45)
        plt.savefig(f'{gt_uncertainty_path}{input_viewpoint_index}/{key}_{input_viewpoint_index}_after_train.png')



    pass

if __name__=="__main__":
    cfg = tyro.cli(ExpConfig) # config
    cfg.env.scale = 0.3333 * 0.5 / 6
    num_azimuth = 8
    num_elevation = 8
    radius = [2]
    num_candidate_viewpoint = num_azimuth*num_elevation*len(radius)
    start_viewpoint_index = 0
    viewpoint_index_interval = 1
    

    # 生成GT images
    gt_images_path = f'zzq_tools/test_1130/{num_azimuth}_{num_elevation}_{len(radius)}/gt_obs/'
    gt_uncertainty_path = f'zzq_tools/test_1130/{num_azimuth}_{num_elevation}_{len(radius)}/gt_uncertainty/'
    env = set_env(cfg)
    # 如果gt_images_path存在，以及这个目录下存在{num_candidate_viewpoint-1}.png 那么就不用生成直接读取
    if os.path.exists(gt_images_path) and os.path.exists(f'{gt_images_path}{num_candidate_viewpoint-1}_gt.png'):
        print('loading from existed gt images')
        image_files = sorted([f for f in os.listdir(gt_images_path) if f.endswith("_gt.png")])
        gt_images = []
        for file_name in image_files:
            file_path = os.path.join(gt_images_path, file_name)
            img = Image.open(file_path).convert("RGB")  # 确保是 RGB 格式
            img = img.resize((512, 512))  # 如果图片不是 512x512，可以调整大小
            img_array = np.array(img)  # 转换为 NumPy 数组，形状为 (512, 512, 3)
            gt_images.append(img_array)
        gt_images = np.stack(gt_images,axis=0)
        gt_images = torch.from_numpy(gt_images)
        gt_images = gt_images.float().to('cuda')
    else:
        NeRF_pipeline = NeRF_init(cfg)
        t1 = time.time()
        _,_,_,_ = env.step(poses)
        t2 = time.time()
        print(f'generating gt images, time used: {t2-t1:2f} seconds')
        gt_images = np.array(env.obs_history[-num_candidate_viewpoint:])
        gt_images = torch.stack([torch.FloatTensor(iii)[...,:3] for iii in gt_images]).to(NeRF_pipeline.trainer.device)
        print('saving groundtruth images')
        if not os.path.exists(gt_images_path):
            os.makedirs(gt_images_path,exist_ok=True)
        print(gt_images.shape)
        for index,gt_image in enumerate(gt_images):
            image = Image.fromarray(gt_image.cpu().detach().numpy().astype(np.uint8))
            image.save(f'{gt_images_path}{index}_gt.png')


    for input_viewpoint_index in range(start_viewpoint_index,num_candidate_viewpoint,viewpoint_index_interval):
        empty_cache()

        print(f'dealing with {input_viewpoint_index}')
        # input_viewpoint_index = random.randint(0,len(azimuths)-1) # 随机采样一个视角
        input_view_point = (azimuths[input_viewpoint_index:input_viewpoint_index+1],elevations[input_viewpoint_index:input_viewpoint_index+1],radius[input_viewpoint_index:input_viewpoint_index+1])
        input_pose = polar2pose(input_view_point[0],input_view_point[1],input_view_point[2]) # xyzw xyz

        # 创建NeRF
        NeRF_pipeline = NeRF_init(cfg)

        # 在训练之前可视化每个视角上的不确定性
        Uncertainty_before,image_before = Uncertainty_distribution_calculation(cfg, NeRF_pipeline, poses, gt_images,input_viewpoint_index)
        empty_cache() 
        # 训练NeRF
        env = set_env(cfg)
        obs,_,_,_=env.step(input_pose)
        t1 = time.time()
        NeRF_pipeline.add_image(images=obs,poses=input_pose,model_option=None)
        t2 = time.time()
        print(f'NeRF trained, time used: {t2-t1:2f} seconds')

        # 计算每个视角上的不确定性
        Uncertainty_after,image_after = Uncertainty_distribution_calculation(cfg, NeRF_pipeline, poses, gt_images,input_viewpoint_index)

        # 摄像机视角
        visualize_distribution_one_image(cfg,input_viewpoint_index, poses, Uncertainty_before, Uncertainty_after,gt_uncertainty_path)
        # visualize_distribution_ball_version(cfg, input_viewpoint_index, poses, Uncertainty_before, Uncertainty_after,gt_uncertainty_path)
        
        image_before.save(f'zzq_tools/obs/{input_viewpoint_index}_before_obs.png')
        image_after.save(f'zzq_tools/obs/{input_viewpoint_index}_after_obs.png')





