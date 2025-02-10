# 可视化训练后NeRF的不确定性分布
# 输入时NeRF本身，gt image(用来计算PSNR那些)
# 输出几张图像

# 再实现一个函数，把算好的不确定性列表画成图
import sys
import os
from tqdm import tqdm
from pathlib import Path
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import math
import numpy as np
sys.path.append("/home/zhengquan/04-fep-nbv")

from fep_nbv.utils.utils import *
from fep_nbv.env.utils import *
from fep_nbv.utils.generate_viewpoints import *
from fep_nbv.utils.transform_viewpoints import *

def visualize_HEALPIix_distribution_sphere(Uncertainty,mode='uncertainty',n_side=2,save_path=None,original_viewpoint=np.array([0,0,1])):
    poses = generate_HEALPix_viewpoints(n_side=n_side,original_viewpoint=original_viewpoint)

    if mode!=None:
        values = Uncertainty[mode]
    else:
        mode = '-'
        values = Uncertainty
    min_ = np.min(values)
    max_ = np.max(values)
    norm = Normalize(vmin=min_, vmax=max_)

    phi_grid, theta_grid = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    radius_ = 2
    x_grid = radius_ * np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = radius_ * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = radius_ * np.cos(theta_grid)

    grid_values = griddata(
        points=np.array(poses[:,4:]),  # HEALPix 的笛卡尔坐标
        values=norm(values),  # HEALPix 的值
        xi=np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1),
        method='nearest',
        fill_value=0
    ).reshape(x_grid.shape)

    # 绘制球面热力图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制球面，颜色表示值
    surf = ax.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(grid_values), rstride=1, cstride=1, alpha=0.9)

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(norm(Uncertainty[mode]))
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    # 设置标题和显示
    ax.set_box_aspect([1, 1, 1])  # 等比例缩放
    ax.set_title(mode)
    ax.view_init(elev=45, azim=45)
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig(f'data/test/distribution_visualizetion_test/{mode}_{save_path}.png')
    plt.close()

def visualize_HEALPIix_distribution_polar(Uncertainty,mode='uncertainty',n_side=2,save_path=None,original_viewpoint=np.array([0,0,1])):
    poses = generate_HEALPix_viewpoints(n_side=n_side,original_viewpoint=original_viewpoint)
    phi,theta = pose2polar(poses)

    if mode!=None and isinstance(Uncertainty,dict):
        values = Uncertainty[mode]
    else:
        mode = '-'
        values = Uncertainty
    min_ = np.min(values)
    max_ = np.max(values)
    norm = Normalize(vmin=min_, vmax=max_)
    
    phi_grid, theta_grid = np.meshgrid(
        np.linspace(0, 2 * np.pi, 200),
        np.linspace(0, np.pi, 100)
    )

    grid_values = griddata(
        points=np.stack([phi, theta], axis=1),
        values=norm(values),
        xi=np.stack([phi_grid.ravel(), theta_grid.ravel()], axis=1),
        method='nearest',
        fill_value=0
    ).reshape(phi_grid.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    im = ax.contourf(phi_grid, theta_grid, grid_values, cmap='viridis')

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Value')
    ax.set_title("Continuous Polar Projection Visualization")

    # Add labels and title
    plt.xlabel('Azimuth Index')
    plt.ylabel('Elevation Index')
    plt.title(f"{mode.capitalize()} Heatmap")

    # Save or display the heatmap
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig(f'data/test/distribution_visualizetion_test/{mode}_{save_path}_polar.png')

    plt.close()

def visualize_distribution_ball(Uncertainty,mode='uncertainty',save_path=None):
    min_ = min(Uncertainty[mode])
    max_ = max(Uncertainty[mode])
    norm = Normalize(vmin=min_ + (max_ - min_) * 0, vmax=max_)

    phi_grid, theta_grid = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    radius_ = 2
    x_grid = radius_ * np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = radius_ * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = radius_ * np.cos(theta_grid)

    

    a,e,r = generate_candidate_viewpoint()
    origin_points = polar2pose(a,e,r)
    grid_points = np.array([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T  # 网格点
    grid_values = griddata(origin_points[:,4:].numpy(), norm(Uncertainty[mode]), grid_points, method='nearest', fill_value=0)
    grid_values = grid_values.reshape(x_grid.shape)

    # 绘制球面热力图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制球面，颜色表示值
    surf = ax.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(grid_values), rstride=1, cstride=1, alpha=0.9)

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(norm(Uncertainty[mode]))
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    # 设置标题和显示
    ax.set_box_aspect([1, 1, 1])  # 等比例缩放
    ax.set_title(mode)
    ax.view_init(elev=45, azim=45)
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig(f'data/test/distribution_visualizetion_test/{mode}_{save_path}.png')


def visualize_distribution_polar(Uncertainty,mode=None,save_path=None):
    if mode!=None:
        values = Uncertainty[mode]
    else:
        mode = '-'
        values = Uncertainty
    min_ = np.min(values)
    max_ = np.max(values)
    norm = Normalize(vmin=min_, vmax=max_)
    rows=pols=round(math.sqrt(len(values)))
    grid_values = np.array(values).reshape((rows, pols))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(grid_values, cmap='viridis', norm=norm, aspect='auto', origin='upper') 

    cbar = plt.colorbar()
    cbar.set_label('Uncertainty')

    # Add labels and title
    plt.xlabel('Azimuth Index')
    plt.ylabel('Elevation Index')
    plt.title(f"{mode.capitalize()} Heatmap")

    # Save or display the heatmap
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig(f'data/test/distribution_visualizetion_test/{mode}_{save_path}_polar.png')


        


if __name__=='__main__':
    uncertainty_data_path = "/home/zhengquan/04-fep-nbv/data/test/distribution_generation_test/airplane/1a04e3eab45ca15dd86060f189eb133"
    output_path = os.path.join(uncertainty_data_path,'visualization')
    os.makedirs(output_path,exist_ok=True)
    uncertainty_path = os.path.join(uncertainty_data_path,'uncertainties')

    uncertainties = [f for f in os.listdir(uncertainty_path) if f.endswith('json')]
    uncertainties.sort()

    uncertainty_dict = {}
    for index,f in enumerate(tqdm(uncertainties)):
        uncertainty = load_from_json(Path(os.path.join(uncertainty_path,f)))
        uncertainty_dict[index]=uncertainty

    poses = generate_HEALPix_viewpoints(n_side=2)

    for index in tqdm(uncertainty_dict.keys(),desc='index'):
        for mode in tqdm(uncertainty_dict[0].keys(),leave=False,desc='mode'):
            save_path_sphere = os.path.join(output_path,f'{mode}_{index}_sphere.png')
            save_path_polar = os.path.join(output_path,f'{mode}_{index}_polar.png')
            visualize_HEALPIix_distribution_sphere(uncertainty_dict[index],mode,save_path=save_path_sphere,original_viewpoint=np.array(poses[index,4:]))
            visualize_HEALPIix_distribution_polar(uncertainty_dict[index],mode,save_path=save_path_polar,original_viewpoint=np.array(poses[index,4:]))
            # if os.path.exists(save_path) and os.path.exists(save_path_polar):
            #     continue
            # visualize_distribution_ball(uncertainty_dict[index],mode,save_path=save_path)
            # visualize_distribution_polar(uncertainty_dict[index],mode,save_path=save_path_polar)

    # model_path = "/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/03938244"
    # model_paths = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path,f))]
    
    # for model_path in model_paths:
        
    # model_path = "/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/03938244/55a09adb1315764f8a766d8de50bb39d"
    # offset = model_path.split('/')[-2]
    # category = offset2word(offset)

    # dataset_path = model_path.replace('/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2','/remote-home/zzq/data/ShapeNet/distribution_dataset')
    # dataset_path = dataset_path.replace(offset,category)

    # dataset_path_all = '/remote-home/zzq/data/ShapeNet/distribution_dataset/pillow'
    # dataset_paths = [f for f in os.listdir(dataset_path_all) if os.path.isdir(os.path.join(dataset_path_all,f))]
    # dataset_paths = []
    # for dataset_path in dataset_paths:
    #     print(f'dealing with {dataset_path}')
    #     dataset_path = os.path.join(dataset_path_all,dataset_path)
    #     output_path = os.path.join(dataset_path,'visualization')
    #     os.makedirs(output_path,exist_ok=True)

    #     uncertainty_path = os.path.join(dataset_path,'uncertainties')

    #     uncertainties = [f for f in os.listdir(uncertainty_path) if f.endswith('json')]
    #     uncertainties.sort()

    #     uncertainty_dict = {}
    #     for index,f in enumerate(tqdm(uncertainties)):
    #         uncertainty = load_from_json(Path(os.path.join(uncertainty_path,f)))
    #         uncertainty_dict[index]=uncertainty

    #     for index in tqdm(uncertainty_dict.keys(),desc='index'):
    #         for mode in tqdm(uncertainty_dict[0].keys(),leave=False,desc='mode'):
    #             save_path = os.path.join(output_path,f'{mode}_{index}.png')
    #             save_path_polar = os.path.join(output_path,f'{mode}_{index}_polar.png')
    #             # if os.path.exists(save_path) and os.path.exists(save_path_polar):
    #             #     continue
    #             visualize_distribution_ball(uncertainty_dict[index],mode,save_path=save_path)
    #             visualize_distribution_polar(uncertainty_dict[index],mode,save_path=save_path_polar)

