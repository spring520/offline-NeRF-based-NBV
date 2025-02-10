from nltk.corpus import wordnet as wn
import numpy as np
import torch
import mathutils
import sys
import healpy as hp
import pandas as pd
sys.path.append("/home/zhengquan/04-fep-nbv")

from nvf.active_mapping.active_mapping import ActiveMapper
from nvf.uncertainty.entropy_renderers import VisibilityEntropyRenderer, WeightDistributionEntropyRenderer

def offset2word(offset):
    if isinstance(offset,str):
        offset = int(offset)

    pos = 'n' 
    synset = wn.synset_from_pos_and_offset(pos, offset)
    words = synset.lemma_names()

    return words[0]

def generate_candidate_viewpoint(num_azimuth=8,num_elevation=8,radius=[2]):
    azimuths = np.linspace(0, 2 * np.pi, num=num_azimuth, endpoint=False)
    elevations = np.linspace(0, np.pi, num=num_elevation)
    radius = np.asarray(radius)
    azimuths,elevations,radius = np.meshgrid(azimuths, elevations, radius)
    azimuths = azimuths.ravel()
    elevations = elevations.ravel()
    radius = radius.ravel()

    return azimuths,elevations,radius

def generate_HEALPix_viewpoints(n_side = 2,radius = 2):
    num_pixels = hp.nside2npix(n_side) # 计算总像素数
    theta, phi = hp.pix2ang(n_side, np.arange(num_pixels)) # 获取每个像素的中心坐标 (θ, φ)

    # 转换为笛卡尔坐标
    x = radius*np.sin(theta) * np.cos(phi)
    y = radius*np.sin(theta) * np.sin(phi)
    z = radius*np.cos(theta)

    points = np.stack([x, y, z], axis=-1)

    poses = xyz2pose(points[:,0],points[:,1],points[:,2])
    return poses

def index_to_coordinates(index, num_azimuth=8, num_elevation=8, radius=[2]):
    # 获取生成的坐标
    azimuths, elevations, radii = generate_candidate_viewpoint(num_azimuth, num_elevation, radius)

    # 检查索引范围是否有效
    num_points = len(azimuths)
    if index < 0 or index >= num_points:
        raise IndexError(f"Index {index} out of range for total {num_points} points.")

    # 根据索引获取坐标
    azimuth = azimuths[index]
    elevation = elevations[index]
    radius = radii[index]

    return azimuth, elevation, radius



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

def save_dict_to_excel(file, metric):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(metric)

    # Save the DataFrame to an Excel file
    df.to_excel(file, index=True)

def print_cuda_allocated():
    # 当前显存占用（以 MB 为单位）
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024

    print(f"显存占用: {allocated:.2f} MB, 已预留显存: {reserved:.2f} MB")