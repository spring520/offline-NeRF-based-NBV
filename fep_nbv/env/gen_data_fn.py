import numpy as np
import torch
import mathutils
import sys
import os
root_path = os.getenv('nbv_root_path', '/default/path')
shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')
distribution_dataset_path = os.getenv('distribution_dataset_path', '/default/distribution/dataset/path')
sys.path.append(root_path)

from fep_nbv.env.utils import tensor2pose

def shapenet_full(num_azimuth=8,num_elevation=5,radius=[2]):
    trajs = [] # pose_point_to(loc=point*scale, target=target*scale, up=np.random.rand(3)) 旋转矩阵的列表
    azimuths = np.linspace(0, 2 * np.pi, num=num_azimuth, endpoint=False)
    elevations = np.linspace(0, np.pi, num=num_elevation)
    radius = np.asarray(radius)
    azimuths,elevations,radius = np.meshgrid(azimuths, elevations, radius)
    azimuths = azimuths.ravel()
    elevations = elevations.ravel()
    radius = radius.ravel()

    x = radius * np.sin(azimuths) * np.cos(elevations)
    y = radius * np.sin(azimuths) * np.sin(elevations)
    z = radius * np.cos(azimuths)

    cameras = np.stack([x, y, z], axis=1)
    directions =  - cameras
    directions = [mathutils.Vector(row) for row in directions]
    for direction in directions:
        direction = direction.normalize()
    rot_quat = torch.tensor([direction.to_track_quat('-Z', 'Y') for direction in directions]) # wxyz
    trajs = torch.concat((rot_quat[:,[1,2,3,0]], torch.tensor(cameras)),dim=1)
    trajs = tensors2poses(trajs)
    return trajs

def shapenet_eval(num_azimuth=8,num_elevation=6,radius=[2]):
    trajs = [] # pose_point_to(loc=point*scale, target=target*scale, up=np.random.rand(3)) 旋转矩阵的列表
    azimuths = np.linspace(0, 2 * np.pi, num=num_azimuth, endpoint=False)
    elevations = np.linspace(0, np.pi, num=num_elevation,endpoint=False)[1:]
    radius = np.asarray(radius)
    azimuths,elevations,radius = np.meshgrid(azimuths, elevations, radius)
    azimuths = azimuths.ravel()
    elevations = elevations.ravel()
    radius = radius.ravel()

    x = radius * np.sin(azimuths) * np.cos(elevations)
    y = radius * np.sin(azimuths) * np.sin(elevations)
    z = radius * np.cos(azimuths)

    cameras = np.stack([x, y, z], axis=1)
    directions =  - cameras
    directions = [mathutils.Vector(row) for row in directions]
    for direction in directions:
        direction = direction.normalize()
    rot_quat = torch.tensor([direction.to_track_quat('-Z', 'Y') for direction in directions]) # wxyz
    trajs = torch.concat((rot_quat[:,[1,2,3,0]], torch.tensor(cameras)),dim=1)
    trajs = tensors2poses(trajs)
    return trajs

def shapenet_init(n=1):
    trajs = [] # pose_point_to(loc=point*scale, target=target*scale, up=np.random.rand(3)) 旋转矩阵的列表
    azimuths = np.random.uniform(0, 2 * np.pi, n)  # [0, 2π)
    elevations = np.random.uniform(0, np.pi, n)   # [0, π]
    radius = 2 * torch.ones((n))

    x = radius * np.sin(azimuths) * np.cos(elevations)
    y = radius * np.sin(azimuths) * np.sin(elevations)
    z = radius * np.cos(azimuths)

    cameras = np.stack([x, y, z], axis=1)
    directions =  - cameras
    directions = [mathutils.Vector(row) for row in directions]
    for direction in directions:
        direction = direction.normalize()
    rot_quat = torch.tensor([direction.to_track_quat('-Z', 'Y') for direction in directions]) # wxyz
    trajs = torch.concat((rot_quat[:,[1,2,3,0]], torch.tensor(cameras)),dim=1)

    trajs = tensors2poses(trajs)
    return trajs

def tensors2poses(trajs):
    temp = []
    for traj in trajs:
        temp.append(tensor2pose(traj).matrix())
    trajs = np.stack(temp,axis=0)
    return trajs