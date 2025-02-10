# 可视化视角生成和旋转的过程
import sys
import os
import json
from tqdm import tqdm
import tyro
import numpy as np
import time
import torch
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure
import matplotlib.pyplot as plt
sys.path.append("/home/zhengquan/04-fep-nbv")

from config import *
from fep_nbv.utils.utils import *
from fep_nbv.env.utils import *
from fep_nbv.env.shapenet_env import set_env
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints,generate_fibonacci_viewpoints,generate_polar_viewpoints
from nerfstudio.cameras.cameras import Cameras, CameraType
from nvf.active_mapping.mapping_utils import to_transform

import warnings
warnings.filterwarnings("ignore")

if __name__=='__main__':
    # candidate_poses = generate_HEALPix_viewpoints(n_side=1)
    relative_poses_0 = generate_HEALPix_viewpoints(n_side=1)
    relative_poses_1 = generate_HEALPix_viewpoints(n_side=1,offset_phi=0.25*np.pi)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    # 绘制球面网格
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    x = 2 * np.sin(v) * np.cos(u)  # 球面半径为 2
    y = 2 * np.sin(v) * np.sin(u)
    z = 2 * np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

    gif = GIFSaver()

    for pose in relative_poses_0:
        scatter = ax.scatter(pose[4], pose[5], pose[6], c=(1,0,0), cmap='viridis', s=50, alpha=0.7)
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif.add(image_array)

    for pose in relative_poses_1:
        scatter = ax.scatter(pose[4], pose[5], pose[6], c=pose[6], cmap='viridis', s=50, alpha=0.7)
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif.add(image_array)
    save_img(image_array,'/home/zhengquan/04-fep-nbv/data/test/test.png')
    gif.save('/home/zhengquan/04-fep-nbv/data/test/test.gif')