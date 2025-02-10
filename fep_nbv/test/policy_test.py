# 根据不确定性的分布结果进行视角选择的策略test、
import tyro
import sys
from tqdm import tqdm
from pathlib import Path
import os
import random
import numpy as np
from PIL import Image
import time
import math
from datetime import timedelta, datetime
import healpy as hp
sys.path.append("/home/zhengquan/04-fep-nbv")

from config import *
from fep_nbv.utils.utils import *
from fep_nbv.env.utils import *
from fep_nbv.env.shapenet_env import set_env
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints,generate_fibonacci_viewpoints,generate_polar_viewpoints,index2pose_HEALPix
from fep_nbv.visualization.uncertainty_distribution_visualization import * 
from nvf.metric.MetricTracker import RefMetricTracker


import warnings
warnings.filterwarnings("ignore")

def Brute_Search(Uncertainty_dict,depth=5):
    # find a policy that multipulation of uncertainty_dict minimized
    raise NotImplementedError

def Heuristic_Search_absolute(Uncertainty_dict,threshold, depth=5):
    # select viewpoint maximize uncertainty and out of a certain range (lower than a threshold)
    raise NotImplementedError

def Heuristic_Search_gradient(Uncertainty_dict,threshold, depth=5):
    # select viewpoint maximize uncertainty and out of a certain range (lower than a threshold)
    raise NotImplementedError

# 找到最后一个视角不确定性最小的n个点的index，并从working_index中删除，如果他们还在working_index中
# 找到最后一个视角，归一化不确定性的10%对应的点，从working_index中删除，如果他们还在working_index中
# 每次选取不确定性中不确定性变化最大的那个点


def policy_1(uncertainty_dict, depth=20, mode='uncertainty'):
    # 直接用不确定性的值，每次讲不同视角的不确定性分布相乘，从中选取最大值作为下一个视角
    # PSNR和SSIM同理，不过每次选取最小值
    working_index = [i for i in range(len(uncertainty_dict.keys()))]
    index_his = [random.randint(0, len(uncertainty_dict.keys()) - 1)]
    current_uncertainty = np.array(uncertainty_dict[index_his[-1]][mode])
    for _ in range(depth):
        working_index.remove(index_his[-1])
        if mode=='uncertainty' or mode=='MSE':
            extreme_index = np.argmax(current_uncertainty[working_index])
        elif mode=='PSNR' or mode=='SSIM':
            extreme_index = np.argmin(current_uncertainty[working_index])
        else:
            raise NotImplementedError
         # uncertainty
        result_index = working_index[extreme_index]

        index_his.append(result_index)
        current_uncertainty = current_uncertainty * np.array(uncertainty_dict[index_his[-1]][mode])

        print(index_his)
    return index_his

def policy_2(uncertainty_dict,depth=20,mode='uncertainty',top_n=3):
    working_index = [i for i in range(len(uncertainty_dict.keys()))]
    index_his = [random.randint(0, len(uncertainty_dict.keys()) - 1)]
    working_index.remove(index_his[-1])

    while len(working_index)>0 and len(index_his)<=depth:
        # 找到最后一个视角不确定性最大的top_n个点的index，并从working_index中删除，如果他们还在working_index中
        current_uncertainty = np.array(uncertainty_dict[index_his[-1]][mode])
        if mode=='uncertainty' or mode=='MSE':
            indices = np.argpartition(current_uncertainty, top_n)[:top_n]
        elif mode=='PSNR' or mode=='SSIM':
            indices = np.argpartition(-current_uncertainty, top_n)[:top_n]
        else:
            raise NotImplementedError
        
        working_index = [x for x in working_index if x not in set(indices)]
        if len(working_index)>0:
            if mode=='uncertainty' or mode=='MSE':
                extreme_index = np.argmax(current_uncertainty[working_index])
            elif mode=='PSNR' or mode=='SSIM':
                extreme_index = np.argmin(current_uncertainty[working_index])
            else:
                raise NotImplementedError
            result_index = working_index[extreme_index]
            index_his.append(result_index)
            working_index.remove(index_his[-1])
    return index_his


def policy_3(uncertainty_dict,depth=20,mode='uncertainty', threshold=0.1):
    working_index = [i for i in range(len(uncertainty_dict.keys()))]
    index_his = [random.randint(0, len(uncertainty_dict.keys()) - 1)]
    working_index.remove(index_his[-1]) 
    while len(working_index)>0 and len(index_his)<=depth:
        # 找到最后一个视角不确定性最大的top_n个点的index，并从working_index中删除，如果他们还在working_index中
        current_uncertainty = np.array(uncertainty_dict[index_his[-1]][mode])
        current_uncertainty = (current_uncertainty-np.min(current_uncertainty))/(np.max(current_uncertainty)-np.min(current_uncertainty))
        if mode=='uncertainty' or mode=='MSE':
            indices = np.where(current_uncertainty <= threshold)[0]
        elif mode=='PSNR' or mode=='SSIM':
            indices = np.where(current_uncertainty >= (1-threshold))[0]
        else:
            raise NotImplementedError
        
        working_index = [x for x in working_index if x not in set(indices)]
        if len(working_index)>0:
            if mode=='uncertainty' or mode=='MSE':
                extreme_index = np.argmax(current_uncertainty[working_index])
            elif mode=='PSNR' or mode=='SSIM':
                extreme_index = np.argmin(current_uncertainty[working_index])
            else: 
                raise NotImplementedError
            result_index = working_index[extreme_index]
            index_his.append(result_index)
            working_index.remove(index_his[-1])
    return index_his

def caculate_neighbor_differences_HEALPix(uncertainty,n_side=2,mode='uncertainty'):
    # 获取像素数量
    num_pixels = hp.nside2npix(n_side)
    uncertainties = np.array(uncertainty[mode])  # 不确定性数组
    assert len(uncertainties) == num_pixels, "不确定性数组长度应与 HEALPix 像素数量一致"

     # 初始化结果数组
    result = np.zeros(num_pixels)

    # 遍历每个像素，计算与其邻居的差值
    for pixel in range(num_pixels):
        # 获取当前像素的所有邻居
        neighbors = hp.get_all_neighbours(n_side, pixel)
        neighbors = neighbors[neighbors >= 0]  # 移除无效邻居（极点可能存在 -1）

        # 计算与邻居的差值
        differences = np.abs(uncertainties[pixel] - uncertainties[neighbors])
        result[pixel] = np.max(differences)  # 记录最大差值

    return result

def caculate_neighbor_differences(uncertainty,mode='uncertainty'):
    # 根据输入网格，计算网格中每个点和周围点的差值中最大/最小的值
    rows=cols=round(math.sqrt(len(uncertainty[mode])))
    grid = np.array(uncertainty[mode]).reshape(rows,cols)

    result = np.zeros_like(grid)
    for i in range(rows):
        for j in range(cols):
            # 获取邻居的坐标（考虑周期性边界）
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # 跳过自身
                    # ni = (i + di) % rows  # 垂直方向周期性
                    ni = (i + di) # 垂直方向不需要周期性
                    if ni<0 or ni>=rows:
                        continue
                    nj = (j + dj) % cols  # 水平方向周期性
                    neighbors.append(grid[ni, nj])
            # 计算差值
            differences = np.abs(grid[i, j] - np.array(neighbors))
            result[i, j] = np.max(differences)
    return result

def policy_4(uncertainty_dict,depth=20,mode='uncertainty'):
    working_index = [i for i in range(len(uncertainty_dict.keys()))]
    index_his = [random.randint(0, len(uncertainty_dict.keys()) - 1)]
    working_index.remove(index_his[-1]) 
    while len(index_his)<=depth:
        difference_result = caculate_neighbor_differences(uncertainty_dict[index_his[-1]],mode)
        max_value = np.max(difference_result[working_index])
        indices = np.where(difference_result[working_index] == max_value)[0]

        related_uncertainty = [uncertainty_dict[index_his[-1]][mode][i] for i in working_index]
        related_uncertainty = [related_uncertainty[i] for i in indices]
        if mode=='uncertainty' or mode=='MSE':
            extreme_index = np.argmax(related_uncertainty)
        elif mode=='PSNR' or mode=='SSIM':
            extreme_index = np.argmin(related_uncertainty)
        else: 
            raise NotImplementedError

        result_index = working_index[indices[extreme_index]]
        index_his.append(result_index)
        working_index.remove(index_his[-1])
    return index_his




if __name__=='__main__':
    # parameter
    num_azimuth = 8
    num_elevation = 8
    radius = [2]
    depth = 20
    distribution_dataset_path = '/home/zhengquan/04-fep-nbv/data/test/distribution_generation_test'
    num_offset_phi = 8

    # env create
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a6ad7a24bb89733f412783097373bdc'
    offset = model_path.split('/')[-2]
    category = offset2word(offset)
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path
    # cfg.train_iter = 10

    # generate candidate viewpoint
    candidate_viewpoint_poses = generate_HEALPix_viewpoints(n_side=2)
    num_candidate_viewpoint = len(candidate_viewpoint_poses)

    # load uncertainty
    dataset_path = model_path.replace('/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2',distribution_dataset_path)
    dataset_path = dataset_path.replace(offset,category)
    uncertainty_path = os.path.join(dataset_path,'uncertainties')
    obs_path = os.path.join(dataset_path,'images')
    assert os.path.exists(uncertainty_path), "no uncertainty file"
    uncertainties = [f for f in os.listdir(uncertainty_path) if f.endswith('json')]
    assert len(uncertainties)==num_candidate_viewpoint*num_offset_phi, "uncertrainty not completely generated"

    uncertainty_dict = {}
    min_uncertainty = 1e10
    for index,f in enumerate(tqdm(uncertainties)):
        uncertainty = load_from_json(Path(os.path.join(uncertainty_path,f)))
        uncertainty_dict[index]=uncertainty

        if min(uncertainty['uncertainty'])<min_uncertainty:
            min_uncertainty = min(uncertainty['uncertainty'])

    for policy in ['policy_2', 'policy_3','policy_4']:
        gif_current_uncertainty = GIFSaver()
        gif_obs = GIFSaver()
        for mode in ['uncertainty','PSNR','SSIM','MSE']:
            print(f'dealing with {policy} {mode}')
            current_uncertainty =  np.ones_like(uncertainty_dict[0]['uncertainty'])
            index_his = eval(policy)(uncertainty_dict=uncertainty_dict,mode=mode)

            # policy eval
            # 输入一个poses，输出基于这个poses得到的PSNR等？
            env = set_env(cfg)
            cfg.exp_name = f'data/test/policy_eval_test/car/{policy}/{mode}'
            tracker = RefMetricTracker(cfg, env=env)
            tracker.setup_writer(f'{cfg.exp_name}')
            NeRF_pipeline = NeRF_init(cfg)
            if os.path.exists(os.path.join(cfg.exp_name,f'uncertainty_{len(index_his)-1}.png')):
                print(f'{policy} {mode} already finished ')
                continue
            os.makedirs(cfg.exp_name,exist_ok=True)
            

            for i,pose_index in enumerate(tqdm(index_his)):
                empty_cache()       
                NeRF_pipeline.reset()
                obs = Image.open(os.path.join(obs_path,f'viewpoint_{pose_index}.png'))
                obs = [np.array(obs)]

                current_uncertainty = current_uncertainty*uncertainty_dict[pose_index][mode]
                gif_current_uncertainty.add(current_uncertainty)
                gif_obs.add(obs[0])

                pose = index2pose_HEALPix(pose_index).unsqueeze(0)
                # pose = polar2pose([a],[e],[r])

                if i==0:
                    tracker.init_trajectory(pose)

                t1 = time.time()
                NeRF_pipeline.add_image(images=obs,poses=pose,model_option=None)
                t2 = time.time()
                # print(f'NeRF trained in {timedelta(seconds=t2-t1)}')
                tracker.update(NeRF_pipeline, pose, i)
                save_img(obs[0],f'{cfg.exp_name}/obs_{i}_{pose_index}.png')
                visualize_HEALPIix_distribution_polar(current_uncertainty,mode,save_path=f'{cfg.exp_name}/current_uncertainty_{i}.png',original_viewpoint=np.array(pose[0,4:]))
                visualize_HEALPIix_distribution_polar(uncertainty_dict[pose_index][mode],mode,save_path=f'{cfg.exp_name}/uncertainty_{i}.png',original_viewpoint=np.array(pose[0,4:]))
                # visualize_distribution_polar(Uncertainty=current_uncertainty,save_path=f'{cfg.exp_name}/current_uncertainty_{i}.png')
                # visualize_distribution_polar(Uncertainty=uncertainty_dict[pose_index][mode],save_path=f'{cfg.exp_name}/uncertainty_{i}.png')

                if i>1:
                    tracker.gif.save(f'{cfg.exp_name}/eval.gif')
                    gif_current_uncertainty.save(f'{cfg.exp_name}/current_uncertainty.gif')
                    gif_obs.save(f'{cfg.exp_name}/obs.gif')
                    save_dict_to_excel(f'{cfg.exp_name}/metrics.xlsx', tracker.metric_hist)

            del env
            del NeRF_pipeline
    
    print('policy tested')