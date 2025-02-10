# 这段代码想实现的时候可视化，从固定视角观察目标，随着agent获得越来越多的观察，每次在新的观察位置上放一个摄像机
import sys
import os
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import mathutils

import numpy as np
import torch
import time
from datetime import timedelta, datetime

from PIL import Image, ImageDraw, ImageFont

from nvf.active_mapping.agents import *
from nvf.env.Enviroment import Enviroment
from nvf.env.utils import get_images, GIFSaver, stack_img, save_img, set_seed
from dataclasses import dataclass, field
from nvf.env.utils import empty_cache
from nvf.log_utils import *
import inspect
from torch.utils.tensorboard import SummaryWriter
from nvf.active_mapping.active_mapping import ActiveMapper
from torch.nn import MSELoss

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
# from torchmetrics import PeakSignalNoiseRatio
# from torchmetrics.functional import structural_similarity_index_measure
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import pickle as pkl
import gtsam
import gym
import os

import nerfstudio
from nvf.metric.MetricTracker import RefMetricTracker
from nvf.uncertainty.entropy_renderers import VisibilityEntropyRenderer, WeightDistributionEntropyRenderer
from nvf.eval_utils import *
from typing import Literal, Optional
import tyro

from config import *

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

def set_nvf_params_local(cfg, pipeline):
    '''
    only used for testing locally, with limited cuda requirements
    '''
    # pipeline.use_visibility = True
    pipeline.trainer.config.nvf_batch_size = 64#65536#*4#*16
    pipeline.trainer.config.nvf_num_iterations = 10
    pipeline.trainer.config.nvf_train_batch_repeat = 1

    pipeline.config.max_num_iterations = 50
    pipeline.trainer.pipeline.model.n_uniform_samples = 10
    # pipeline.trainer.pipeline.config.target_num_samples = 512*64

    # aabb = torch.tensor([[-1., -1, -1], [2., 2., 2.]])
    # breakpoint()
    pass

def set_nvf_params(cfg, pipeline):
    '''
    only used for testing locally, with limited cuda requirements
    '''
    # pipeline.use_visibility = True
    # pipeline.trainer.config.nvf_batch_size = 65536//2
    pipeline.trainer.config.nvf_num_iterations = 200
    # pipeline.trainer.config.nvf_train_batch_repeat = 1

    # pipeline.config.max_num_iterations = 100
    # pipeline.trainer.pipeline.model.n_uniform_samples = 10
    # pipeline.trainer.pipeline.config.target_num_samples = 512*64

    # aabb = torch.tensor([[-1., -1, -1], [2., 2., 2.]])
    # breakpoint()
    pass

def set_aabb(pipeline, aabb):
    pipeline.trainer.pipeline.datamanager.train_dataset.scene_box.aabb[...] = aabb
    pipeline.trainer.pipeline.model.field.aabb[...] = aabb

def set_agent(cfg):
    agent = eval(cfg.agent)(cfg)
    return agent

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

def update_params(cfg, agent, step):
    if step == cfg.horizon - 1:
        print('reset last step')
        agent.pipeline.config.max_num_iterations = int(cfg.train_iter * cfg.train_iter_last_factor)#*4 #20000
    else:
        agent.pipeline.config.max_num_iterations = cfg.train_iter

    set_params(cfg, agent.pipeline)
    agent.pipeline.trainer.pipeline.model.renderer_entropy.set_iteration(step)

def evaluate(cfg, run_number=0):
    t1 = time.time()
    env = set_env(cfg)

    agent = set_agent(cfg)

    if cfg.agent == "VisibilityAgent":
        agent.init_visibility(env)

    # set_params(cfg, agent.pipeline)

    tracker = RefMetricTracker(cfg, env=env)
    tracker.setup_writer(f'{cfg.exp_name}/agent/run-{run_number}')

    save_path = f'{cfg.exp_folder}/run-{run_number}'
    os.makedirs(save_path, exist_ok=True)


    # input(action)

    
    # print(action)

    gif = GIFSaver()
    for i in range(cfg.horizon):
        agent.pipeline.reset()
        update_params(cfg, agent, i)
        agent.pipeline.trainer.reset_writer(f'{cfg.exp_name}/train/run-{run_number}/step-{i}', cfg.train_use_tensorboard)

        # print(i)
        if i==0:
            obs = env.obs_history # 3 512 512 4
            action = env.pose_history
            tracker.init_trajectory(env.pose_history)
        # breakpoint()
        action = agent.act(obs, action)

        agent.pipeline.save_ckpt(f'{save_path}/ckpt/step-{i}.ckpt')
        empty_cache()
        tracker.update(agent.pipeline, action, i)
        print(f'Select pos at {i}', action[0][-3:].numpy())
        empty_cache()
        obs, _, _, _ = env.step(action) # 
        save_img(obs[0], f'{save_path}/obs/step-{i}.png')
        # print(type(obs[0]), obs[0].shape)
        gif.add(obs[0], fn=lambda img: ImageDraw.Draw(img).text((3, 3), f"T={i}", fill=(255, 255, 255)))
        
        # gif.save('results/test2_agent.gif')
        # tracker.gif.save('results/test2_eval.gif')
        # gif.save('results/test_sim1.gif')
        

    # print(action)
    # breakpoint()
    gif.save(f'{save_path}/agent.gif')
    tracker.gif.save(f'{save_path}/eval.gif')
    # tracker.save_metric(f'{save_path}/results.pkl')
    result = save_result(f'{save_path}/results.pkl', tracker, agent)
    # save agent.pose_his as a txt file in save_path pose_his_{run_number}
    np.savetxt(f'{save_path}/pose_his_{run_number}.txt', np.asarray(agent.pose_hist))

    # breakpoint()
    save_dict_to_excel(f'{save_path}/metrics.xlsx', tracker.metric_hist)
    # for k,v in tracker.metric_hist.items():
    #     print(k,v)
    
    # breakpoint()
    t2 = time.time()
    print(f'Run {run_number} done in {timedelta(seconds=t2-t1)}')
    return result

def multi_evaluation(cfg: ExpConfig):

    now = datetime.now()

    if cfg.name is None:
        cfg.name = now.strftime(r'%m%d_%H%M%S')
    pd_path = f'{cfg.exp_folder}/results.xlsx'
    cfg.exp_name = f'{cfg.scene.name}_{cfg.method.name}_{cfg.model.name}_{cfg.agent.name}_{cfg.sampler.name}_{cfg.name}'
    cfg.exp_folder =f'{cfg.exp_folder}/{cfg.exp_name}'
    os.makedirs(cfg.exp_folder, exist_ok=True) # './results/hubble_nvf_ngp_base_base_0830_155359'

    # print('Exp Name: ', cfg.exp_name)

    all_results = []

    
    # core loop for 3 runs 
    for i in range(cfg.n_repeat):
        set_seed(100+i)
        print('Exp Name: ', cfg.exp_name)
        print(f'Run {i}')
        empty_cache() # clear cache 
        if i>=1: cfg.env.gen_init = False; cfg.env.gen_eval = False
        result = evaluate(cfg, i)
        all_results.append(result)

    # def record_idx(k):
    #     if k == 'corr':
    #         return 0
    #     else:
    #         return -1
    def record_value(hist,k):
        if k == 'corr':
            return hist[0]
        elif k.endswith('_time'):
            return hist[:-1]
        else:
            return hist[-1]


    metric_keys = all_results[-1]['metric'].keys()
    metric = {k:np.mean([record_value(d['metric'][k],k) for d in all_results]) for k in metric_keys}
    metric_std = {k:np.std([record_value(d['metric'][k], k) for d in all_results]) for k in metric_keys}
    
    metric_hist = {k:np.array([ d['metric'][k] for d in all_results ]) for k in metric_keys}
    # metric_hist_std = {k:np.std([d['metric'][k] for d in all_results], axis=0) for k in metric_keys}
    
    record_keys = ['scene','method','model', 'agent', 'sampler', 'name','n_repeat']

    # data_per_run_list = []
    # for i,d in enumerate(all_results):
    #     data_per_run = {k:cfg.__dict__[k] for k in record_keys[:-1]}
    #     data_per_run['run_id'] = i
    #     data_per_run.update({k: d['metric'][k][-1] for k in metric_keys})
    #     data_per_run_list.append(data_per_run)
    writer_log_path = os.path.join(RefMetricTracker.base_log_dir,cfg.exp_name,'agent', 'avg')
    avg_writer = SummaryWriter(log_dir=writer_log_path)

    for k in metric_keys:
        mean_m = metric_hist[k].mean(axis=0)
        for step in range(mean_m.shape[0]):
            avg_writer.add_scalar(f'eval/{k}', mean_m[step], step)
    

    data = {
        # 'results':all_results,
        'time': now.strftime("%Y-%m-%d %H:%M"),
        'metric':metric,
        'metric_std': metric_std,
        'metric_hist': metric_hist,
        # 'metric_hist_std': metric_hist_std,
    }

    print('\n')
    print(cfg.exp_name,'Final results:')
    print('metric_std',metric_std)
    print('metric',metric)


    save_cfg(f'{cfg.exp_folder}/cfg.yaml', cfg)
    import nvf.active_mapping.agents.Sampler as Sampler
    import socket
    modules = [multi_evaluation, VisibilityEntropyRenderer, 
               WeightDistributionEntropyRenderer, ActiveMapper, BaseAgent, Sampler]
    save_codes(f'{cfg.exp_folder}/source.log', *modules)

    with open(f'{cfg.exp_folder}/summary.pkl', 'wb') as f:
        pkl.dump(data, f)

    summarized_metric ={k: np.mean([d['metric'][k] for d in all_results], axis=0) for k in metric_keys}
    summarized_metric.update({k+'_std': np.std([d['metric'][k] for d in all_results], axis=0) for k in metric_keys})

    save_dict_to_excel(f'{cfg.exp_folder}/summary.xlsx', summarized_metric)
    
    record_data = {'time': now.strftime("%Y-%m-%d %H:%M")}
    
    def get_config_record_value(x):
        if type(x) is SceneType:
            return x.name 
        return x
    record_data.update({k:get_config_record_value(cfg.__dict__[k]) for k in record_keys})
    record_data.update(metric)
    record_data.update({k+'_std':v for k,v in metric_std.items()})
    
    record_data.update({'server': socket.gethostname()})

    try:
        save_result_excel(record_data, pd_path, sheet_name=0, lock=True)
        # save_result_excel(data_per_run_list, pd_path, sheet_name=1, lock=True)
    except:
        raise

def visualize_path(cfg,log_path):
    run_number = int(log_path.split('-')[-1])
    pose_hist = np.loadtxt(f'{log_path}/pose_his_{run_number}.txt')

    env = set_env(cfg)
    env.scene.set_white_background()

    # 计算相机位置，目标位置和视角方向的四元数
    camera_position = mathutils.Vector(cfg.camera_aabb[0])
    target_position = mathutils.Vector(cfg.camera_aabb[1])
    direction = target_position - camera_position 
    rot_quat = torch.tensor(direction.to_track_quat('-Z', 'Y'))

    # 渲染固定斜上方视角下的目标图像并保存
    fixed_pose = torch.concat((rot_quat, torch.tensor(camera_position)))
    image = env.scene.render_pose(fixed_pose)
    # 图像存在log_path的views目录下，需要创建views并保存
    save_img(image, f'{log_path}/views/fixed_pose.png')

    gif = GIFSaver()
    # 对pose_hist中的每个视角，在scene中添加圆锥并渲染图像然后保存
    for i,pose in enumerate(pose_hist):
        pose = torch.tensor(pose)
        env.scene.add_cone_at_pose(pose[-3:],pose[[3,0,1,2]],id=i)
        image = env.scene.render_pose(fixed_pose)
        # 存在log_path的views目录下，需要创建views并保存
        save_img(image, f'{log_path}/views/pose_{i}.png')

        # 如果i>=2,那么从obs路径下中读取step-{i-2}的图像，和image拼在一起形成一张图
        if i>=3:
            obs = Image.open(f'{log_path}/obs/step-{i-3}.png')
            img = Image.new('RGB', (obs.width*2, obs.height))
            img.paste(obs, (0, 0))
            if image.max()>1:
                img.paste(Image.fromarray(np.uint8(image)), (obs.width, 0))
            else:
                img.paste(Image.fromarray(np.uint8(image*255)), (obs.width, 0))
            gif.add(np.asarray(img))

    # 在log_path下把gif保存views.gif
    gif.save(f'{log_path}/views.gif')

# 可视化训练集图像越来越多的过程
def visual_trainset(log_path):
    # 读取log_path下的obs目录下的所有图像
    # 生成一个gif，每一帧是一个5x4的图像，随着帧数的增加，obs0出现在第一个位置，obs1出现在第二个位置……
    # 保存在log_path下的trainset.gif
    gif = GIFSaver()
    obs = Image.open(f'{log_path}/obs/step-0.png')
    img = Image.new('RGB', (obs.width*5, obs.height*4))
    for i in range(20):
        obs = Image.open(f'{log_path}/obs/step-{i}.png')
        # 计算20张图像应该在gif图像的那个位置，obs0是0，0，obs1是1，0，obs2是2，0……
        img.paste(obs, (i%5*obs.width, i//5*obs.height))
        gif.add(np.asarray(img))
        # 在train_set目录下单独保存每一个img
        img.save(f'{log_path}/obs/trainset-{i}.png')
    gif.save(f'{log_path}/trainset.gif')
                             
def extract_string(path):
    # 使用正则表达式提取 {} 位置的字符串
    match = re.search(r'results_vis_1112/(.*?)_.*', path)
    if match:
        return match.group(1)
    return None

def visualize_circle(cfg,log_path):
    num_cameras = 100
    radius = 2
    theta = np.linspace(0, 2*np.pi, num_cameras)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(num_cameras)

    # x = radius * np.cos(theta)
    # y = np.zeros(num_cameras)
    # z = radius * np.sin(theta)

    cameras = np.stack([x, y, z], axis=1)
    cameras = torch.tensor(cameras, dtype=torch.float32).to('cuda')
    cameras = cameras.unsqueeze(1)
    # 生成相机的方向，从相机位置指向原点
    directions = -cameras
    # directions = directions / torch.norm(directions, dim=2, keepdim=True)
    directions = directions.squeeze(1)
    directions = [mathutils.Vector(row) for row in directions.cpu().numpy()]
    for direction in directions:
        direction.normalize()
    rot_quat = torch.tensor([direction.to_track_quat('-X', 'Z') for direction in directions]).to('cuda')
    pose_hist = torch.concat((rot_quat, torch.tensor(cameras).squeeze()),dim=1)

    env = set_env(cfg)
    env.scene.set_white_background()

    # 计算相机位置，目标位置和视角方向的四元数
    camera_position = mathutils.Vector(cfg.camera_aabb[0])
    target_position = mathutils.Vector(cfg.camera_aabb[1])
    direction = target_position - camera_position 
    rot_quat = torch.tensor(direction.to_track_quat('-Z', 'Y'))

    # 渲染固定斜上方视角下的目标图像并保存
    fixed_pose = torch.concat((rot_quat, torch.tensor(camera_position)))
    image = env.scene.render_pose(fixed_pose)
    # 图像存在log_path的views目录下，需要创建views并保存
    save_img(image, f'{log_path}/views/fixed_pose.png')

    gif = GIFSaver()
    # 对pose_hist中的每个视角，在scene中添加圆锥并渲染图像然后保存
    for i,pose in enumerate(pose_hist):
        pose = torch.tensor(pose)
        env.scene.add_cone_at_pose(pose[-3:],pose[:4],id=i)
        image = env.scene.render_pose(fixed_pose)
        # 存在log_path的views目录下，需要创建views并保存
        save_img(image, f'{log_path}/views/pose_{i}.png')
        gif.add(np.asarray(image))

    # 在log_path下把gif保存views.gif
    gif.save(f'{log_path}/views.gif')


if __name__ == "__main__":
    # log_path = "results_vis_1112/hubble_nvf_ngp_dist_base_1112_155259/run-0" # for circle visual
    log_path = "results_xu_1210/hubble_nvf_ngp_dist_base_1215_162004/run-0"
    scene_name = 'hubble'#extract_string(log_path) 
    sys.argv = ["visualize_path.py", "--scene", scene_name]


    cfg = tyro.cli(ExpConfig) # config
    cfg.env.scale = 0.3333 * 0.5 / 6
    visualize_path(cfg,log_path)
    # visual_trainset(log_path)
    # visualize_circle(cfg,log_path)
