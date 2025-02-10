# 可视化加入移动代价后算法获得的精度变化
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import torch
import collections
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AgentType, SceneType, MethodType, ModelType, SamplerType


def find_folders_with_target(file_path,mark_file_name='summary.pkl'):
    folders_with_summary = []

    # 遍历目录结构
    for root, dirs, files in os.walk(file_path):
        if mark_file_name in files:
            folders_with_summary.append(root)

    return folders_with_summary

def compare(results):
    sorted_keys = sorted(results.keys())
    excluded_list = ['0.05',"0.1"]

    # 绘制三张对比图，对比不同path下的psnr、ssim和move_dist
    plt.figure(figsize=(18, 6))
    # 绘制PSNR图
    plt.subplot(1, 3, 1)
    for dist in sorted_keys:
        if dist in excluded_list:
            continue
        plt.plot(results[dist]['psnr'], marker='o',label=dist)
    plt.title('PSNR over Time')
    plt.xlabel('Index')
    plt.ylabel('PSNR')
    plt.legend()
    # 绘制SSIM图
    plt.subplot(1, 3, 2)
    for dist in sorted_keys:
        if dist in excluded_list:
            continue
        plt.plot(results[dist]['ssim'], marker='o',label=dist)
    plt.title('SSIM over Time')
    plt.xlabel('Index')
    plt.ylabel('SSIM')
    plt.legend()
    # 绘制move_dist图
    plt.subplot(1, 3, 3)
    for dist in sorted_keys:
        if dist in excluded_list:
            continue
        plt.plot(results[dist]['move_dist'], marker='o',label=dist)
    plt.title('Move Distance over Time')
    plt.xlabel('Index')
    plt.ylabel('Move Distance')
    plt.legend()

    output_dir = 'zzq_tools'
    output_path = os.path.join(output_dir, 'compare_plot_weight_dist.png')
    plt.tight_layout()
    plt.savefig(output_path)


if __name__=='__main__':
    file_path = "results_weight_dist"
    runs = ['run-0','run-1','run-2']

    # 找到file_path下有summary.pkl这个子文件的所有文件夹路径
    folders = find_folders_with_target(file_path)


    results = {}

    # 对folders里面的每个文件夹，把文件夹下的summary.xlsx文件读取出来
    for folder in folders:
        with open(os.path.join(folder, 'cfg.yaml'), 'r') as yaml_file:
            for line in yaml_file:
                line = line.strip()
                if line.startswith('weight_dist'):
                    weight_dist = float(line.split(': ')[1])
                else:
                    continue
        results[str(weight_dist)] = {}

        df = pd.read_excel(folder+'/summary.xlsx')
        psnr_data = df['psnr']
        ssim_data = df['ssim']
        results[str(weight_dist)]['psnr'] = np.asarray(psnr_data)
        results[str(weight_dist)]['ssim'] = np.asarray(ssim_data)

        # 读取run-0 run-1 run-2下面的move dist
        move_dist = []
        for run in runs:
            move_dist.append(0)
            pose_his = np.loadtxt(f'{folder}/{run}/pose_his_{run[-1]}.txt')
            for i in range(4,pose_his.shape[0]):
                move_dist.append(np.linalg.norm(pose_his[i,-3:]-pose_his[i-1,-3:])+move_dist[-1])
            
        move_dist = np.asarray(move_dist).reshape((3,-1)).mean(axis=0)
        
        results[str(weight_dist)]['move_dist'] = np.asarray(move_dist)

    compare(results)



    