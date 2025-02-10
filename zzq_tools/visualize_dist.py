# 可视化加入移动代价后算法获得的精度变化
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def compare(base_metric_path,euclidean_metric_path,manhattan_metric_path):
    # 从每个path下面的run-0读取mētrics.xlsx和pose_his_0.txt
    results = {}
    for path in [base_metric_path,euclidean_metric_path,manhattan_metric_path]:
        # 初始化
        results[path] = {}
        # 读取path/matrics.xlsx文件
        df = pd.read_excel(path+'/run-0/metrics.xlsx')
        psnr_data = df['psnr']
        ssim_data = df['ssim']
        results[path]['psnr'] = np.asarray(psnr_data)
        results[path]['ssim'] = np.asarray(ssim_data)

        pose_his = np.loadtxt(path+'/run-0/pose_his_0.txt')
        # 计算移动代价 使用
        move_dist = [0]
        for i in range(4,pose_his.shape[0]):
            # 使用曼哈顿距离
            move_dist.append(np.sum(np.abs(pose_his[i,-3:]-pose_his[i-1,-3:]))+move_dist[-1])
            # 使用欧式距离计算移动代价
            # move_dist.append(np.linalg.norm(pose_his[i,-3:]-pose_his[i-1,-3:])+move_dist[-1])
        results[path]['move_dist'] = np.asarray(move_dist)
    
    # 绘制三张对比图，对比不同path下的psnr、ssim和move_dist
    plt.figure(figsize=(18, 6))
    # 绘制PSNR图
    plt.subplot(1, 3, 1)
    plt.plot(results[base_metric_path]['psnr'], marker='o',label='base')
    plt.plot(results[euclidean_metric_path]['psnr'], marker='o',label='euclidean')
    plt.plot(results[manhattan_metric_path]['psnr'], marker='o',label='manhattan')
    plt.title('PSNR over Time')
    plt.xlabel('Index')
    plt.ylabel('PSNR')
    plt.legend()
    # 绘制SSIM图
    plt.subplot(1, 3, 2)
    plt.plot(results[base_metric_path]['ssim'], marker='o',label='base')
    plt.plot(results[euclidean_metric_path]['ssim'], marker='o',label='euclidean')
    plt.plot(results[manhattan_metric_path]['ssim'], marker='o',label='manhattan')
    plt.title('SSIM over Time')
    plt.xlabel('Index')
    plt.ylabel('SSIM')
    plt.legend()
    # 绘制move_dist图
    plt.subplot(1, 3, 3)
    plt.plot(results[base_metric_path]['move_dist'], marker='o',label='base')
    plt.plot(results[euclidean_metric_path]['move_dist'], marker='o',label='euclidean')
    plt.plot(results[manhattan_metric_path]['move_dist'], marker='o',label='manhattan')
    plt.title('Move Distance (manhattan) over Time')
    plt.xlabel('Index')
    plt.ylabel('Move Distance')
    plt.legend()
    output_dir = 'zzq_tools'
    output_path = os.path.join(output_dir, 'compare_plot.png')
    plt.tight_layout()
    plt.savefig(output_path)

if __name__=='__main__':
    base_metric_path = 'results/hubble_nvf_ngp_base_base_1006_115652'
    euclidean_metric_path = 'results/hubble_nvf_ngp_base_base_1005_190507'
    manhattan_metric_path = 'results/hubble_nvf_ngp_dist_base_1007_105725'

    compare(base_metric_path,euclidean_metric_path,manhattan_metric_path)