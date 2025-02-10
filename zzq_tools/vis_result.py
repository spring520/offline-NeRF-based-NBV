
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def load_all_result_single_eval(eval_path):
    results = {}
    runs = [f.path for f in os.scandir(eval_path) if f.is_dir() and f.name.startswith('run')]

    # 如果路径下没有cfg.yaml文件，直接返回
    if not os.path.exists(os.path.join(eval_path, 'cfg.yaml')):
        return None
    
    # weight dist
    with open(os.path.join(eval_path, 'cfg.yaml'), 'r') as yaml_file:
        for line in yaml_file:
            line = line.strip()
            if line.startswith('weight_dist'):
                weight_dist = float(line.split(': ')[1])
            else:
                continue
    results['weight_dist'] = weight_dist

    # PSNR and SSIM
    df = pd.read_excel(eval_path+'/summary.xlsx')
    psnr_data = df['psnr']
    ssim_data = df['ssim']
    results['psnr'] = np.asarray(psnr_data)
    results['ssim'] = np.asarray(ssim_data)

    for run in runs:
        run_name = run.split('/')[-1]
        results[run_name] = {}

        move_dist = [0]
        pose_his = np.loadtxt(f'{eval_path}/{run_name}/pose_his_{run[-1]}.txt')
        for i in range(4,pose_his.shape[0]):
            move_dist.append(np.linalg.norm(pose_his[i,-3:]-pose_his[i-1,-3:])+move_dist[-1])
        results[run_name]['move_dist'] = np.asarray(move_dist)

        entropy = np.loadtxt(f'{eval_path}/{run_name}/reconstruction_uncertainty.txt')
        results[run_name]['entropy'] = np.asarray(entropy)
    
    # 对所有runs的entropy和move_dist求平均
    results['move_dist'] = np.zeros_like(results[run_name]['move_dist'])
    results['entropy'] = np.zeros_like(results[run_name]['entropy'])
    for run in runs:
        run_name = run.split('/')[-1]
        results['move_dist'] += results[run_name]['move_dist']
        results['entropy'] += results[run_name]['entropy']
    results['move_dist'] /= len(runs)
    results['entropy'] /= len(runs)
    results['entropy0'] = results['entropy'][:,0]
    results['entropy1'] = results['entropy'][:,1]
    try:
        results['entropy2'] = results['entropy'][:,2]
        results['entropy3'] = results['entropy'][:,3]
    except IndexError:
        pass


    return results
        

def load_all_results(path):
    all_results = {}

    sub_folders = [f.path for f in os.scandir(path) if f.is_dir()]

    for folder in sub_folders:
        results = load_all_result_single_eval(folder)
        if results == None:
            continue
        else:
            all_results[folder] = results

    return all_results

def visualize_average_over_step(all_results, keys):
    for path in all_results.keys():
        if not os.path.exists(f'{path}/visualization'):
            os.makedirs(f'{path}/visualization')
        for key in keys:
            if key not in all_results[path].keys():
                continue
            plt.figure()
            plt.plot(all_results[path][key], marker='o')
            plt.title(f'{key} over step')
            plt.xlabel('Step Index')
            plt.ylabel(key)
            plt.legend()
            plt.savefig(f'{path}/visualization/average_{key}.png') 


def visualize_a_over_b(all_results,a,b):
    for path in all_results.keys():
        if not os.path.exists(f'{path}/visualization'):
            os.makedirs(f'{path}/visualization')
        if a not in all_results[path].keys() or b not in all_results[path].keys():
            continue
        plt.figure()
        plt.plot(all_results[path][b], all_results[path][a], marker='o')
        plt.title(f'{a} over {b}')
        plt.xlabel(b)
        plt.ylabel(a)
        plt.legend()
        plt.savefig(f'{path}/visualization/{a}_over_{b}.png') 

def visualize_linear_entropy_dist(all_results,path=None,step_length=-1):
    if path==None:
        path = all_results.keys()
    else:
        path = [path]

    for p in path:
        print(f'working on {p}')
        entropy = []
        move_distance = []
        correlation_coefficient_list = []
        slope_list = []
        R2_list = []
        runs = [os.path.basename(f.path) for f in os.scandir(p) if f.is_dir() and f.name.startswith('run')]
        for run in runs:
            entropy.append(all_results[p][run]['entropy'][:step_length,3])
            move_distance.append(all_results[p][run]['move_dist'][:step_length])
            correlation_coefficient, p_value = pearsonr(entropy[-1], move_distance[-1])
            correlation_coefficient_list.append(correlation_coefficient)
            model = LinearRegression()
            model.fit(entropy[-1][:,None], move_distance[-1][:,None])
            slope_list.append(model.coef_)
            R2_list.append(model.score(entropy[-1][:,None], move_distance[-1][:,None]))


        # entropy = np.asarray(entropy).reshape(-1)
        # move_distance = np.asarray(move_distance).reshape(-1)

        # # 计算皮尔逊相关系数
        # correlation_coefficient, p_value = pearsonr(entropy, move_distance)
        # print(f"皮尔逊相关系数: {correlation_coefficient:.2f}")
        # print(f"p值: {p_value:.4f}")

        # # 线性回归分析
        # entropy = entropy.reshape(-1, 1)  # 将 x 变为二维数组以符合 scikit-learn 的输入格式
        # model = LinearRegression()
        # model.fit(entropy, move_distance)

        # # 获取回归系数和截距
        # slope = model.coef_[0]
        # intercept = model.intercept_
        # print(f"线性回归方程: y = {slope:.2f}x + {intercept:.2f}")
        # print(f"R^2 决定系数: {model.score(entropy, move_distance):.2f}")

        print(f'平均皮尔逊相关系数:{sum(correlation_coefficient_list)/len(correlation_coefficient_list)}')
        print(f'平均R^2 决定系数:{sum(R2_list)/len(R2_list)}')
        print(f'平均斜率:{sum(slope_list)/len(slope_list)}')


if __name__=='__main__':
    path = "results_xu_1210"
    all_results = load_all_results(path)

    keys = ['psnr', 'ssim', 'move_dist', 'entropy0','entropy1','entropy2','entropy3']
    visualize_average_over_step(all_results, keys)

    # visualize_a_over_b(all_results, 'entropy3', 'move_dist')

    visualize_linear_entropy_dist(all_results,path='results_xu_1210/hubble_nvf_ngp_dist_base_1215_162004',step_length=20)





