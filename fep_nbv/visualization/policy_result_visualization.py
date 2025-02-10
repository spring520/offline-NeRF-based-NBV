# 写一个函数，输入xlsx文件的路径列表，和想要可视化的东西（PSNR、SSIM啥的）
# 画出图来 神奇吧
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/zhengquan/04-fep-nbv")


if __name__=='__main__':
    result_paths = [
        # 'data/test/policy_eval_test/plane/policy_1/MSE/metrics.xlsx',
        'data/test/policy_eval_test/plane/policy_1/PSNR/metrics.xlsx',
        # 'data/test/policy_eval_test/plane/policy_1/SSIM/metrics.xlsx',
        # 'data/test/policy_eval_test/plane/policy_1/uncertainty/metrics.xlsx',
        '/home/zhengquan/04-fep-nbv/data/test/eval_test/shapenet_nvf_ngp_dist_base_0205_205545/run-0/metrics.xlsx',
        # '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_3/uncertainty/metrics.xlsx',
        '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_3/MSE/metrics.xlsx',
        # '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_3/SSIM/metrics.xlsx',
        # '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_3/PSNR/metrics.xlsx',
        # '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_2/uncertainty/metrics.xlsx',
        '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_2/MSE/metrics.xlsx',
        # '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_2/SSIM/metrics.xlsx',
        # '/home/zhengquan/04-fep-nbv/data/test/policy_eval_test/plane/policy_2/PSNR/metrics.xlsx',
    ]
    keys = ['psnr', 'ssim']

    results = {}
    for result_path in result_paths:
        df = pd.read_excel(result_path)
        results[result_path] = {}
        psnr_data = df['psnr']
        ssim_data = df['ssim']
        results[result_path]['psnr'] = np.asarray(psnr_data)
        results[result_path]['ssim'] = np.asarray(ssim_data)


    for key in keys:
        plt.figure()
        for path in result_paths:
            plt.plot(results[path][key], marker='o',label=path.split('/')[-2]+path.split('/')[-3])
            plt.title(f'{key} over step')
            plt.xlabel('Step Index')
            plt.ylabel(key)
            plt.legend()
            plt.savefig(f'/home/zhengquan/04-fep-nbv/data/test/visualizaton/{key}.png') 