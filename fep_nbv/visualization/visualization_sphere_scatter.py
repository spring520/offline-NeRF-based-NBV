
# 一个用来可视化，视角转换问题的代码
import sys
sys.path.append("/home/zhengquan/04-fep-nbv")
import matplotlib.pyplot as plt
import numpy as np
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints
from fep_nbv.env.utils import offset2word,save_img,empty_cache,GIFSaver

def plot_spherical_points_with_values(x, y, z, values=(0,1,0),save_path='/home/zhengquan/04-fep-nbv/data/test/test.png'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 使用颜色映射表示数值大小
    scatter = ax.scatter(x, y, z, c=values, cmap='viridis', s=50, alpha=0.7)
    scatter = ax.scatter(x[:4], y[:4], z[:4], c=(0,0,1), cmap='viridis', s=50, alpha=0.5)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Value')

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置球面显示效果
    # ax.set_box_aspect([1, 1, 1])
    plt.title('Scatter Points on a Sphere with Values')
    plt.savefig(save_path)



if __name__=='__main__':
    candidate_poses = generate_HEALPix_viewpoints(n_side=1)
    relative_poses_0 = generate_HEALPix_viewpoints(n_side=1)
    relative_poses_1 = generate_HEALPix_viewpoints(n_side=1,original_viewpoint=candidate_poses[1,4:])
    relative_poses_2 = generate_HEALPix_viewpoints(n_side=1,original_viewpoint=candidate_poses[2,4:])

    values = relative_poses_1[:,6]


    gif = GIFSaver()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    scatter = ax.scatter(0, 0, 0, c=(0,0,0), cmap='viridis', s=150, alpha=0.7)

    pose = candidate_poses[6]
    scatter = ax.scatter(pose[4], pose[5], pose[6], c=(0,0,0), cmap='viridis', s=50, alpha=0.7)

    cos_angles = np.dot(relative_poses_1[:,4:]/2, pose[4:]/2)
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    # 筛选距离小于阈值的采样点
    angle_threshold = np.radians(45)  # 10 度阈值
    nearby_indices = np.where(angles < angle_threshold)[0]
    far_indices = np.where(angles > angle_threshold)[0]

    for indice in nearby_indices:
        scatter = ax.scatter(relative_poses_1[indice,4], relative_poses_1[indice,5], relative_poses_1[indice,6], c=(1,0,0), cmap='viridis', s=50, alpha=0.7)
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif.add(image_array)

    for indice in far_indices:
        scatter = ax.scatter(relative_poses_1[indice,4], relative_poses_1[indice,5], relative_poses_1[indice,6], c=(0,0,1), cmap='viridis', s=50, alpha=0.7)
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif.add(image_array)
    save_img(image_array,'/home/zhengquan/04-fep-nbv/data/test/test.png')

    # for pose in relative_poses_0:
    #     scatter = ax.scatter(pose[4], pose[5], pose[6], c=(1,0,0), cmap='viridis', s=50, alpha=0.7)
    #     fig.canvas.draw()
    #     image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     gif.add(image_array)

    # for pose in relative_poses_1:
    #     scatter = ax.scatter(pose[4], pose[5], pose[6], c=pose[6], cmap='viridis', s=50, alpha=0.7)
    #     fig.canvas.draw()
    #     image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     gif.add(image_array)
    # save_img(image_array,'/home/zhengquan/04-fep-nbv/data/test/test.png')

    # for pose in relative_poses_2:
    #     scatter = ax.scatter(pose[4], pose[5], pose[6], c=(0,0,1), cmap='viridis', s=50, alpha=0.7)
    #     fig.canvas.draw()
    #     image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     gif.add(image_array)

    gif.save('/home/zhengquan/04-fep-nbv/data/test/test.gif')

