# 可视化球面插值

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 示例数据生成
def generate_spherical_points(n_points, offset_theta=0, offset_phi=0):
    """
    生成球面上的点并转为笛卡尔坐标。
    :param n_points: 球面点数量
    :param offset_theta: θ 偏移
    :param offset_phi: φ 偏移
    :return: 笛卡尔坐标 (N, 3)，球面坐标 (theta, phi)
    """
    theta = np.linspace(0, np.pi, n_points) + offset_theta  # 球面 θ
    phi = np.linspace(0, 2 * np.pi, n_points) + offset_phi  # 球面 φ
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    return points, theta.ravel(), phi.ravel()

# 已知点
n_points = 20
known_points, known_theta, known_phi = generate_spherical_points(n_points)
known_values = np.sin(known_theta)  # 示例已知值（可根据需求调整）

# 目标点
target_points, target_theta, target_phi = generate_spherical_points(n_points, offset_theta=0.1, offset_phi=0.1)

# 插值
interpolated_values = griddata(
    known_points, known_values, target_points, method='linear', fill_value=0
)

# 可视化
fig = plt.figure(figsize=(12, 6))

# 球面点分布
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(known_points[:, 0], known_points[:, 1], known_points[:, 2], c=known_values, cmap='viridis', label='Known Points')
ax1.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='red', alpha=0.6, label='Target Points')
ax1.set_title("3D Sphere with Known and Target Points")
ax1.legend()

# 插值结果
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c=interpolated_values, cmap='coolwarm', label='Interpolated Values')
ax2.set_title("Interpolated Values on Target Points")
ax2.legend()

plt.tight_layout()
plt.savefig('/home/zhengquan/04-fep-nbv/data/test/visualizaton/interpolate_sphere.png')