import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# 输入采样点和对应的值
# 示例：球面上的点
np.random.seed(42)
num_points = 100
theta = np.random.uniform(0, 2 * np.pi, num_points)  # 方位角
phi = np.random.uniform(0, np.pi, num_points)        # 极角
radius = 2  # 球半径

# 球面坐标转换为笛卡尔坐标
x = radius * np.sin(phi) * np.cos(theta)
y = radius * np.sin(phi) * np.sin(theta)
z = radius * np.cos(phi)

# 对应的值（可以替换为实际数据）
values = np.sin(phi) * np.cos(theta)  # 示例值

# 创建球面网格
phi_grid, theta_grid = np.mgrid[0:np.pi:200j, 0:2*np.pi:200j]
x_grid = radius * np.sin(phi_grid) * np.cos(theta_grid)
y_grid = radius * np.sin(phi_grid) * np.sin(theta_grid)
z_grid = radius * np.cos(phi_grid)

# 插值到球面网格
points = np.array([x, y, z]).T  # 原始点坐标
grid_points = np.array([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T  # 网格点
grid_values = griddata(points, values, grid_points, method='linear', fill_value=0)
grid_values = grid_values.reshape(x_grid.shape)

# 绘制球面热力图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制球面，颜色表示值
surf = ax.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.viridis(grid_values), rstride=1, cstride=1, alpha=0.9)

# 添加颜色条
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(values)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Value')

# 设置标题和显示
ax.set_box_aspect([1, 1, 1])  # 等比例缩放
ax.set_title("3D Spherical Heatmap")
plt.savefig('zzq_tools/vis_3d_distribution_ball.png')