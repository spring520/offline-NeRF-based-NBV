# 用来可视化欧拉角在万向节锁附近的不连续性
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 定义一组欧拉角（ZYX 旋转顺序）
angles = np.linspace(85, 95, 100)  # Y轴旋转从 85° 到 95°
euler_angles = np.zeros((100, 3))  # 初始化欧拉角矩阵
euler_angles[:, 1] = angles  # 设置 Y轴旋转

# 转换为旋转矩阵
rotations = R.from_euler('ZYX', euler_angles, degrees=True)

# 转换回欧拉角，观察不连续性
new_euler_angles = rotations.as_euler('ZYX', degrees=True)

# 绘制欧拉角的变化
plt.plot(angles, new_euler_angles[:, 0], label="X angle (phi)")
plt.plot(angles, new_euler_angles[:, 1], label="Y angle (theta)")
plt.plot(angles, new_euler_angles[:, 2], label="Z angle (psi)")
plt.xlabel("Input Y angle")
plt.ylabel("Euler angles")
plt.legend()
plt.title("Euler Angle Discontinuity")
plt.grid()
plt.savefig('zzq_tools/eular_vis.png')