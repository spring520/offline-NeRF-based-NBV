# 画colorbar，拼图用
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# 定义 colorbar 的范围和 colormap
values = np.linspace(0, 1, 100)
cmap = cm.get_cmap('viridis')
# cmap = cm.get_cmap('plasma')
norm = Normalize(vmin=0, vmax=1)

# 创建 figure 来绘制 colorbar
fig, ax = plt.subplots(figsize=(2, 6))  # figsize 控制 colorbar 的大小
fig.subplots_adjust(left=0.5, right=0.8)  # 调整 colorbar 位置

# 绘制 colorbar
colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
colorbar.set_label("Color Scale")

# 保存 colorbar 图像
plt.savefig('colorbar.png', transparent=True, bbox_inches='tight')
plt.close()
