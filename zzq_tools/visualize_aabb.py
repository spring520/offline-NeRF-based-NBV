import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from nvf.env.Scene import HubbleScene
from config import *
import mathutils
from nvf.env.utils import get_images, GIFSaver, save_img


if __name__=='__main__':
    cfg = ExpConfig()
    cfg.object_aabb = torch.tensor([[-1, -1.1, -1.1], [1, 1.1, 1.8]])#*1.1
    factor = 2.5
    cfg.target_aabb = cfg.object_aabb*factor
    cfg.camera_aabb = cfg.object_aabb*factor
    cfg.env.scale = 0.3333 * 0.5 / 6
    cfg.density_threshold = 1e-3
    cfg.env.resolution = (1024,1024)


    scene = HubbleScene(cfg.env)
    scene.set_white_background()
    scene.add_coordinate_axes()

    # 测试添加长方体
    object_aabb=torch.Tensor([[-1.0000, -1.1000, -1.1000],[ 1.0000,  1.1000,  1.8000]])
    scene.add_transparent_cube(object_aabb,cube_color=(1,0,0,0.5))
    camera_aabb=torch.Tensor([[-2.5000, -2.7500, -2.7500],[ 2.5000,  2.7500,  4.5000]])
    scene.add_transparent_cube(camera_aabb,cube_color=(0,1,0,0.3))

    # 计算相机位置，目标位置和视角方向的四元数
    camera_position = mathutils.Vector(cfg.camera_aabb[0])
    target_position = mathutils.Vector(cfg.camera_aabb[1])
    direction = target_position - camera_position 
    rot_quat = torch.tensor(direction.to_track_quat('-Z', 'Y'))

    # 渲染固定斜上方视角下的目标图像并保存
    fixed_pose = torch.concat((rot_quat, torch.tensor(camera_position))) # 乘以1.5确保能够看到camera_position对应的框
    image = scene.render_pose(fixed_pose)


    # 图像存在log_path的views目录下，需要创建views并保存
    save_img(image, "zzq_tools/visualize_aabb.png")