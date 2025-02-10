# from nvf.env.utils import get_conf, pose_point_to, rgb_to_rgba, sharpness, variance_of_laplacian, load_from_json, write_to_json
# from nvf.env.utils import save_img
import os
from PIL import Image
Image.init()
# Image.OPEN = ["JPEG", "PNG"]
import numpy as np
import random 
import torch
import gc
import tempfile
import string
from gtsam import Pose3, Rot3
import cv2
import json
from pathlib import Path
import sys
sys.path.append("/home/zhengquan/04-fep-nbv")

from fep_nbv.utils.utils import offset2word

# from fep_nbv.env.shapenet_env import ShapeNetEnviroment

def is_rgb(image):
    """Check if the image is in RGB format."""
    return image.ndim == 3 and image.shape[2] == 3

def rgb_to_rgba(image, mask):
    """Convert an RGB image to RGBA."""
    if not is_rgb(image):
        raise ValueError("The input image is not in RGB format.")
    
def stack_img(img_mat,sep,gap_rgb=[255,255,255], shape=None):
    '''
    stack img to matrix
    img_mat:[[img1,img2],[img3,img4]]
    sep: (gap_size_height, gap_size_weight)
    '''
    if shape is not None:
        img_mat = reshape_img_matrix(img_mat, shape)

    if img_mat[0][0].max()<=1:
        gap_rgb=np.array(gap_rgb)/255.
    
    img_lines = []
    if type(img_mat) is list:
        num_row = len(img_mat)
    else:
        num_row=img_mat.shape[0]
    for j in range(num_row):
        line = img_mat[j]
        line_new = []
        for i in range(len(line)):
            if i==0:
                w_gap_h = line[i].shape[0]
                w_gap = np.ones((w_gap_h,sep[1],3))
                w_gap[...,:] = np.array(gap_rgb)
                w_gap = w_gap.astype(line[i].dtype)
                line_new.append(line[i])
            else:
                line_new.append(w_gap)
                line_new.append(line[i])
        # print([s.shape for s in line_new])
        img_line = cv2.hconcat(line_new)
        if j==0:
            h_gap_w = img_line.shape[1]
            h_gap = np.ones((sep[0],h_gap_w,3))
            h_gap[...,:] = np.array(gap_rgb)
            h_gap = h_gap.astype(img_line.dtype)
        else:
            img_lines.append(h_gap)
        img_lines.append(img_line)
    img_mat = cv2.vconcat(img_lines)
    return img_mat


def get_images(idx_start=None, idx_end=None, file='data/nerfstudio/hubble_mask/transforms.json', img_path=None, return_quat=True):
    """copies images from hubble dataset used for testing add_images()"""
    # path='cfg/initial_transforms.json'
    # hubble_dataset_path = "data/nerfstudio/hubble_mask/img"
    # hubble_dataset_path = "/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/img"
    # files = Path(hubble_dataset_path).glob('*')

    transforms_dict = load_from_json(Path(file))
    # transforms_dict = load_from_json(Path("/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/transforms/transforms.json"))
    # transforms_dict = load_from_json(Path("data/nerfstudio/hubble_mask/transforms.json"))

    np_images = []
    transforms = []
    i = 0
    if idx_start is None:
        idx_start=0
    if idx_end is None:
        idx_end=len(transforms_dict["frames"])

    for i in range(idx_start, idx_end):
        if img_path:
            image = Image.open(os.path.join(img_path,f"{i:04}.png"))
        else:
            image = Image.open(os.path.join(os.path.dirname(file),transforms_dict["frames"][i]['file_path']))
        np_image = np.array(image)
        # np_image = rgb_to_rgba(np_image)
        np_images.append(np_image)

        transform = np.asarray(transforms_dict["frames"][i]['transform_matrix'])
        # import pdb; pdb.set_trace()
        if return_quat:
            pose = Pose3(transform)  
            q = pose.rotation().toQuaternion() # w, x, y, z
            t = pose.translation()
            transforms.append(torch.FloatTensor([q.x(), q.y(), q.z(), q.w(), t[0], t[1], t[2]]))
        else:
            transforms.append(torch.FloatTensor(transform))

    return np_images, transforms

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(image):
	# image = cv2.imread(imagePath)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def is_rgb(image):
    """Check if the image is in RGB format."""
    return image.ndim == 3 and image.shape[2] == 3

def rgb_to_rgba(image, mask):
    """Convert an RGB image to RGBA."""
    if not is_rgb(image):
        raise ValueError("The input image is not in RGB format.")
    
    height, width, _ = image.shape
    
    if image.dtype == np.uint8:
        alpha_channel = np.ones((height, width, 1), dtype=np.uint8) * 255 *mask
        
    elif np.issubdtype(image.dtype, np.floating):
        alpha_channel = np.ones((height, width, 1), dtype=image.dtype)*mask
    else:
        raise ValueError("Image dtype not supported. Expected uint8 or float.")
    return np.concatenate((image, alpha_channel), axis=2).astype(image.dtype)

def pose_point_to(loc, target=[0,0,0.], up=[0,0,1.]):
    '''
    - input
        loc: camera position

    - output
        pose: 4x4矩阵，其中3x4是有效的
    '''
    x0 = np.array(target)
    p = np.array(loc)
    up = np.array(up)
    up = up/np.linalg.norm(up)
    ez = (p-x0)/ np.linalg.norm(p-x0)
    ex = np.cross(up, ez)
    if np.linalg.norm(ex)<1e-7:
        up = up+np.array([0.1, 0, 0])
        up = up/np.linalg.norm(up)
        ex = np.cross(up, ez)
    ex = ex/np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey = ey/np.linalg.norm(ey)
    # print(np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez))
    rot = np.array([ex, ey, ez]).T
    pose = np.eye(4)
    pose[:3,:3] = rot
    pose[:3,3] = p
    return pose

def pose2tensor(pose):
    '''
    - input
        3x4
    - output
        xyzwxyz
    '''
    if type(pose) is not Pose3:
        pose = Pose3(pose)  
    q = pose.rotation().toQuaternion() # w, x, y, z
    t = pose.translation()
    return torch.FloatTensor([q.x(), q.y(), q.z(), q.w(), t[0], t[1], t[2]])

def tensor2pose(pose):
    '''
    - input
        xyzwxyz
    - output
        3x4
    '''
    pose = pose.detach().cpu().numpy()
    quat = pose[:4]
    pos = pose[-3:]
    return Pose3(Rot3.Quaternion(quat[-1], *quat[:3]), pos)

def save_img(img, filepath):
    '''
    保存图片
    - input
        img:    numpy.array
        filepat: str
    - output
        no
    '''
    path = os.path.dirname(filepath)
    if not os.path.exists(path):
        os.makedirs(path)
    if img.max()>1:
        img = Image.fromarray(np.uint8(img))
    else:
        img = Image.fromarray(np.uint8(img*255))
    
    # print(filepath)
    img.save(filepath)
    img.close()

class GIFSaver(object):
    """docstring for GIFSaver"""
    def __init__(self, name=None, path=None, temp_format="png"):
        super(GIFSaver, self).__init__()
        # self.arg = arg
        # self.count=0
        if name is not None:
            self.name = name
            # self.isname = True
        else:
            # self.isname = False
            self.name = ''.join(random.choice(string.ascii_uppercase) for i in range(6))
        self.path = path
        self.temp_format = temp_format.lower()
        self.temp_path = tempfile.gettempdir()
        self.file_list = []
        self.fig_list = []
        self.count=0

    def __call__(self,count=None):
        if count is None:
            count=self.count
        fname = 'gif_tmp_'+self.name+f'_{count}.{self.temp_format}'
        fpath = os.path.join(self.temp_path, fname)
        self.file_list.append(fpath)
        self.count+=1
        return fpath
    
    def add(self, img, fn=None):
        if img.shape[-1]>3:
            img = img[...,:3] # remove alpha channel
        if img.max()>1:
            img = Image.fromarray(np.uint8(img))
        else:
            img = Image.fromarray(np.uint8(img*255))
        if fn:
            fn(img)
        self.fig_list.append(img)
        self.count+=1

    def save(self,name=None,path=None, duration=500, loop=0):
        if name :
            if os.sep in name:
                output_path = name
            elif path is not None:
                output_path = os.path.join(path, name)
            elif self.path is not None:
                output_path = os.path.join(self.path, name)
            else:
                output_path = os.path.join(os.getcwd(), name)
        else:
            if path is not None:
                output_path = os.path.join(path, self.name)
            elif self.path is not None:
                output_path = os.path.join(self.path, self.name)
            else:
                output_path = os.path.join(os.getcwd(), self.name)
        if not output_path.endswith('.gif'):
            output_path+='.gif'

        if not self.fig_list:
            images=[]
            for img_file in self.file_list:
                im = Image.open(img_file)
                images.append(im)
        else:
            images = self.fig_list
        assert len(images)>=2, 'Need at least two images in the list'
        images[0].save(os.path.join(output_path), save_all=True, append_images=images[1:], duration=duration, loop=loop)
        for img_file in self.file_list:
            os.remove(img_file)

def empty_cache():
    torch.cuda.empty_cache(); gc.collect()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)

def random_shapenet_model_path():
    shapenet_dataset_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2'
    categories = [
        category for category in os.listdir(shapenet_dataset_path)
        if os.path.isdir(os.path.join(shapenet_dataset_path, category))
    ]
    
    if not categories:
        raise ValueError("No categories found in the provided root directory.")

    # 随机选择一个类别
    selected_category = random.choice(categories)

    # 获取该类别下的所有模型目录
    category_path = os.path.join(shapenet_dataset_path, selected_category)
    models = [
        os.path.join(category_path, model)
        for model in os.listdir(category_path)
        if os.path.isdir(os.path.join(category_path, model))
    ]

    if not models:
        raise ValueError(f"No models found in category {selected_category}.")

    # 随机选择一个模型
    selected_model = random.choice(models)

    return selected_model


def shapenet_model_path_dict():
    shapenet_dataset_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2'
    shapenet_dict = {}
    # 遍历根目录下的类别目录
    for category in os.listdir(shapenet_dataset_path):
        category_path = os.path.join(shapenet_dataset_path, category)
        # 确保目录存在且为文件夹
        if os.path.isdir(category_path):
            # 获取该类别目录下所有模型目录
            model_dirs = [
                os.path.join(category_path, model) 
                for model in os.listdir(category_path)
                if os.path.isdir(os.path.join(category_path, model))
            ]
            # 将类别名作为 key，模型目录列表作为 value
            categoryname = offset2word(category)
            shapenet_dict[categoryname] = model_dirs
    
    return shapenet_dict

# def set_env(cfg):

#     if cfg.scene.name == 'hubble':
#         # breakpoint()
#         #aabb = ([[-0.92220873, -1.00288355, -1.03578806],
#     #    [ 0.92220724,  1.05716348,  1.75192416]])
#         cfg.object_aabb = torch.tensor([[-1, -1.1, -1.1], [1, 1.1, 1.8]])#*1.1
#         factor = 2.5
#         cfg.target_aabb = cfg.object_aabb*factor
#         cfg.camera_aabb = cfg.object_aabb*factor
#         cfg.env.scale = 0.3333 * 0.5
#         cfg.density_threshold = 1e-3

#     elif cfg.scene.name =='lego':
#         # array([[-0.6377874 , -1.14001584, -0.34465557],
#     #    [ 0.63374418,  1.14873755,  1.00220573]])
#         factor = 2.5
#         cfg.object_aabb = torch.tensor([[-0.7, -1.2, -0.345], [0.7, 1.2, 1.1]])
        
#         ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

#         cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

#         # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
#         cfg.target_aabb = cfg.camera_aabb
#     elif cfg.scene.name =='drums':
#         # array([[-1.12553668, -0.74590737, -0.49164271],
#         #[ 1.1216414 ,  0.96219957,  0.93831432]])
#         factor = 2.5
#         cfg.object_aabb = torch.tensor([[-1.2, -0.8, -0.49164271], [1.2, 1.0, 1.0]])
        
#         ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

#         cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

#         # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
#         cfg.target_aabb = cfg.camera_aabb

#         cfg.cycles_samples = 50000
#         # cfg.env.n_init_views = 5
#     elif cfg.scene.name =='hotdog':
#         # wrong aabb [[-1.22326267 -1.31131911 -0.19066653]
#         # [ 1.22326279  1.13520646  0.32130781]]
        
#         # correct aabb [[-1.19797897 -1.28603494 -0.18987501]
#         # [ 1.19797897  1.10992301  0.31179601]]
#         # factor = 3
#         cfg.object_aabb = torch.tensor([[-1.3, -1.4, -0.18987501], [1.3, 1.2, 0.5]])

#         diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
#         cfg.camera_aabb = cfg.object_aabb+diff_box
#         cfg.target_aabb = cfg.camera_aabb

#         cfg.env.n_init_views = 5
#         # cfg.check_density = True

#     elif cfg.scene.name =='room':
#         factor = 1
#         cfg.object_aabb = torch.tensor([[-12.4, -4.5,-0.22], [4.1, 6.6, 5.2]])
#         cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
#         cfg.target_aabb = cfg.camera_aabb
#         cfg.env.scale = 0.3333 * 0.5
#     elif cfg.scene.name =='ship':
#         # [[-1.27687299 -1.29963005 -0.54935801]
#         # [ 1.37087297  1.34811497  0.728508  ]]
#         cfg.object_aabb = torch.tensor([[-1.35, -1.35,-0.54935801], [1.45, 1.45, 0.73]])
        
#         diff_box = torch.tensor([[-1.7,-1.7,0.43], [1.7,1.7,3.3]])
        
#         cfg.camera_aabb = cfg.object_aabb+diff_box
#         cfg.target_aabb = cfg.camera_aabb

#         # cfg.env.n_init_views = 3

#         # if cfg.d0 > 0.: cfg.d0=0.8

#     elif cfg.scene.name =='chair':
#         # [[-0.72080803 -0.69497311 -0.99407679]
#         # [ 0.65813684  0.70561057  1.050102  ]]

#         cfg.object_aabb = torch.tensor([[-0.8, -0.8,-0.99407679], [0.8, 0.8, 1.1]])
        
#         diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
#         cfg.camera_aabb = cfg.object_aabb+diff_box
#         cfg.target_aabb = cfg.camera_aabb

#     elif cfg.scene.name =='mic':
#     #     array([[-1.25128937, -0.90944701, -0.7413525 ],
#     #    [ 0.76676297,  1.08231235,  1.15091646]])
#         # factor = 2.5
#         cfg.object_aabb = torch.tensor([[-1.3, -1.0,-0.7413525], [0.8, 1.2, 1.2]])
#         diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
#         cfg.camera_aabb = cfg.object_aabb+diff_box
        
#         # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

#         # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
        

#         cfg.target_aabb = cfg.camera_aabb
#         # cfg.env.n_init_views = 5
#         # breakpoint()

#     elif cfg.scene.name =='materials':
#         # [[-1.12267101 -0.75898403 -0.23194399]
#         # [ 1.07156599  0.98509198  0.199104  ]]
#         # factor = torch.tensor([2.5, 2.5, 3.5]).reshape(-1,3)
#         cfg.object_aabb = torch.tensor([[-1.2, -0.8,-0.23194399], [1.2, 1.0, 0.3]])
#         # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

#         # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

#         diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
#         cfg.camera_aabb = cfg.object_aabb+diff_box
#         cfg.target_aabb = cfg.camera_aabb

#         cfg.target_aabb = cfg.camera_aabb
#         # breakpoint()
#     elif cfg.scene.name =='ficus':
#         #[[-0.37773791 -0.85790569 -1.03353798]
#         #[ 0.55573422  0.57775307  1.14006007]]
#         factor = 2.5
#         cfg.object_aabb = torch.tensor([[-0.4, -0.9, -1.03353798], [0.6, 0.6, 1.2]])

#         ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

#         cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
#         cfg.target_aabb = cfg.camera_aabb

#         # cfg.env.n_init_views = 5
#     elif cfg.scene.name =='shapenet':
#         cfg.object_aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]])#*1.1
#         factor = 2
#         cfg.target_aabb = cfg.object_aabb
#         cfg.camera_aabb = cfg.object_aabb
#         # cfg.env.scale = 0.3333 * 0.5
#         cfg.density_threshold = 1e-3
#     else:
#         raise NotImplementedError
#     env = ShapeNetEnviroment(cfg.env)
#     # breakpoint()
#     return env

if __name__=='__main__':
    model_path_dict = shapenet_model_path_dict()
    for category, models in model_path_dict.items():
        print(f"Category: {category}, \t\tNumber of models: {len(models)}")
    model_path = random_shapenet_model_path()
    print(f'random seleted model path: {model_path}')

  