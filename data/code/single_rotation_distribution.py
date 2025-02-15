import sys
import time
import numpy as np
import tyro
import os
from PIL import Image
import torch
from torchmetrics.functional import structural_similarity_index_measure
import os
import cProfile
import pstats
root_path = os.getenv('nbv_root_path', '/default/path')
shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')
distribution_dataset_path = os.getenv('distribution_dataset_path', '/default/distribution/dataset/path')
sys.path.append(root_path)


from config import *
from fep_nbv.env.shapenet_env import set_env
from fep_nbv.utils.utils import *
from fep_nbv.env.utils import *
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints
from nerfstudio.cameras.cameras import Cameras, CameraType
from nvf.active_mapping.mapping_utils import to_transform

import warnings
warnings.filterwarnings("ignore")

def Uncertainty_distribution_calculation(cfg, NeRF_pipeline, absolute_viewpoint_poses, gt_images, input_viewpoint_index):
    Uncertainty = {'PSNR':[],"uncertainty":[],'SSIM':[],"MSE":[]}
    
    # transforms = [to_transform(pose[[3,0,1,2,4,5,6]]) for pose in poses]
    transforms = [to_transform(absolute_viewpoint_pose) for absolute_viewpoint_pose in absolute_viewpoint_poses]
    fov = cfg.env.fov /180 *np.pi
    width = cfg.env.resolution[1] * torch.ones(len(transforms),1, dtype=torch.float32)
    height = cfg.env.resolution[0] * torch.ones(len(transforms),1, dtype=torch.float32)
    fx = 0.5*width/np.tan(fov/2)
    fy = fx
    cx = width//2
    cy = height//2

    with torch.no_grad():
        for i in range(len(transforms)):
            cameras = Cameras(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                camera_type=CameraType.PERSPECTIVE,
                camera_to_worlds=transforms[i][:-1,:] # 3x4
                ).to(NeRF_pipeline.trainer.device)
            camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=None)
            outputs = NeRF_pipeline.trainer.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            predicted_rgb = outputs['rgb']
            gt_rgb=gt_images[i]


            psnr = NeRF_pipeline.trainer.pipeline.model.psnr(gt_rgb/255,predicted_rgb)
            ssim = structural_similarity_index_measure(gt_rgb.permute(2, 0, 1).unsqueeze(0)/255,predicted_rgb.permute(2, 0, 1).unsqueeze(0))
            mse = torch.mean((gt_rgb/255 - predicted_rgb) ** 2)
            
            Uncertainty['PSNR'].append(psnr.item())
            Uncertainty['SSIM'].append(ssim.item())
            Uncertainty['MSE'].append(mse.item())
            Uncertainty['uncertainty'].append(outputs['entropy'].mean().item())

            if i==0:
                image1 = Image.fromarray(gt_rgb.cpu().detach().numpy().astype(np.uint8))
                image2 = Image.fromarray((predicted_rgb*255).cpu().detach().numpy().astype(np.uint8))
            if i==round(len(transforms)-1-input_viewpoint_index):
                image3 = Image.fromarray(gt_rgb.cpu().detach().numpy().astype(np.uint8))
                image4 = Image.fromarray((predicted_rgb*255).cpu().detach().numpy().astype(np.uint8))

            # Image.fromarray(gt_rgb.cpu().detach().numpy().astype(np.uint8)).save(f'data/test/transformation_test/{input_viewpoint_index}_{i}_gt.png')
            # Image.fromarray((predicted_rgb*255).cpu().detach().numpy().astype(np.uint8)).save(f'data/test/transformation_test/{input_viewpoint_index}_{i}_render.png')
            del cameras
            del camera_ray_bundle
            del outputs
    del transforms
    empty_cache()

    width, height = image1.size
    canvas = Image.new("RGB", (2 * width, 2 * height))
    canvas.paste(image1, (0, 0))  # 左上角
    canvas.paste(image2, (width, 0))  # 右上角
    canvas.paste(image3, (0, height))  # 左下角
    canvas.paste(image4, (width, height))  # 右下角
        
    return Uncertainty,canvas

def generate_gt_images(cfg, absolute_viewpoint_poses, gt_images_path):
    env = set_env(cfg)
    NeRF_pipeline = NeRF_init(cfg)

    if os.path.exists(gt_images_path):
        existing_images = [f for f in os.listdir(gt_images_path) if f.endswith('_gt.png')]
        if len(existing_images) == len(absolute_viewpoint_poses):
            print(f"Found {len(existing_images)} GT images, loading directly from {gt_images_path}")
            # 读取已存在的图片
            gt_images = []
            for index in range(len(absolute_viewpoint_poses)):
                image_path = os.path.join(gt_images_path, f'{index}_gt.png')
                img = Image.open(image_path)
                img = torch.FloatTensor(np.array(img)[...,:3])  # 保持与生成流程一致
                gt_images.append(img)
            # 将读取的图片转换为 Tensor
            gt_images = torch.stack(gt_images).to(NeRF_pipeline.trainer.device)
            print(f"Loaded {len(gt_images)} GT images successfully!")

            del env
            del NeRF_pipeline
            empty_cache()
            return gt_images

    t1 = time.time()
    _,_,_,_ = env.step(absolute_viewpoint_poses)
    t2 = time.time()
    print(f'generating gt images, time used: {t2-t1:2f} seconds')
    gt_images = np.array(env.obs_history[-len(absolute_viewpoint_poses):])
    gt_images = torch.stack([torch.FloatTensor(iii)[...,:3] for iii in gt_images]).to(NeRF_pipeline.trainer.device)
    print('saving groundtruth images')
    if not os.path.exists(gt_images_path):
        os.makedirs(gt_images_path,exist_ok=True)
    print(gt_images.shape)
    for index,gt_image in enumerate(gt_images):
        image = Image.fromarray(gt_image.cpu().detach().numpy().astype(np.uint8))
        image.save(os.path.join(gt_images_path,f'{index}_gt.png'))

    del env
    del NeRF_pipeline
    empty_cache()
    
    return gt_images

def generate_distribution_single_rotation(model_path,input_viewpoint_index,offset_phi_index,cfg):
    offset_phi = offset_phi_index * 0.2 * np.pi
    candidate_viewpoint_poses = generate_HEALPix_viewpoints(n_side=2)
    candidate_viewpoint_pose = candidate_viewpoint_poses[input_viewpoint_index]
    output_dataset_path = distribution_dataset_path
    env = set_env(cfg)

    # 创建各种文件夹 image uncertainty nerf ckpt
    offset = model_path.split('/')[-2]
    category = offset2word(offset)
    output_path = model_path.replace(shapenet_path,output_dataset_path)
    output_path = output_path.replace(offset,category)
    # 创建 image 和 uncertainty 文件夹
    image_dir = os.path.join(output_path, "images")
    uncertainty_dir = os.path.join(output_path, "uncertainties")
    nerf_dir = os.path.join(output_path, "nerf")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)
    os.makedirs(nerf_dir, exist_ok=True)

    time1 = time.time()
    print(f'dealing with viewpoint {input_viewpoint_index} and rotate {offset_phi}')
    
    # 如果东西都存在了，那就说明和这个input_viewpoint_index有关的东西都已经处理过了
    image_path = os.path.join(image_dir, f"viewpoint_{input_viewpoint_index}_offset_phi_{offset_phi_index}.png")
    example_image_path = os.path.join(image_dir, f"viewpoint_example_{input_viewpoint_index}_offset_phi_{offset_phi_index}.png")
    uncertainty_path = os.path.join(uncertainty_dir, f"viewpoint_{input_viewpoint_index}_offset_phi_{offset_phi_index}.json")
    gt_images_path = os.path.join(image_dir, f"gt_images/{input_viewpoint_index}_offset_phi_{offset_phi_index}")
    if os.path.exists(image_path) and os.path.exists(example_image_path) and os.path.exists(uncertainty_path):
        print(f'viewpoint {input_viewpoint_index} and rotate {offset_phi} already finished so skip')
        return
    
    absolute_viewpoint_poses = generate_HEALPix_viewpoints(n_side=2,original_viewpoint=np.array(candidate_viewpoint_pose[4:]),offset_phi=offset_phi)
    gt_images = generate_gt_images(cfg, absolute_viewpoint_poses, gt_images_path)

    # 训练NeRF
    NeRF_pipeline = NeRF_init(cfg)
    env = set_env(cfg)
    empty_cache() 
    obs,_,_,_=env.step(absolute_viewpoint_poses[0])
    NeRF_pipeline.add_image(images=obs,poses=absolute_viewpoint_poses[0].unsqueeze(0),model_option=None)
    t2 = time.time()

    # 计算每个视角上的不确定性
    Uncertainty, image = Uncertainty_distribution_calculation(cfg, NeRF_pipeline, absolute_viewpoint_poses, gt_images,input_viewpoint_index)

    # 储存input viewpoint和uncertrainty对
    save_img(obs[0],image_path)
    image.save(example_image_path)
    with open(uncertainty_path, 'w') as f:
        json.dump(Uncertainty, f)
    nerf_path = os.path.join(nerf_dir, f"viewpoint_{input_viewpoint_index}_offset_phi_{offset_phi_index}.ckpt")
    NeRF_pipeline.save_ckpt(nerf_path)
    
    del Uncertainty
    del image
    del NeRF_pipeline
    del gt_images
    del absolute_viewpoint_poses
    del env
    del obs
    empty_cache()
    print_cuda_allocated()
    time2=time.time()
    print(f"single viewpoint single offset_phi运行时间: {time2-time1:.2f} 秒")
    print(f"single viewpoint single offset_phi运行时间: {(time2-time1) / 60:.2f} 分钟")

if __name__=='__main__':
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    model_path = cfg.env.target_path
    obj_file_path = cfg.env.target_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path

    # 从命令行参数中获取输入
    
    input_viewpoint_index =  cfg.env.viewpoint_index
    offset_phi_index =  cfg.env.offset_phi_index
    profile_output = "profile_output.prof"
    generate_distribution_single_rotation(model_path,input_viewpoint_index,offset_phi_index,cfg)
    # cProfile.run("generate_distribution_single_rotation(model_path,input_viewpoint_index,offset_phi_index,cfg)", profile_output)
    # 读取并排序
    # stats = pstats.Stats(profile_output)
    # stats.strip_dirs().sort_stats("cumulative").print_stats(50)