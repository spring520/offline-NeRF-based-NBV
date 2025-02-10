import numpy as np
import torch
import mathutils
import tyro
import sys
import time
from torchmetrics.functional import structural_similarity_index_measure
from tqdm import tqdm
sys.path.append("/home/zhengquan/04-fep-nbv")

from config import *
from fep_nbv.utils import *
from fep_nbv.env.utils import *
from fep_nbv.env.shapenet_env import set_env
from nvf.active_mapping.mapping_utils import to_transform
from nerfstudio.cameras.cameras import Cameras, CameraType

import warnings
warnings.filterwarnings("ignore")

def Uncertainty_distribution_calculation(cfg, NeRF_pipeline, poses, gt_images, input_viewpoint_index):
    Uncertainty = {'PSNR':[],"uncertainty":[],'SSIM':[],"MSE":[]}
    
    # transforms = [to_transform(pose[[3,0,1,2,4,5,6]]) for pose in poses]
    transforms = [to_transform(pose) for pose in poses]
    fov = cfg.env.fov /180 *np.pi
    width = cfg.env.resolution[1] * torch.ones(len(transforms),1, dtype=torch.float32)
    height = cfg.env.resolution[0] * torch.ones(len(transforms),1, dtype=torch.float32)
    fx = 0.5*width/np.tan(fov/2)
    fy = fx
    cx = width//2
    cy = height//2

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

        # gt_images[i] = NeRF_pipeline.trainer.pipeline.model.renderer_rgb.blend_background(gt_images[i])
        # predicted_rgb, gt_rgb = NeRF_pipeline.trainer.pipeline.model.renderer_rgb.blend_background_for_loss_computation(
        #     gt_image=gt_images[i], pred_image=outputs["rgb"], pred_accumulation=outputs["accumulation"]
        # ) # 3 512 512; 512 512 3;
        # print(f'predicted_rgb max: {predicted_rgb.max()}')
        # print(f'gt_rgb max: {gt_rgb.max()}')
        # print(f'gt_images[i] max {gt_images[i].max()}')

        psnr = NeRF_pipeline.trainer.pipeline.model.psnr(gt_rgb/255,predicted_rgb)
        ssim = structural_similarity_index_measure(gt_rgb.permute(2, 0, 1).unsqueeze(0)/255,predicted_rgb.permute(2, 0, 1).unsqueeze(0))
        mse = torch.mean((gt_rgb/255 - predicted_rgb) ** 2)
        
        Uncertainty['PSNR'].append(psnr.item())
        Uncertainty['SSIM'].append(ssim.item())
        Uncertainty['MSE'].append(mse.item())
        Uncertainty['uncertainty'].append(outputs['entropy'].mean().item())

        if i==input_viewpoint_index:
            image1 = Image.fromarray(gt_rgb.cpu().detach().numpy().astype(np.uint8))
            image2 = Image.fromarray((predicted_rgb*255).cpu().detach().numpy().astype(np.uint8))
        if i==0:
            gt_rgb_0 = gt_rgb
            predicted_rgb_0=predicted_rgb*255
        if i==round(len(transforms)/2):
            gt_rgb_half = gt_rgb
            predicted_rgb_half=predicted_rgb*255

    if input_viewpoint_index<abs(len(transforms)/2-input_viewpoint_index):
        image3 = Image.fromarray(gt_rgb_half.cpu().detach().numpy().astype(np.uint8))
        image4 = Image.fromarray(predicted_rgb_half.cpu().detach().numpy().astype(np.uint8))
    else:
        image3 = Image.fromarray(gt_rgb_0.cpu().detach().numpy().astype(np.uint8))
        image4 = Image.fromarray(predicted_rgb_0.cpu().detach().numpy().astype(np.uint8))
    width, height = image1.size
    canvas = Image.new("RGB", (2 * width, 2 * height))
    canvas.paste(image1, (0, 0))  # 左上角
    canvas.paste(image2, (width, 0))  # 右上角
    canvas.paste(image3, (0, height))  # 左下角
    canvas.paste(image4, (width, height))  # 右下角
        
    return Uncertainty,canvas

def distribution_dataset_construction(model_path, num_azimuth=8, num_elevation=8, radius=[2], start_viewpoint_index=1, viewpoint_index_interval=1):
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path
    env = set_env(cfg)
    # cfg.train_iter=10
    NeRF_pipeline = NeRF_init(cfg)
    

    # output path 
    offset = model_path.split('/')[-2]
    category = offset2word(offset)
    output_path = model_path.replace('/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2','/mnt/hdd/zhengquan/Shapenet/distribution_dataset')
    output_path = output_path.replace(offset,category)
    os.makedirs(output_path,exist_ok=True)
    # 创建 image 和 uncertainty 文件夹
    image_dir = os.path.join(output_path, "images")
    uncertainty_dir = os.path.join(output_path, "uncertainties")
    nerf_dir = os.path.join(output_path, "nerf")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)
    os.makedirs(nerf_dir, exist_ok=True)

    # generate candidate viewpoint
    num_candidate_viewpoint = num_azimuth*num_elevation*len(radius)
    azimuths,elevations,radius = generate_candidate_viewpoint(num_azimuth=num_azimuth,num_elevation=num_elevation,radius=radius)
    poses = polar2pose(azimuths,elevations,radius)

    # 生成GT images
    gt_images_path = model_path.replace('/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2','/mnt/hdd/zhengquan/Shapenet/gt_shapenet')
    gt_images_path = gt_images_path.replace(offset,category)

    # 如果已经这个模型的GT image已经生成完毕，那么直接读取，而不是生成
    if os.path.exists(gt_images_path):
        existing_images = [f for f in os.listdir(gt_images_path) if f.endswith('_gt.png')]
        if len(existing_images) == num_candidate_viewpoint:
            print(f"Found {len(existing_images)} GT images, loading directly from {gt_images_path}")
            # 读取已存在的图片
            gt_images = []
            for index in range(num_candidate_viewpoint):
                image_path = os.path.join(gt_images_path, f'{index}_gt.png')
                img = Image.open(image_path)
                img = torch.FloatTensor(np.array(img)[...,:3])  # 保持与生成流程一致
                gt_images.append(img)
            # 将读取的图片转换为 Tensor
            gt_images = torch.stack(gt_images).to('cuda')
            print(f"Loaded {len(gt_images)} GT images successfully!")
        else:
            print(f"Number of images in {gt_images_path} is {len(existing_images)}, expected {num_candidate_viewpoint}. Regenerating...")
            gt_images = generate_gt_images(cfg, num_candidate_viewpoint, poses, gt_images_path)
    else:
        gt_images = generate_gt_images(cfg, num_candidate_viewpoint, poses, gt_images_path)
    
    # viewpoint 循环
    for input_viewpoint_index in tqdm(range(start_viewpoint_index,num_candidate_viewpoint,viewpoint_index_interval),desc='viewpoint'):
        # 如果东西都存在了，那就说明和这个input_viewpoint_index有关的东西都已经处理过了
        image_path = os.path.join(image_dir, f"viewpoint_{input_viewpoint_index}.png")
        example_image_path = os.path.join(image_dir, f"viewpoint_example_{input_viewpoint_index}.png")
        uncertainty_path = os.path.join(uncertainty_dir, f"viewpoint_{input_viewpoint_index}.json")
        if os.path.exists(image_path) and os.path.exists(example_image_path) and os.path.exists(uncertainty_path):
            continue
        
        print(f'dealing with {input_viewpoint_index}')
        
        # input_viewpoint_index = random.randint(0,len(azimuths)-1) # 随机采样一个视角
        input_view_point = (azimuths[input_viewpoint_index:input_viewpoint_index+1],elevations[input_viewpoint_index:input_viewpoint_index+1],radius[input_viewpoint_index:input_viewpoint_index+1])
        input_pose = polar2pose(input_view_point[0],input_view_point[1],input_view_point[2]) # xyzw xyz

        # 训练NeRF
        NeRF_pipeline = NeRF_init(cfg)
        env.reset()
        empty_cache() 
        obs,_,_,_=env.step(input_pose)
        t1 = time.time()
        NeRF_pipeline.add_image(images=obs,poses=input_pose,model_option=None)
        print_cuda_allocated()
        t2 = time.time()
        # print(f'NeRF trained, time used: {t2-t1:2f} seconds')

        # 计算每个视角上的不确定性
        Uncertainty, image = Uncertainty_distribution_calculation(cfg, NeRF_pipeline, poses, gt_images,input_viewpoint_index)

        # 储存input viewpoint和uncertrainty对
        save_img(obs[0],image_path)
        image.save(example_image_path)
        with open(uncertainty_path, 'w') as f:
            json.dump(Uncertainty, f)
        nerf_path = os.path.join(nerf_dir, f"viewpoint_{input_viewpoint_index}.ckpt")
        NeRF_pipeline.save_ckpt(nerf_path)

        del Uncertainty
        del image
        del NeRF_pipeline
        del input_pose
        empty_cache()



def generate_gt_images(cfg, num_candidate_viewpoint, poses, gt_images_path):
    env = set_env(cfg)

    NeRF_pipeline = NeRF_init(cfg)
    t1 = time.time()
    _,_,_,_ = env.step(poses)
    t2 = time.time()
    print(f'generating gt images, time used: {t2-t1:2f} seconds')
    gt_images = np.array(env.obs_history[-num_candidate_viewpoint:])
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
        

if __name__=='__main__':
    # parameter:
    num_azimuth = 8
    num_elevation = 8
    radius = [2]
    start_viewpoint_index = 0
    viewpoint_index_interval = 1
    output_path = '/mnt/hdd/zhengquan/Shapenet/distribution_dataset'
    progress_file = output_path+"/progress.json"

    # 加载已处理的模型记录
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            processed_data = json.load(file)  # 格式：{"category1": ["model1", "model2"], ...}
    else:
        processed_data = {}

    model_dict = shapenet_model_path_dict()
    for category, models in tqdm(model_dict.items(), desc="Processing Categories", unit="category"):
        # 跳过已完成的类别
        if category in processed_data and len(processed_data[category]) == len(models):
            continue

        # 确保当前类别的进度记录存在
        if category not in processed_data:
            processed_data[category] = []

        for model_path in tqdm(models, desc=f"Processing Models in {category}", unit="model", leave=False):
            if model_path in processed_data[category]:
                pass  # 跳过已处理的模型
            else:
                print(f'\n{category}:{model_path}\n\n')
                empty_cache()
                distribution_dataset_construction(model_path, num_azimuth, num_elevation, radius, start_viewpoint_index, viewpoint_index_interval)
                # 记录成功处理的模型
                processed_data[category].append(model_path)
                # 保存进度
                with open(progress_file, "w") as file:
                    json.dump(processed_data, file)





    # model_path = random_shapenet_model_path()
    # print(f'model path: {model_path}')

    # distribution_dataset_construction(model_path, num_azimuth, num_elevation, radius, start_viewpoint_index, viewpoint_index_interval)



    # 20250101 temp
    # 先渲染出一个模型的结果来用
    # model_path = "/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/03710193/1e4df43ee2f2da6967f9cc18b363cf72"
    # empty_cache()
    # distribution_dataset_construction(model_path, num_azimuth, num_elevation, radius, start_viewpoint_index, viewpoint_index_interval)