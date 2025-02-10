import sys
import tyro
import mathutils
import torch

# sys.path.append('/attached/data/remote-home2/zzq/demos/nvf_cvpr24')
sys.path.append('/attached/data/remote-home2/zzq/04-fep-nbv')

from config import *
from fep_nbv.env.shapenet_scene import ShapeNetScene
# from nvf.env.Enviroment import Enviroment

def set_env(cfg):

    if cfg.scene.name == 'hubble':
        # breakpoint()
        #aabb = ([[-0.92220873, -1.00288355, -1.03578806],
    #    [ 0.92220724,  1.05716348,  1.75192416]])
        cfg.object_aabb = torch.tensor([[-1, -1.1, -1.1], [1, 1.1, 1.8]])#*1.1
        factor = 2.5
        cfg.target_aabb = cfg.object_aabb*factor
        cfg.camera_aabb = cfg.object_aabb*factor
        cfg.env.scale = 0.3333 * 0.5
        cfg.density_threshold = 1e-3
    
    elif cfg.scene.name == 'shapenet':
        print('快想办法写写吧')
        pass

    elif cfg.scene.name =='lego':
        # array([[-0.6377874 , -1.14001584, -0.34465557],
    #    [ 0.63374418,  1.14873755,  1.00220573]])
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-0.7, -1.2, -0.345], [0.7, 1.2, 1.1]])
        
        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb
    elif cfg.scene.name =='drums':
        # array([[-1.12553668, -0.74590737, -0.49164271],
        #[ 1.1216414 ,  0.96219957,  0.93831432]])
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-1.2, -0.8, -0.49164271], [1.2, 1.0, 1.0]])
        
        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb

        cfg.cycles_samples = 50000
        # cfg.env.n_init_views = 5
    elif cfg.scene.name =='hotdog':
        # wrong aabb [[-1.22326267 -1.31131911 -0.19066653]
        # [ 1.22326279  1.13520646  0.32130781]]
        
        # correct aabb [[-1.19797897 -1.28603494 -0.18987501]
        # [ 1.19797897  1.10992301  0.31179601]]
        # factor = 3
        cfg.object_aabb = torch.tensor([[-1.3, -1.4, -0.18987501], [1.3, 1.2, 0.5]])

        diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        cfg.env.n_init_views = 5
        # cfg.check_density = True

    elif cfg.scene.name =='room':
        factor = 1
        cfg.object_aabb = torch.tensor([[-12.4, -4.5,-0.22], [4.1, 6.6, 5.2]])
        cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb
        cfg.env.scale = 0.3333 * 0.5
    elif cfg.scene.name =='ship':
        # [[-1.27687299 -1.29963005 -0.54935801]
        # [ 1.37087297  1.34811497  0.728508  ]]
        cfg.object_aabb = torch.tensor([[-1.35, -1.35,-0.54935801], [1.45, 1.45, 0.73]])
        
        diff_box = torch.tensor([[-1.7,-1.7,0.43], [1.7,1.7,3.3]])
        
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        # cfg.env.n_init_views = 3

        # if cfg.d0 > 0.: cfg.d0=0.8

    elif cfg.scene.name =='chair':
        # [[-0.72080803 -0.69497311 -0.99407679]
        # [ 0.65813684  0.70561057  1.050102  ]]

        cfg.object_aabb = torch.tensor([[-0.8, -0.8,-0.99407679], [0.8, 0.8, 1.1]])
        
        diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

    elif cfg.scene.name =='mic':
    #     array([[-1.25128937, -0.90944701, -0.7413525 ],
    #    [ 0.76676297,  1.08231235,  1.15091646]])
        # factor = 2.5
        cfg.object_aabb = torch.tensor([[-1.3, -1.0,-0.7413525], [0.8, 1.2, 1.2]])
        diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        
        # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
        

        cfg.target_aabb = cfg.camera_aabb
        # cfg.env.n_init_views = 5
        # breakpoint()

    elif cfg.scene.name =='materials':
        # [[-1.12267101 -0.75898403 -0.23194399]
        # [ 1.07156599  0.98509198  0.199104  ]]
        # factor = torch.tensor([2.5, 2.5, 3.5]).reshape(-1,3)
        cfg.object_aabb = torch.tensor([[-1.2, -0.8,-0.23194399], [1.2, 1.0, 0.3]])
        # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        cfg.target_aabb = cfg.camera_aabb
        # breakpoint()
    elif cfg.scene.name =='ficus':
        #[[-0.37773791 -0.85790569 -1.03353798]
        #[ 0.55573422  0.57775307  1.14006007]]
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-0.4, -0.9, -1.03353798], [0.6, 0.6, 1.2]])

        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
        cfg.target_aabb = cfg.camera_aabb

        # cfg.env.n_init_views = 5
    else:
        raise NotImplementedError
    env = Enviroment(cfg.env)
    # breakpoint()
    return env


if __name__=='__main__':
    cfg = tyro.cli(ExpConfig)
    print(cfg) # 打印出来是hubble scene
    # input
    model_path = "/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/02691156/1a6ad7a24bb89733f412783097373bdc"
    # function
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path 
    scene = eval(cfg.env.scene)(cfg=cfg.env)

    camera_position = mathutils.Vector((1, 1, 1))
    target_position = mathutils.Vector((0, 0, 0))
    direction = target_position - camera_position
    rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y')) # wxyz
    fixed_pose = torch.concat((rot_quat[[1,2,3,0]], torch.tensor(camera_position))) # xyzwxyz

    img = scene.render_pose(fixed_pose) # RGBA