from dataclasses import dataclass, field
from nvf.eval_utils import StrEnum
from typing import Literal, Optional, List, Tuple
import torch

class ModelType(StrEnum):
    ngp = 'instant-ngp'
    # nerfacto = 'nerfacto'

class AgentType(StrEnum):
    base = 'BaseAgent' # Base Agent
    opt = 'OptAgent' # Opt Agent
    dist = 'DistAgent' # Distance Agent

class SamplerType(StrEnum):
    base = 'BaseSampler' # Base sampler
    cf = 'CFSampler' # Collision free sampler
    spherical='SphericalSampler' # Spherical sampler
    healpix='HEALPixSampler'

class MethodType(StrEnum):
    nvf = 'NVF' # our method
    wd = 'WeightDist' # Entropy estimation method proposed by Lee 2022 RAL https://arxiv.org/abs/2209.08409

class SceneType(StrEnum):
    hubble = 'HubbleScene'
    lego = 'LegoScene'
    room = 'RoomScene'
    hotdog = 'HotdogScene'
    shapenet = 'ShapeNetScene'

class ActionType(StrEnum):
    discrete = 'discrete'
    sphere = 'continuous_sphere'
    aabb = 'continuous_aabb' 

@dataclass
class EnvConfig:
    scene: SceneType = SceneType.shapenet
    resolution: Tuple[int, int] = (512, 512)
    resolution_percentage: int = 100
    fov: float = 90
    hdri_rotation: Tuple[float, float, float] = (0., 0, 0)
    cycles: bool = True
    cycles_samples: int = 10000
    gpu = True # True
    scale: float = 2.
    horizon: int = 20
    n_init_views: int = 3
    gen_init: bool = False
    gen_eval: bool  = False
    save_data: bool  = True
    root: str = "data"
    target_path: str='/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/02691156/1ac29674746a0fc6b87697d3904b168b'
    viewpoint_index: int=45
    offset_phi_index: int=1
    action_space_mode = ActionType.discrete

@dataclass
class ExpConfig:
    method: MethodType = MethodType.wd
    agent: AgentType = AgentType.dist #AgentType.base 
    sampler: SamplerType = SamplerType.healpix # SamplerType.base
    model: ModelType = ModelType.ngp
    # scene: Literal["hubble", "lego", "room"] = "hubble"
    scene: SceneType = SceneType.shapenet
    task: Literal["map"] = "map"
    env: EnvConfig = EnvConfig(scene=scene)
    
    n_repeat: int = 3 # number of reruns
    horizon: int = 20 # number of steps for evaluation
    weight_dist: float = 1.0 # 移动代价在目标函数中的占比
    # init_views: int = 10 # number of initial views
    
    n_sample: int = 512 # number of samples for the sampler
    n_opt: int = 3 # number of samples for the optimization in OptSampler
    opt_iter: int = 20 # number of iterations for the optimization in OptSampler
    opt_lr: float = 1e-4 # learning rate for the optimization in OptSampler
    check_density: bool = True # check density for Sampler
    density_threshold: float = 1e-4 # density threshold for Sampler
    
    train_iter: int = 5000 # number of iterations for NeRF training
    train_iter_last_factor: float = 1.0 # number of iterations for NeRF training last step
    train_use_tensorboard: bool = True
    use_uniform: bool = False
    use_huber: bool = True
    use_vis: bool = True
    use_var: bool = True
    mu: float = 0.95
    
    name: Optional[str] = None # experiment short name
    exp_name: Optional[str] = None # experiment full name (this param will get override)
    # exp_folder: str = "./data/logs/results_debugging" # experiment save folder (this param will get override)
    exp_folder: str = "./data/test/eval_test"

    object_aabb: Optional[torch.Tensor] = None # object aabb shape 3x2, used for nerfacto_field
    target_aabb: Optional[torch.Tensor] = None # target aabb shape 3x2, used for sampler
    camera_aabb: Optional[torch.Tensor] = None # agent aabb shape 3x2

    d0: float = 1.0 # d0 \approx kD (see Eq.19 in the appendix)

    def __post_init__(self):
        self.env.scene= self.scene
        self.env.horizon = self.horizon

        if self.method == 'Random':
            self.agent = AgentType.random
        
        if self.scene.name == 'room':
            self.d0 = self.d0 * 6.0 # room scene has larger scale
