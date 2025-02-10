import sys
from pathlib import Path
sys.path.append("/attached/data/remote-home2/zzq/04-fep-nbv")
from fep_nbv.env.utils import load_from_json

if __name__=='__main__':
    Uncertainty = load_from_json(Path('/remote-home/zzq/data/ShapeNet/distribution_dataset/laptop/f00ec45425ee33fe18363ee01bdc990/uncertainties/viewpoint_0.json'))
    a = 1