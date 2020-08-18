import os
import json

import sys
#sys.path.append("../")
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Hopper-v2"],
    "mode": ["ppo"],
    "clip_eps": [1e32],
    "out_dir": ["ppo_noclip_hopper/agents"],
    "norm_rewards": ["rewards"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [6e-5] * 72,
    "entropy_coeff": [-0.005],
    "lambda": [0.925],
    "val_lr": [4e-4],
    "cpu": [True],
    "clip_rewards": [2.5],
    "clip_grad_norm": [4.],
    "save_iters": [10],
    "advanced_logging": [True]
}

generate_configs(BASE_CONFIG, PARAMS)
