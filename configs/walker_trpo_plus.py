import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["trpo"],
    "out_dir": ["trpo_plus_walker/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "max_kl": [0.07] * 144,
    "max_kl_final": [0.04],
    "val_lr": [1e-4],
    "cpu": [True],
    "clip_grad_norm": [1.0],
    "advanced_logging": [True],
    "save_iters": [10],
}

generate_configs(BASE_CONFIG, PARAMS)
