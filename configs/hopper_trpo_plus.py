import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Hopper-v2"],
    "mode": ["trpo"],
    "out_dir": ["trpo_plus_hopper/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "max_kl": [0.04] * 72,
    "max_kl_final": [0.07],
    "clip_grad_norm": [1.0],
    "val_lr": [2e-4],
    "advanced_logging": [True],
    "save_iters": [10],
    "cpu": [True]
}

generate_configs(BASE_CONFIG, PARAMS)
