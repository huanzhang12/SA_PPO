import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2"],
    "mode": ["trpo"],
    "out_dir": ["trpo_plus_humanoid/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [True],
    "max_kl": [0.1] * 72,
    "max_kl_final": [0.05],
    "val_lr": [5e-5],
    "clip_grad_norm": [0.5],
    "lambda": [0.85],
    "cpu": [True],
    "save_iters": [10],
    "advanced_logging": [True]
}

generate_configs(BASE_CONFIG, PARAMS)
