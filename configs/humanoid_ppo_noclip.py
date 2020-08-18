import os
import json

import sys
sys.path.append("../")
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2"],
    "mode": ["ppo"],
    "clip_eps": [1e32],
    "out_dir": ["ppo_noclip_humanoid/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [True],
    "entropy_coeff": [0.005],
    "ppo_lr_adam": [2e-5] * 72,
    "clip_grad_norm": [0.5],
    "val_lr": [5e-5],
    "lambda": [0.85],
    "cpu": [True],
    "advanced_logging": [True],
    "save_iters": [10],
}

generate_configs(BASE_CONFIG, PARAMS)
