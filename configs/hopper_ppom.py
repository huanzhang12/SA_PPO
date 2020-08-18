import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Hopper-v2"],
    "mode": ["ppo"],
    "out_dir": ["ppom_hopper/agents"],
    "norm_rewards": ["none"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "ppo_lr_adam": [1.7e-4] * 40,
    "val_lr": [4e-4],
    "clip_rewards": [-1],
    "clip_observations" : [-1],
    "cpu": [True],
    "advanced_logging": [True],
    "save_iters": [10]
}

generate_configs(BASE_CONFIG, PARAMS)
