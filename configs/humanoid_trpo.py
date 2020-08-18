import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2"],
    "mode": ["trpo"],
    "out_dir": ["trpo_humanoid/agents"],
    "norm_rewards": ["none"],
    "initialization": ["xavier"],
    "anneal_lr": [False],
    "value_clipping": [False],
    "max_kl": [0.07] * 40,
    "max_kl_final": [0.07],
    "val_lr": [3e-4],
    "clip_rewards": [-1],
    "clip_observations" : [-1],
    "cpu": [True],
    "advanced_logging": [True],
    "save_iters": [10]
}

generate_configs(BASE_CONFIG, PARAMS)
