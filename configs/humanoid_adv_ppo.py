import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Humanoid-v2"],
    "mode": ["ppo"],
    "out_dir": ["adv_ppo_humanoid/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [1.5e-4]*32,
    "val_lr": [1e-4],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [100],
    "train_steps": [4882],
    "attack_method": ["critic", "action"],
    "attack_ratio": [0.5, 1.0],
    "attack_eps": [0.075],
}

generate_configs(BASE_CONFIG, PARAMS)
