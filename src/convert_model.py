import torch
import copy
import json
import argparse
import numpy as np
from policy_gradients.custom_env import Env

parser = argparse.ArgumentParser(description='Convert DDPG model to PPO model')
# python convert_model.py --ddpg-model walker-ddpg.model --ddpg-json Walker2d_robust.json
parser.add_argument('--ddpg-model', type=str, required=True,
                    help='path to ddpg model')
parser.add_argument('--ddpg-json', type=str, required=True,
                    help='path to ddpg json')
parser.add_argument('--output', type=str, default='output.model',
                    help='path to output model')
args = parser.parse_args()

ddpg_model = torch.load(args.ddpg_model)
ddpg_json = json.load(open(args.ddpg_json, "r"))
ddpg_std = np.array(ddpg_json['data_config']['state_std'])
print(ddpg_model.keys())
print(ddpg_std)
env = Env(ddpg_json['env_id'], norm_states=True, norm_rewards=None, params={}, clip_obs=-1, clip_rew=-1)

new_model = {}
new_model['policy_model'] = {}
# Copy parameters
new_model['policy_model']['affine_layers.0.weight'] = ddpg_model['fc_action.0.weight'].cpu().clone() * torch.tensor(ddpg_std)
new_model['policy_model']['affine_layers.0.bias'] = ddpg_model['fc_action.0.bias'].cpu().clone()
new_model['policy_model']['affine_layers.1.weight'] = ddpg_model['fc_action.2.weight'].cpu().clone()
new_model['policy_model']['affine_layers.1.bias'] = ddpg_model['fc_action.2.bias'].cpu().clone()
new_model['policy_model']['final_mean.weight'] = ddpg_model['fc_action.4.weight'].cpu().clone()
new_model['policy_model']['final_mean.bias'] = ddpg_model['fc_action.4.bias'].cpu().clone()
# No noise
new_model['policy_model']['log_stdev'] = torch.ones_like(new_model['policy_model']['final_mean.bias']) * -100
print(new_model['policy_model'].keys())
env.reset()
env.state_filter.rs._M[:] = 0.0
env.state_filter.rs._S[:] = ddpg_std * ddpg_std
env.state_filter.rs._n = 2
print(env.state_filter.rs.std)
# Environment
new_model['envs']=[env]
torch.save(new_model, args.output)

