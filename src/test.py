import pickle
from policy_gradients.agent import Trainer
import git
import numpy as np
import os
import copy
import random
import argparse
from policy_gradients import models
from policy_gradients.torch_utils import ZFilter
import sys
import json
import torch
import torch.optim as optim
from cox.store import Store, schema_from_dict
from run import main, add_common_parser_opts, override_json_params
from auto_LiRPA.eps_scheduler import LinearScheduler


def main(params):
    override_params = copy.deepcopy(params)
    excluded_params = ['config_path', 'out_dir_prefix', 'num_episodes', 'row_id', 'exp_id',
            'load_model', 'seed', 'deterministic', 'scan_config', 'compute_kl_cert', 'use_full_backward']
    sarsa_params = ['sarsa_enable', 'sarsa_steps', 'sarsa_eps', 'sarsa_reg', 'sarsa_model_path']
    # original_params contains all flags in config files that are overridden via command.
    for k in list(override_params.keys()):
        if k in excluded_params:
            del override_params[k]

    # Append a prefix for output path.
    if params['out_dir_prefix']:
        params['out_dir'] = os.path.join(params['out_dir_prefix'], params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")

    if params['config_path']:
        # Load from a pretrained model using existing config.
        # First we need to create the model using the given config file.
        json_params = json.load(open(params['config_path']))
        
        params = override_json_params(params, json_params, excluded_params + sarsa_params)

    if params['sarsa_enable']:
        assert params['attack_method'] == "none" or params['attack_method'] is None, \
                "--train-sarsa is only available when --attack-method=none, but got {}".format(params['attack_method'])

    if 'load_model' in params and params['load_model']:
        for k, v in zip(params.keys(), params.values()):
            assert v is not None, f"Value for {k} is None"

        # Create the agent from config file.
        p = Trainer.agent_from_params(params, store=None)
        print('Loading pretrained model', params['load_model'])
        pretrained_model = torch.load(params['load_model'])
        if 'policy_model' in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model['policy_model'])
        if 'val_model' in pretrained_model:
            p.val_model.load_state_dict(pretrained_model['val_model'])
        if 'policy_opt' in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model['policy_opt'])
        if 'val_opt' in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model['val_opt'])
        # Restore environment parameters, like mean and std.
        if 'envs' in pretrained_model:
            p.envs = pretrained_model['envs']
        for e in p.envs:
            e.normalizer_read_only = True
            e.setup_visualization(params['show_env'], params['save_frames'], params['save_frames_path'])
    else:
        # Load from experiment directory. No need to use a config.
        base_directory = params['out_dir']
        store = Store(base_directory, params['exp_id'], mode='r')
        if params['row_id'] < 0:
            row = store['final_results'].df
        else:
            checkpoints = store['checkpoints'].df
            row_id = params['row_id']
            row = checkpoints.iloc[row_id:row_id+1]
        print("row to test: ", row)
        if params['cpu'] == None:
            cpu = False
        else:
            cpu = params['cpu']
        p, _ = Trainer.agent_from_data(store, row, cpu, extra_params=params, override_params=override_params, excluded_params=excluded_params)
        store.close()

    rewards = []

    if params['sarsa_enable']:
        num_steps = params['sarsa_steps']
        # learning rate scheduler: linearly annealing learning rate after 
        lr_decrease_point = num_steps * 2 / 3
        decreasing_steps = num_steps - lr_decrease_point
        lr_sch = lambda epoch: 1.0 if epoch < lr_decrease_point else (decreasing_steps - epoch + lr_decrease_point) / decreasing_steps
        # robust training scheduler. Currently using 1/3 epochs for warmup, 1/3 for schedule and 1/3 for final training.
        eps_start_point = int(num_steps * 1 / 3)
        robust_eps_scheduler = LinearScheduler(params['sarsa_eps'], f"start={eps_start_point},length={eps_start_point}")
        robust_beta_scheduler = LinearScheduler(1.0, f"start={eps_start_point},length={eps_start_point}")
        # reinitialize value model, and run value function learning steps.
        p.setup_sarsa(lr_schedule=lr_sch, eps_scheduler=robust_eps_scheduler, beta_scheduler=robust_beta_scheduler)
        # Run Sarsa training.
        for i in range(num_steps):
            print(f'Step {i+1} / {num_steps}, lr={p.sarsa_scheduler.get_last_lr()}')
            mean_reward = p.sarsa_step()
            rewards.append(mean_reward)
            # for w in p.val_model.parameters():
            #     print(f'{w.size()}, {torch.norm(w.view(-1), 2)}')
        # Save Sarsa model.
        saved_model = {
                'state_dict': p.sarsa_model.state_dict(),
                'metadata': params,
                }
        torch.save(saved_model, params['sarsa_model_path'])
    else:
        print('Gaussian noise in policy:')
        print(torch.exp(p.policy_model.log_stdev))
        if params['deterministic']:
            print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
            p.policy_model.log_stdev.data[:] = -100
        num_episodes = params['num_episodes']
        all_rewards = []
        all_lens = []
        all_kl_certificates = []
        
        for i in range(num_episodes):
            print('Episode %d / %d' % (i+1, num_episodes))
            ep_length, ep_reward, actions, action_means, states, kl_certificates = p.run_test(compute_bounds=params['compute_kl_cert'], use_full_backward=params['use_full_backward'])
            if i == 0:
                all_actions = actions.copy()
                all_states = states.copy()
            else:
                all_actions = np.concatenate((all_actions, actions), axis=0) 
                all_states = np.concatenate((all_states, states), axis=0)
            if params['compute_kl_cert']:
                print('Epoch KL certificates:', kl_certificates)
                all_kl_certificates.append(kl_certificates)
            all_rewards.append(ep_reward)
            all_lens.append(ep_length)
    
        attack_dir = 'attack-{}-eps-{}'.format(params['attack_method'], params['attack_eps'])
        if 'sarsa' in params['attack_method']:
            attack_dir += '-sarsa_steps-{}-sarsa_eps-{}-sarsa_reg-{}'.format(params['sarsa_steps'], params['sarsa_eps'], params['sarsa_reg'])
            if 'action' in params['attack_method']:
                attack_dir += '-attack_sarsa_action_ratio-{}'.format(params['attack_sarsa_action_ratio'])
        save_path = os.path.join(params['out_dir'], params['exp_id'], attack_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for name, value in [('actions',all_actions), ('states', all_states), ('rewards', all_rewards), ('length', all_lens)]:
            with open(os.path.join(save_path, '{}.pkl'.format(name)), 'wb') as f:
                pickle.dump(value, f)
        print(params)
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        print('\n')
        print('all rewards:', all_rewards) 
        print('rewards stats:\nmean: {}, std:{}, min:{}, max:{}'.format(np.mean(all_rewards), np.std(all_rewards), np.min(all_rewards), np.max(all_rewards)))
        if params['compute_kl_cert']:
            print('KL certificates stats: mean: {}, std: {}, min: {}, max: {}'.format(np.mean(all_kl_certificates), np.std(all_kl_certificates), np.min(all_kl_certificates), np.max(all_kl_certificates)))
          

def get_parser():
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, default='', required=False,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default='', required=False,
                        help='prefix for output log path')
    parser.add_argument('--exp-id', type=str, help='experiement id for testing', default='')
    parser.add_argument('--row-id', type=int, help='which row of the table to use', default=-1)
    parser.add_argument('--num-episodes', type=int, help='number of episodes for testing', default=50)
    parser.add_argument('--compute-kl-cert', action='store_true', help='compute KL certificate')
    parser.add_argument('--use-full-backward', action='store_true', help='Use full backward LiRPA bound for computing certificates')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for evaluation')
    parser.add_argument('--load-model', type=str, help='load a pretrained model file', default='')
    parser.add_argument('--seed', type=int, help='random seed', default=1234)
    # Sarsa training related options.
    parser.add_argument('--sarsa-enable', action='store_true', help='train a sarsa attack model.')
    parser.add_argument('--sarsa-steps', type=int, help='Sarsa training steps.', default=30)
    parser.add_argument('--sarsa-model-path', type=str, help='path to save the sarsa value network.', default='sarsa.model')
    parser.add_argument('--sarsa-eps', type=float, help='eps for actions for sarsa training.', default=0.3)
    parser.add_argument('--sarsa-reg', type=float, help='regularization term for sarsa training.', default=0.1)
    # Other configs
    parser.add_argument('--scan-config', type=str, default=None)
    parser = add_common_parser_opts(parser)
    
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.load_model:
        assert args.config_path, "Need to specificy a config file when loading a pretrained model."
    params = vars(args)
    seed = params['seed']
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
 
    main(params)

