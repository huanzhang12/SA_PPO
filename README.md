# SA-PPO: State-adversarial PPO for robust deep reinforcement learning

This repository contains a reference implementation for State-Adversarial
Proximal Policy Optimization (SA-PPO).  SA-PPO includes a theoretically
principled robust KL regularization term based on SA-MDP to obtain a PPO
agent that is robust to noises on state observations, including adversarial
perturbations. See our paper for more details:

*Huan Zhang\*, Hongge Chen\*, Chaowei Xiao, Bo Li, Duane Boning,* and *Cho-Jui
Hsieh*, "Robust Deep Reinforcement Learning against Adversarial Perturbations
on State Observations".  
[**NeurIPS 2020
(Spotlight)**](https://proceedings.neurips.cc/paper/2020/file/f0eb6568ea114ba6e293f903c34d7488-Paper.pdf)
(\*Equal contribution)

**Please also checkout our new work** on **optimal adversary** and the **alternating
training of learned adversary and agent (ATLA)** framework for training a more robust RL agent which achieves better robustness than SA-PPO (code is available in the [ATLA_Robust_RL repository](https://github.com/huanzhang12/ATLA_robust_RL)):

*Huan Zhang\*, Hongge Chen\*, Duane Boning,*
and *Cho-Jui Hsieh*, "Robust Reinforcement Learning on State Observations with Learned Optimal Adversary" (\*Equal contribution)    
[**ICLR 2020**](https://arxiv.org/pdf/2101.08452.pdf) [(Code)](https://github.com/huanzhang12/ATLA_robust_RL)

Our code is based on a carefully implemented PPO baseline:
[MadryLab/implementation-matters](https://github.com/MadryLab/implementation-matters`).
The code also uses the [auto_lirpa](https://github.com/KaidiXu/auto_LiRPA)
library for computing convex relaxations of neural networks, which provides a
wide of range of possibilities for the convex relaxation based method,
including forward and backward mode bound analysis
([CROWN](https://arxiv.org/pdf/1811.00866.pdf)), and interval bound propagation
(IBP).


If you are looking for robust *off-policy* actor-critic algorithms, please see
our [SA-DDPG repository](https://github.com/huanzhang12/SA_DDPG).  If you are
looking for robust DQN (e.g., agents for Atari games) please see our [SA-DQN
repository](https://github.com/chenhongge/SA_DQN).

## SA-PPO Demo

Adversarial attacks on state observations (e.g., position and velocity
measurements) can easily make an agent fail. Our SA-PPO agents are robust
against adversarial attacks, including our strong Robust Sarsa attack.

| ![humanoid_vanilla_ppo_attack_615.gif](/assets/humanoid_vanilla_ppo_attack_615.gif) | ![humanoid_sappo_attack_6042.gif](/assets/humanoid_sappo_attack_6042.gif) | ![walker_vanilla_ppo_attack_620.gif](/assets/walker_vanilla_ppo_attack_620.gif) | ![walker_sappo_attack_5044.gif](/assets/walker_sappo_attack_5044.gif) |
|:--:| :--:| :--:| :--:|
| Humanoid *Vanilla PPO* <br> reward under attack: **615** | Humanoid *SA-PPO* <br> reward under attack: **6042** | Walker2d *Vanilla PPO* <br> reward under attack: **620**  | Walker2d *SA-PPO* <br> reward under attack: **5044** |

## Setup

First clone this repository and install necessary Python packages:

```bash
git submodule update --init
pip install -r requirements.txt
cd src
```

Python 3.7+ is required. Note that you need to install MuJoCo 1.5 first to use
the Gym environments.  See
[here](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/README.md#requirements)
for instructions.

## Pretrained agents

The pretrained agents are packed as `.model` files in the
`models/median-models` folder. We repeatedly train each agent configuration at
least 15 times, and rank them with their average cumulative rewards over 50
episodes under the **strongest** attack (among 5 attacks used).  In our paper
and the table below, we report performance for the median agents:

| Environment | Evaluation                                 | Vanilla PPO | SA-PPO (convex) | SA-PPO (SGLD) |
|-------------|--------------------------------------------|-------------|-----------------|---------------|
| Humanoid-v2 | No attack                                  |   5270.6    |     6400.6      |     6624.0    |
|             | Strongest attack                           |   884.1     |     4690.3      |    **6073.8** |
| Walker2d-v2 | No attack                                  |   4619.5    |     4486.6      |    4911.8     |
|             | Strongest attack                           |   913.7     |     2076.1      |    **2468.4** |
| Hopper-v2   | No attack                                  |   3167.6    |     3704.1      |    3523.1     |
|             | Strongest attack                           |   733       |     1224.2      |    **1403.3** |


Note that reinforcement learning algorithms typically have large variance
across training runs. We report the agents with **median** rewards under
attacks across over multiple training runs. We do not report the best agents
which can be sometimes significantly better than the median, however we do
include these agents in `models/best-models`.  **For a fair comparison to our method it
is important to train each environment multiple times** and show a distribution
or median of results, like the box plots in [our
paper](https://arxiv.org/pdf/2003.08938.pdf).

The pretrained agents can be evaluated using `test.py` (see the next sections
for more usage details). For example,

```bash
# Walker2D agents: vanilla PPO, SA-PPO (convex) and SA-PPO (SGLD)
python test.py --config-path config_walker_vanilla_ppo.json --load-model models/median-models/model-ppo-walker-median.model --deterministic
python test.py --config-path config_walker_robust_ppo_convex.json --load-model models/median-models/model-sappo-convex-walker-median.model --deterministic
python test.py --config-path config_walker_robust_ppo_sgld.json --load-model models/median-models/model-sappo-sgld-walker-median.model --deterministic
# Hopper agents
python test.py --config-path config_hopper_vanilla_ppo.json --load-model models/median-models/model-ppo-hopper-median.model --deterministic
python test.py --config-path config_hopper_robust_ppo_convex.json --load-model models/median-models/model-sappo-convex-hopper-median.model --deterministic
python test.py --config-path config_hopper_robust_ppo_sgld.json --load-model models/median-models/model-sappo-sgld-hopper-median.model --deterministic
# Humanoid agents
python test.py --config-path config_humanoid_vanilla_ppo.json --load-model models/median-models/model-ppo-humanoid-median.model --deterministic
python test.py --config-path config_humanoid_robust_ppo_convex.json --load-model models/median-models/model-sappo-convex-humanoid-median.model --deterministic
python test.py --config-path config_humanoid_robust_ppo_sgld.json --load-model models/median-models/model-sappo-sgld-humanoid-median.model --deterministic
```

## Agent Training

To train a agent, use `run.py` in `src` folder and specify a configuration file
path.  All training hyperparameters are specified in a json configuration file.
Note that for PPO and SA-PPO we use the same set of hyperparameters, and these
hyperparameters are fine-tuned by [Engstrom et
al.](https://github.com/MadryLab/implementation-matters) on PPO rather than
SA-PPO (the only exception is Humanoid, where we found their hyperparameters
were non-optimal so we conducted hyperparameter search for vanilla PPO). For
SA-PPO, we only change one additional parameter (the regularizer, which can be
set via `--robust_ppo_reg`).

Several configuration files are provided in the `src` folder, with filenames
starting with `config`. For example:

Humanoid vanilla PPO training:

```bash
python run.py --config-path config_humanoid_vanilla_ppo.json
```

Humanoid SA-PPO training (solved using SGLD):

```bash
python run.py --config-path config_humanoid_robust_ppo_sgld.json
```

Humanoid SA-PPO training (solved using convex relaxations):

```bash
python run.py --config-path config_humanoid_robust_ppo_convex.json
```

Change `humanoid` to `walker` or `hopper` to run other environments.

Training results will be saved to a directory specified by the `out_dir`
parameter in the json file. For example, for SA-PPO training using SGLD it is
`robust_ppo_sgld_humanoid_best/agents`. To allow multiple runs, each experiment is
assigned a unique experiment ID (e.g., `2fd2da2c-fce2-4667-abd5-274b5579043a`),
which is saved as a folder under `out_dir` (e.g.,
`robust_ppo_sgld_humanoid_best/agents/2fd2da2c-fce2-4667-abd5-274b5579043a`).

Then the agent can be evaluated using `test.py`.  For example:

```bash
# Change the --exp-id to match the folder name in robust_ppo_sgld_humanoid/agents/
python test.py --config-path config_humanoid_robust_ppo_sgld.json --exp-id YOUR_EXP_ID --deterministic
```

You should expect a cumulative reward (mean over 50 episodes) over 6000.

**Tips on training.** All robust training related hyperparameters start with
`--robust-ppo` and can be found in configuration files or overridden in command
line.  Generally, assuming you start with a set of good parameters for vanilla
PPO, and a known perturbation epsilon (set via `--robust-ppo-eps`), for SA-PPO
you don't need to change anything except the `--robust-ppo-reg` parameter.
Using a larger `--robust-ppo-reg` leads to a agent more robust to strong
attacks, but it may reduce agent performance. Our released pretrained agents
seek for a balance between natural performance and adversarial performance
rather than providing for an absolutely robust yet low performance agent.

We use the [auto_lirpa](https://github.com/KaidiXu/auto_LiRPA) library which
provides a wide of range of possibilities for the convex relaxation based
method, including forward and backward mode bound analysis, and interval bound
propagation (IBP).  In our paper, we only used the IBP+backward scheme, which
is efficient and stable, and we did not report results using other possible
relaxations as this is not our main focus. If you are interested in trying other
relaxation schemes, e.g., if you want to use the cheapest IBP methods (at the
cost of potential stability issue), you can try this:

```bash
# The below training should run faster per epoch than original due to the use of cheaper relaxations.
# However, you probably need to train more epochs to compensate the instability of IBP.
python run.py --config-path config_walker_robust_ppo_convex.json --robust-ppo-beta-scheduler-opts "start=0,length=1"
```

## Agent Evaluation Under Attacks

We implemented random attack, critic based attack and our proposed Robust Sarsa
(RS) and maximal action difference (MAD) attacks.  On PPO, our proposed Robust
Sarsa (RS) attack typically performs best.

### Robust Sarsa (RS) Attack

In our Robust Sarsa attack, we first learn a *robust* value function for the
policy under evaluation. Then, we attack the policy using this robust value
function. The first step for RS attack is to train a robust value function:

```bash
# Step 1:
python test.py --config-path config_humanoid_vanilla_ppo.json --load-model models/median-models/model-ppo-humanoid-median.model --sarsa-enable --sarsa-model-path sarsa_humanoid_vanilla.model
```

The above training step is usually very fast (e.g., a few minutes).  The value
function will be saved in `sarsa_humanoid_vanilla.model`. Then it can be used for
attack:

```bash
# Step 2:
python test.py --config-path config_humanoid_vanilla_ppo.json --load-model models/median-models/model-ppo-humanoid-median.model --attack-eps=0.075 --attack-method sarsa --attack-sarsa-network sarsa_humanoid_vanilla.model --deterministic
```

The reported mean reward over 50 episodes should be around 1000 (reward
without attack is over 5000). In contrast, our SA-PPO robust agent has a reward
of over 4000 even under attack:

```bash
python test.py --config-path config_humanoid_robust_ppo_convex.json --load-model models/median-models/model-sappo-convex-humanoid-median.model --sarsa-enable --sarsa-model-path sarsa_humanoid_convex.model
python test.py --config-path config_humanoid_robust_ppo_convex.json --load-model models/median-models/model-sappo-convex-humanoid-median.model --attack-eps=0.075 --attack-method sarsa --attack-sarsa-network sarsa_humanoid_convex.model --deterministic
```

The Robust Sarsa attack has two hyperparameters for robustness regularization
(`--sarsa-eps` and `--sarsa-reg`) to build the robust value function.  Although
the default settings generally work well, for a comprehensive robustness
evaluation it is recommended to run Robust Sarsa attack under different
hyperparameters and choose the best attack (the lowest reward) as the final result.
We provide a script, `scan_attacks.sh` for the purpose of comprehensive
adversarial evaluation:

```bash
source scan_attacks.sh
# Usage: scan_attacks model_path config_path output_dir_path
scan_attacks models/median-models/model-sappo-convex-humanoid-median.model config_humanoid_robust_ppo_convex.json sarsa_humanoid_sappo-convex
```

Note: the learning rate of the Sarsa model can be changed by `--val-lr`. The
default value should be good for attacks the provided environments (with
normalized reward). However, if you want to use this attack on a different
environment, this learning rate can be important as the reward maybe
unnormalized (some environment returns large rewards so the Q values are
larger, and a larger `--val-lr` is needed). The rule of thumb is to always
checking the training logs of these Sarsa models - make sure the Q loss has
been reduced sufficiently (close to 0) at the end of training.

### Maximal Action Difference (MAD) Attack

We additionally propose a maximal action difference (MAD) attack where we
attempt to maximize the KL divergence between original action and perturbed
action. It can be invoked by setting `--attack-method` to `action`. For
example:

```bash
python test.py --config-path config_humanoid_vanilla_ppo.json --load-model models/median-models/model-ppo-humanoid-median.model --attack-eps=0.075 --attack-method action --deterministic
```

The reported mean reward over 50 episodes should be around 1000.  In contrast,
our SA-PPO trained agent is more resistant to MAD attack, achieving a reward
around 6000.

```bash
python test.py --config-path config_humanoid_robust_ppo_convex.json --load-model models/median-models/model-sappo-convex-humanoid-median.model --attack-eps=0.075 --attack-method action --deterministic
```

We additionally provide a combined attack of RS+MAD, which can be invoked by
setting `--attack-method` to `sarsa+action`, and the combination ratio can be
set via `--attack-sarsa-action-ratio`. Generally, we first obtain the best RS
attack model and then further enhance it by adding an action attack.

### Critic based attack and random attack

Critic based attack and random attack can be used by setting `--attack-method`
to `critic` and `random`, respectively.  These attacks are relatively weak and
not suitable for evaluating the robustness of PPO agents.

```bash
python test.py --config-path config_humanoid_vanilla_ppo.json --load-model models/median-models/model-ppo-humanoid-median.model --attack-eps=0.075 --attack-method random --deterministic
```

