import os
import sys
proj_root_dir = os.path.dirname(os.path.realpath('.'))   # project root dir
if proj_root_dir not in sys.path:
    sys.path.insert(0,proj_root_dir)
######################
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import copy
import garage
from dowel import logger,tabular
from garage import wrap_experiment, obtain_evaluation_episodes, StepType
from garage.experiment import Snapshotter, deterministic
from garage.torch import NonLinearity,set_gpu_mode,global_device
from garage.trainer import Trainer
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from torch.nn import functional as F
import cloudpickle
from garage import rollout
from tqdm import tqdm

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#region arguments
# Define agents available
AGENTS = {"SAC", "TD3"}

parser = argparse.ArgumentParser(description='RL args using agents / algs from garage and envs from robosuite')

def add_robosuite_args():
    """
    Adds robosuite args to command line arguments list
    """
    parser.add_argument(
        '--env',
        type=str,
        default='Lift',
        help='Robosuite env to run test on')
    parser.add_argument(
        '--robots',
        nargs="+",
        type=str,
        default='Panda',
        help='Robot(s) to run test with')
    parser.add_argument(
        '--eval_horizon',
        type=int,
        default=500,
        help='max num of timesteps for each eval simulation')
    parser.add_argument(
        '--expl_horizon',
        type=int,
        default=500,
        help='max num of timesteps for each eval simulation')
    parser.add_argument(
        '--policy_freq',
        type=int,
        default=20,
        help='Policy frequency for environment (Hz)')
    parser.add_argument(
        '--controller',
        type=str,
        default="OSC_POSE",
        help='controller to use for robot environment. Either name of controller for default config or filepath to custom'
             'controller config')
    parser.add_argument(
        '--reward_scale',
        type=float,
        default=1.0,
        help='max reward from single environment step'
    )
    parser.add_argument(
        '--hard_reset',
        action="store_true",
        help='If set, uses hard resets for this env'
    )

    # Environment-specific arguments
    parser.add_argument(
        '--env_config',
        type=str,
        default=None,
        choices=['single-arm-parallel', 'single-arm-opposed', 'bimanual'],
        help='Robosuite env configuration. Only necessary for bimanual environments')
    parser.add_argument(
        '--prehensile',
        type=str,
        default=None,
        choices=["True", "False", "true", "false"],
        help='Whether to use prehensile config. Only necessary for TwoArmHandoff env'
    )

def add_agent_args():
    """
    Adds args necessary to define a general agent and trainer in rlkit
    """
    parser.add_argument(
        '--agent',
        type=str,
        default="SAC",
        choices=AGENTS,
        help='Agent to use for training')
    parser.add_argument(
        '--qf_hidden_sizes',
        nargs="+",
        type=int,
        default=[256, 256],
        help='Hidden sizes for Q network ')
    parser.add_argument(
        '--policy_hidden_sizes',
        nargs="+",
        type=int,
        default=[256, 256],
        help='Hidden sizes for policy network ')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor')
    parser.add_argument(
        '--policy_lr',
        type=float,
        default=3e-4,
        help='Learning rate for policy')
    parser.add_argument(
        '--qf_lr',
        type=float,
        default=3e-4,
        help='Quality function learning rate')

    # SAC-specific
    parser.add_argument(
        '--soft_target_tau',
        type=float,
        default=5e-3,
        help='Soft Target Tau value for Value function updates')
    parser.add_argument(
        '--target_update_period',
        type=int,
        default=1,
        help='Number of steps between target updates')
    parser.add_argument(
        '--no_auto_entropy_tuning',
        action='store_true',
        help='Whether to automatically tune entropy or not (default is ON)')

    # TD3-specific
    parser.add_argument(
        '--target_policy_noise',
        type=float,
        default=0.2,
        help='Target noise for policy')
    parser.add_argument(
        '--policy_and_target_update_period',
        type=int,
        default=2,
        help='Number of steps between policy and target updates')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='Tau value for training')

def add_training_args():
    """
    Adds training parameters used during the experiment run
    """
    parser.add_argument(
        '--variant',
        type=str,
        default=None,
        help='If set, will use stored configuration from the specified filepath (should point to .json file)')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=2000,
        help='Number of epochs to run')
    parser.add_argument(
        '--trains_per_train_loop',
        type=int,
        default=1000,
        help='Number of training steps to take per training loop')
    parser.add_argument(
        '--expl_ep_per_train_loop',
        type=int,
        default=10,
        help='Number of exploration episodes to take per training loop')
    parser.add_argument(
        '--steps_before_training',
        type=int,
        default=1000,
        help='Number of exploration steps to take before starting training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size per training step')
    parser.add_argument(
        '--num_eval',
        type=int,
        default=10,
        help='Num eval episodes to run for each trial run')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../log/runs/',
        help='directory to save runs')


def parse_cmd_line():

    # experiment args
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='gpu id or -1 for cpu')
    parser.add_argument(
        '--variant',
        type=str,
        default=None,
        help='If set, will use stored configuration from the specified filepath (should point to .json file)')

    add_robosuite_args()

    add_agent_args()

    add_training_args()

    args = parser.parse_args()
    return args
#endregion


@wrap_experiment(snapshot_mode='last')
def robosuite_benchmark(ctxt=None):
    # args = parse_cmd_line()
    if args.gpuid>=0 and torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=args.gpuid)
    else:
        set_gpu_mode(False)
    device = global_device()
    # set the seed
    deterministic.set_seed(args.seed)




    trainer = Trainer(snapshot_config=ctxt)

    # todo : define the robosuite env
    env_params = {}
    env = create_robosuite_env(env_params)


    # todo: define the agent according to the setup in
    #  https://github.com/ARISE-Initiative/robosuite-benchmark

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-5.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(2.5e6))

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=1000,
              max_episode_length_eval=1000,
              replay_buffer=replay_buffer,
              policy_lr=1e-4,
              qf_lr=1e-4,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=1024,
              reward_scale=1.,
              steps_per_epoch=1)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=args.n_epochs, batch_size=1000,store_episodes=False)






if __name__ == '__main__':
    args=parse_cmd_line()

    # options can be a dict whose keys can be subset of:
    # name = self.name,
    # function = self.function,
    # prefix = self.prefix,
    # name_parameters = self.name_parameters,
    # log_dir = self.log_dir,
    # archive_launch_repo = self.archive_launch_repo,
    # snapshot_gap = self.snapshot_gap,
    # snapshot_mode = self.snapshot_mode,
    # use_existing_dir = self.use_existing_dir,
    # x_axis = self.x_axis,
    # signature = self.__signature__
    options = {'name':'robosuite_bm'}
    robosuite_benchmark(options)

    sys.path.remove(proj_root_dir)
