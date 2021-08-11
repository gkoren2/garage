#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import os
import sys
proj_root_dir = os.path.dirname(os.path.realpath('.'))   # project root dir
if proj_root_dir not in sys.path:
    sys.path.insert(0,proj_root_dir)
######################

import argparse
import numpy as np
import pickle as pkl
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from dowel import logger,tabular

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from _labexp.my_envs.walker2d_rand_params import Walker2DRandParamsEnv

parser = argparse.ArgumentParser(description='RL args using agents / algs from garage and walker2d env')
TASKS_FILE_NAME = 'walker2d_rand_tasks.pkl'
DEFAULT_N_TASKS = 100

def parse_cmd_line():

    # experiment args
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='gpu id or -1 for cpu')
    parser.add_argument(
        '--tasks',
        type=int,  # if >0, generate new tasks. else, load from file
        default=-1,     # try to load from file. if not existm throw error
        help='If set, will generate new tasks and save to pkl file. else, will load from file')

    parser.add_argument(
        '--tid',
        type=int,  # if >0, select this task from the array
        help='If not set, will choose randomly')

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=500,
        help='number of epochs')


    parser.add_argument(
        '--log_dir',
        type=str,
        default='.',
        help='directory to save runs')

    args = parser.parse_args()
    return args



@wrap_experiment(snapshot_mode='gap',snapshot_gap=20)
def sac_walker2d_rand(ctxt=None, task_id=1, num_epochs=500,
                         gpuid=0,seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    task=tasks[task_id]
    logger.log(f'task {task_id} parameters: {task}')

    env = Walker2DRandParamsEnv()
    env.set_task(task)
    env = normalize(GymEnv(env, max_episode_length=200),
                    expected_action_scale=1.)


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

    replay_buffer = PathBuffer(capacity_in_transitions=int(5e5))

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

    if torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=gpuid)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=num_epochs, batch_size=1000,store_episodes=True)


if __name__ == '__main__':
    args=parse_cmd_line()
    n_tasks_to_generate=0
    if args.tasks<=0:
        # try to load from file
        try:
            with open(TASKS_FILE_NAME,'rb') as f:
                tasks=pkl.load(f)
            print(f'loaded {len(tasks)} tasks from pkl')
        except FileNotFoundError:
            print(f'Error opening task file at {TASKS_FILE_NAME}')
            n_tasks_to_generate = DEFAULT_N_TASKS
    else:
        n_tasks_to_generate = args.tasks

    if n_tasks_to_generate>0:
        print(f'generating {n_tasks_to_generate} walker2d random tasks')
        env = Walker2DRandParamsEnv()
        tasks = env.sample_tasks(n_tasks_to_generate)
        # save to file
        with open(TASKS_FILE_NAME,'wb') as f:
            pkl.dump(tasks,f)
        print(f'tasks saved to {TASKS_FILE_NAME}')
    # set log dir
    seed = args.seed
    if args.tid and args.tid>=0 and args.tid<len(tasks):
        print(f'running on task {args.tid}')
        task_id=args.tid
    else:
        task_id = np.random.randint(len(tasks))
        print(f'task id was randomly set to {task_id}')

    experiment_name = f'sac_walker2d-ID{task_id}-S{seed}'
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # set the options and call the function:
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
    options = {'name':experiment_name}
    sac_walker2d_rand(options,task_id=task_id,num_epochs=args.n_epochs,
                      seed=seed,gpuid=args.gpuid)

    sys.path.remove(proj_root_dir)
