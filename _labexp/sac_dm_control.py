#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import click

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.dm_control import DMControlEnv
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

@click.command()
@click.option('--num_epochs', default=500)
@click.option('--gpuid', default=0)
@click.option('--domain_name', type=str,default='cartpole')
@click.option('--task_name', type=str,default='swingup')
@click.option('--seed',default=1)


@wrap_experiment(snapshot_mode='gap',snapshot_gap=20)
def sac_dm_control(ctxt=None, domain_name='cartpole',task_name='swingup',
                   num_epochs=500, gpuid=0,seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)


    env = DMControlEnv.from_suite(domain_name, task_name)

    # env = normalize(GymEnv(HalfCheetahVelEnv(task), max_episode_length=200),
    #                 expected_action_scale=1.)


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

    if torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=gpuid)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=num_epochs, batch_size=1000,store_episodes=True)


sac_dm_control()
