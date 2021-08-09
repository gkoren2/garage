# Note : this file was created with the help of the robosuite-benchmark
# train.py and arguments.py script
# and by comparing garage.SAC to rlkit.SAC
import copy
import os
import sys
proj_root_dir = os.path.dirname(os.path.realpath('.'))   # project root dir
if proj_root_dir not in sys.path:
    sys.path.insert(0,proj_root_dir)
######################
import argparse
import json
import torch
import garage
from dowel import logger,tabular
from garage.envs import GymEnv, normalize
from garage import wrap_experiment, obtain_evaluation_episodes, StepType
from garage.experiment import Snapshotter, deterministic
from garage.torch import NonLinearity,set_gpu_mode,global_device
from garage.trainer import Trainer
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from _labexp.my_envs.robosuite_env import GrgGymWrapper,RobosuiteEnvGrg


from robosuite.controllers import load_controller_config, ALL_CONTROLLERS


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#region arguments
# Define agents available
AGENTS = {"SAC", "TD3"}

parser = argparse.ArgumentParser(description='RL args using agents / algs from garage and envs from robosuite')

BOOL_MAP = {
    "true" : True,
    "false" : False
}


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


def get_train_env_kwargs():
    """
    Grabs the robosuite-specific arguments and converts them into an
    rlkit-compatible dict for exploration env
    """
    env_kwargs = dict(
        env_name=args.env,
        robots=args.robots,
        horizon=args.expl_horizon,
        control_freq=args.policy_freq,
        controller=args.controller,
        reward_scale=args.reward_scale,
        hard_reset=args.hard_reset,
        ignore_done=True,
    )

    # Add in additional ones that may not always be specified
    if args.env_config is not None:
        env_kwargs["env_configuration"] = args.env_config
    if args.prehensile is not None:
        env_kwargs["prehensile"] = BOOL_MAP[args.prehensile.lower()]

    # Lastly, return the dict
    return env_kwargs


def get_eval_env_kwargs():
    """
    Grabs the robosuite-specific arguments and converts them into an rlkit-compatible dict for evaluation env
    """
    env_kwargs = dict(
        env_name=args.env,
        robots=args.robots,
        horizon=args.eval_horizon,
        control_freq=args.policy_freq,
        controller=args.controller,
        reward_scale=1.0,
        hard_reset=args.hard_reset,
        ignore_done=True,
    )

    # Add in additional ones that may not always be specified
    if args.env_config is not None:
        env_kwargs["env_configuration"] = args.env_config
    if args.prehensile is not None:
        env_kwargs["prehensile"] = BOOL_MAP[args.prehensile.lower()]

    # Lastly, return the dict
    return env_kwargs


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
        '--discount',
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
    parser.add_argument(
        '--buffer_batch_size',
        type=int,
        default=256,
        help='Batch size sampled from the buffer per training step')
    parser.add_argument(
        '--trains_per_train_loop',
        type=int,
        default=1000,
        help='Number of training steps to take per training loop')
    parser.add_argument(
        '--steps_before_training',
        type=int,
        default=1000,
        help='Number of exploration steps to take before starting training')
    parser.add_argument(
        '--num_eval',
        type=int,
        default=10,
        help='Num eval episodes to run for each trial run')


    # SAC-specific
    parser.add_argument(
        '--soft_target_tau',
        type=float,
        default=5e-3,
        help='Soft Target Tau value for Value function updates')
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
        '--n_epochs',
        type=int,
        default=2000,
        help='Number of epochs to run')
    parser.add_argument(
        '--expl_ep_per_train_loop',
        type=int,
        default=10,
        help='Number of exploration episodes to take per training loop')
    parser.add_argument(
        '--store_episodes',
        action='store_true',
        help='Whether to store the replay buffer in checkpoints')


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
    parser.add_argument(
        '--log_dir',
        type=str,
        default='.',
        help='directory to save runs')

    add_robosuite_args()

    add_agent_args()

    add_training_args()

    args = parser.parse_args()
    return args

def create_variant():
    '''
    parse the command line argument to create the variant
    Returns: dictionary
    '''
    variant = dict(
        algorithm=args.agent,
        seed=args.seed,
        replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=args.qf_hidden_sizes,
        ),
        policy_kwargs=dict(
            hidden_sizes=args.policy_hidden_sizes,
        ),
        algo_kwargs=dict(
            max_episode_length_eval= args.eval_horizon,
            gradient_steps_per_itr = args.trains_per_train_loop,
            fixed_alpha=None if args.no_auto_entropy_tuning else 1.,
            discount=args.discount,
            buffer_batch_size=args.buffer_batch_size,
            min_buffer_size=args.steps_before_training,
            target_update_tau=args.soft_target_tau,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=args.reward_scale,
            num_evaluation_episodes=args.num_eval,
        ),
        train_kwargs=dict(
            n_epochs=args.n_epochs,
            batch_size=args.expl_horizon * args.expl_ep_per_train_loop,
            store_episodes=args.store_episodes
        ),
        train_environment_kwargs=get_train_env_kwargs(),
        eval_environment_kwargs=get_eval_env_kwargs(),
    )

    return variant


#endregion


@wrap_experiment(snapshot_mode='last')
def robosuite_benchmark(ctxt=None,variant=None):
    # args = parse_cmd_line()
    if args.gpuid>=0 and torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=args.gpuid)
    else:
        set_gpu_mode(False)
    device = global_device()
    # set the seed
    deterministic.set_seed(args.seed)
    if variant is None:
        logger.log("ERROR ! Variant is empty. aborting")
        return


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create the robosuite env
    suites=[]
    for env_config in (variant['train_environment_kwargs'],variant['eval_environment_kwargs']):
        # load controller
        suites.append(RobosuiteEnvGrg(**env_config,
                                      has_renderer=False,
                                      has_offscreen_renderer=False,
                                      use_object_obs=True,
                                      use_camera_obs=False,
                                      reward_shaping=True,
                                      ))

        # currently assume same env for train and eval
    env = GymEnv(suites[0],max_episode_length=suites[0].horizon)
    env = normalize(env)
    # env = train_env = normalize(GymEnv(GrgGymWrapper(suites[0])))
    eval_env = GymEnv(suites[1],
                      max_episode_length=suites[1].horizon)
    eval_env = normalize(eval_env)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create the agent

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        **variant['policy_kwargs']
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,**variant['qf_kwargs'])

    qf2 = ContinuousMLPQFunction(env_spec=env.spec, **variant['qf_kwargs'])

    replay_buffer = PathBuffer(capacity_in_transitions=variant['replay_buffer_size'])

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           n_workers=8,
                           worker_class=FragmentWorker)
    # currently supporing only SAC.
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              replay_buffer=replay_buffer,
              eval_env=eval_env,
              **variant['algo_kwargs'])
    sac.to()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create the trainer
    trainer = Trainer(snapshot_config=ctxt)
    # setup and train
    trainer.setup(algo=sac, env=env)
    trainer.train(**variant['train_kwargs'])

    logger.log('Done.')






if __name__ == '__main__':
    args=parse_cmd_line()

    if args.variant:
        try:
            with open(args.variant) as f:
                variant=json.load(f)
        except FileNotFoundError:
            print(f'Error opening variant at {args.variant}')
    else:
        variant = create_variant()
    variant = variant.get('variant',variant)
    # set log dir
    env_name = variant['train_environment_kwargs']['env_name']
    robots = variant['train_environment_kwargs']['robots']
    controller = variant['train_environment_kwargs']['controller']
    seed = args.seed
    experiment_name = f'robosuite_bm-{env_name}-{robots}-{controller}-S{seed}'
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
    robosuite_benchmark(options,variant=variant)

    sys.path.remove(proj_root_dir)
