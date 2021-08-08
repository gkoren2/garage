# Note : this file was created with the help of the robosuite-benchmark
# rollout.py and arguments.py script
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
import time
from signal import signal, SIGINT
from sys import exit
import imageio
import torch
import numpy as np
import garage
from dowel import logger,tabular
from garage.envs import GymEnv, normalize
from garage.experiment import Snapshotter, deterministic
from garage.torch import NonLinearity,set_gpu_mode,global_device
from garage.np import stack_tensor_dict_list
from _labexp.my_envs.robosuite_env import GrgGymWrapper,RobosuiteEnvGrg


from robosuite.controllers import load_controller_config, ALL_CONTROLLERS


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#region arguments

parser = argparse.ArgumentParser(description='RL args using agents / algs from garage and envs from robosuite')

def add_robosuite_args():
    """
    Adds robosuite args to command line arguments list
    """
    parser.add_argument(
        '--env',
        type=str,
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

def add_rollout_args():
    """
    Adds rollout arguments needed for evaluating / visualizing a trained rlkit policy
    """
    parser.add_argument(
        '--policy_dir',
        type=str,
        nargs='+',    # folder and iteration number, if iteration not provided, take last
        required=True,
        help='path to the policy snapshot directory folder')

    parser.add_argument(
        '--env_dir',
        type=str,
        help='path to the folder where we can extract env spec from variant')

    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='Num rollout episodes to run')
    parser.add_argument(
        '--horizon',
        type=int,
        default=None,
        help='Horizon to use for rollouts (overrides default if specified)')
    parser.add_argument(
        '--camera',
        type=str,
        default='frontview',
        help='Name of camera for visualization')
    parser.add_argument('-r',
        '--record_video',
        action='store_true',
        help='If set, will save video of rollouts')


def parse_cmd_line():

    # experiment args
    parser.add_argument('--gpuid', type=int, default=0,
                        help='gpu id or -1 for cpu')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='.',
        help='directory to save runs')

    add_rollout_args()

    add_robosuite_args()

    args = parser.parse_args()
    return args



#endregion

# Define callbacks
video_writer = None

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing video writer and exiting gracefully')
    video_writer.close()
    exit(0)

# Tell Python to run the handler() function when SIGINT is recieved
signal(SIGINT, handler)

def rollout(env,policy,max_path_length=np.inf,render=False,render_kwargs=None,
            video_writer=None):
    return dict()

def rollout_ex (env,
                agent,
                *,
                max_episode_length=np.inf,
                animated=False,         # whether to render or not
                pause_per_frame=None,
                deterministic=False,
                video_writer=None,
                render_kwargs=None):
    """Sample a single episode of the agent in the environment.

    Args:
        agent (Policy): Policy used to select actions.
        env (Environment): Environment to perform actions in.
        max_episode_length (int): If the episode reaches this many timesteps,
            it is truncated.
        animated (bool): If true, render the environment after each step.
        pause_per_frame (float): Time to sleep between steps. Only relevant if
            animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Returns:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape
                :math:`(T + 1, S^*)`, i.e. the unflattened observation space of
                    the current environment.
            * actions(np.array): Non-flattened array of actions. Should have
                shape :math:`(T, S^*)`, i.e. the unflattened action space of
                the current environment.
            * rewards(np.array): Array of rewards of shape :math:`(T,)`, i.e. a
                1D array of length timesteps.
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(np.array): Array of termination signals.

    """
    if render_kwargs is None:
        render_kwargs = {}
    env_steps = []
    agent_infos = []
    observations = []
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    if animated:
        # env.visualize()
        env.render(**render_kwargs)

    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)
        a, agent_info = agent.get_action(last_obs)
        if deterministic and 'mean' in agent_info:
            a = agent_info['mean']
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

        if animated:
            env.render(**render_kwargs)

        if video_writer is not None:
            # We need to directly grab full observations so we can get image data
            # full_obs = env._get_observation()

            # Grab image data (assume relevant camera name is the first in the env camera array)
            # img = full_obs[env.camera_names[0] + "_image"]

            img = np.flip(env.render('rgb_array'),0)
            # Write to video writer
            video_writer.append_data(img[::-1])


    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
    )


# rollout_policy is inspired by rlkit_utils.simulate_policy function in robosuite-benchmark
def rollout_policy(env,policy,horizon,render,video_writer,
                   num_episodes=10,printout=True):
    ep = 0
    # Loop through simulation rollouts
    while ep < num_episodes:
        if printout:
            print(f"Rollout episode {ep} of {num_episodes}")

        path = rollout_ex(env,policy,
                          max_episode_length=horizon,
                          animated=render,
                          video_writer=video_writer)

        # Log diagnostics if supported by env
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        # logger.dump_tabular()

        # Increment episode count
        ep += 1




# @wrap_experiment(snapshot_mode='last')
def robosuite_rollout():

    if args.gpuid>=0 and torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=args.gpuid)
    else:
        set_gpu_mode(False)
    device = global_device()

    # todo: set the logger manualy (copy from the experiment)

    # load snapshot
    policy_snp_folder = args.policy_dir[0]
    policy_itr='last'
    if len(args.policy_dir)>1:
        policy_itr = int(args.policy_dir[1])
    snapshotter = Snapshotter()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load the policy
    policy_snp = snapshotter.load(os.path.expanduser(policy_snp_folder),
                                  itr=policy_itr)
    policy = policy_snp['algo'].policy.to(device)
    if args.env_dir:
        kwargs_fpath = os.path.join(args.env_dir[0], "variant.json")
    else:
        # load the environment from the snapshot and override
        # with command line arguments
        kwargs_fpath = os.path.join(args.policy_dir[0], "variant.json")
    # create the environment
    try:
        with open(kwargs_fpath) as f:
            kwargs = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
              "Please check filepath and try again.".format(kwargs_fpath))
    env_args = kwargs['variant']["eval_environment_kwargs"]
    # Note: Currently not supporting defining env from command line

    if args.horizon is not None:
        env_args["horizon"] = args.horizon
    env_args["render_camera"] = args.camera
    env_args["hard_reset"] = True
    env_args["ignore_done"] = True

    # Specify camera name if we're recording a video
    if args.record_video:
        env_args["camera_names"] = args.camera
        env_args["camera_heights"] = 512
        env_args["camera_widths"] = 512
        # Grab name of this rollout combo
        video_name = "{}-{}-{}".format(
            env_args["env_name"], "".join(env_args["robots"]), env_args["controller"]).replace("_", "-")
        # Calculate appropriate fps
        fps = int(env_args["control_freq"])
        # Define video writer
        video_writer = imageio.get_writer("{}.mp4".format(video_name), fps=fps)


    # Make sure we only pass in the proprio and object obs (no images)
    keys = ["object-state"]
    # assuming only 1 robots
    n_robots = 1
    if isinstance(env_args['robots'],list):
        n_robots = len(env_args['robots'])
    for idx in range(n_robots):
        keys.append(f"robot{idx}_proprio-state")

    env = RobosuiteEnvGrg(**env_args,
                          has_renderer=not args.record_video,
                          has_offscreen_renderer=args.record_video,
                          use_object_obs=True,
                          use_camera_obs=args.record_video,
                          reward_shaping=True,
                          keys=keys)
    env = GymEnv(env,max_episode_length=env_args['horizon'])
    env = normalize(env)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # rollout the policy
    rollout_policy(
        env=env,
        policy=policy,
        horizon=env_args["horizon"],
        render=not args.record_video,
        video_writer=video_writer,
        num_episodes=args.num_episodes,
        printout=True,
    )






if __name__ == '__main__':
    args=parse_cmd_line()

    robosuite_rollout()

    sys.path.remove(proj_root_dir)
