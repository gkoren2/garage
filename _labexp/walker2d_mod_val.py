import os
import sys
proj_root_dir = os.path.dirname(os.path.realpath('.'))   # project root dir
if proj_root_dir not in sys.path:
    sys.path.insert(0,proj_root_dir)
######################
import argparse
import numpy as np
import pickle as pkl
import json
import torch
import torch.nn as nn
import copy
import garage
from dowel import logger,tabular
from scipy import stats
from garage import wrap_experiment, obtain_evaluation_episodes, StepType
from garage.experiment import Snapshotter,deterministic
from garage.torch import NonLinearity,set_gpu_mode,global_device
from torch.nn import functional as F
import cloudpickle
from tqdm import tqdm
from garage.envs import GymEnv, normalize
from _labexp.my_envs.walker2d_rand_params import Walker2DRandParamsEnv

DATA_FOLDER = os.path.join(os.path.expanduser('~'),'labexp_data')

def parse_cmd_line():
    parser = argparse.ArgumentParser()
    # experiment args
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='gpu id or -1 for cpu')
    parser.add_argument(
        '--tasks',
        type=str,
        default='walker2d_rand_tasks.pkl')

    parser.add_argument('--config_path',type=str,default='walker2d_mod_val_config.json',
                        help='path to config json file')

    args = parser.parse_args()
    return args

##########################################
#region tools
from garage.np import discount_cumsum
def analyze_eval_episodes(batch, discount):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []

    undiscounted_returns = []
    termination = []
    success = []

    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))
    average_discounted_return = np.mean([rtn[0] for rtn in returns])
    return average_discounted_return,np.mean(undiscounted_returns),np.std(undiscounted_returns)

#endregion
##########################################


@wrap_experiment(snapshot_mode='none')
def walker2d_mod_val(ctxt=None,args=None):
    deterministic.set_seed(args.seed)
    if args.gpuid>=0 and torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=args.gpuid)
    else:
        set_gpu_mode(False)
    device = global_device()

    # load the tasks
    try:
        with open(args.tasks, 'rb') as f:
            tasks = pkl.load(f)
        logger.log(f'loaded {len(tasks)} tasks from pkl')
    except FileNotFoundError:
        logger.log(f'Error opening task file at {args.tasks}')
        return

    # read the json configuration file
    try:
        with open(args.config_path) as f:
            variant=json.load(f)
    except FileNotFoundError:
        logger.log(f'Error opening config at {args.config_path}')
        return

    val_envs = variant['val_envs']
    models_path = variant['models']
    tgt_env_id=variant['tgt_env_id']
    tgt_env = Walker2DRandParamsEnv()
    task = tasks[tgt_env_id]
    tgt_env.set_task(task)
    tgt_env = normalize(GymEnv(tgt_env, max_episode_length=200),
                    expected_action_scale=1.)

    if isinstance(val_envs,float):
        all_env_ids = [i for i in range(len(tasks))]
        all_env_ids.remove(tgt_env_id)
        n_val_envs = int(val_envs*len(tasks))
        val_envs = np.random.choice(all_env_ids,n_val_envs,replace=False)

    logger.log(f'validation on {len(val_envs)} envs with ids: {val_envs}')
    logger.log(f'target env id {tgt_env_id}')
    snapshotter = Snapshotter()
    results_dict={}
    for p in models_path:
        logger.log(f'evaluation model in {p}')
        policy_snp = snapshotter.load(p, itr='last')
        policy = policy_snp['algo'].policy.to(device)
        avg_disc_rets=[]
        avg_rets=[]
        std_rets=[]
        for tid in val_envs:
            logger.log(f'evaluating on task id {tid}')
            env = Walker2DRandParamsEnv()
            task = tasks[tid]
            env.set_task(task)
            env = normalize(GymEnv(env, max_episode_length=200),
                            expected_action_scale=1.)
            logger.log('evaluating the policy on the target env for 100 episodes')
            discount = policy_snp['algo']._discount
            eval_episodes = obtain_evaluation_episodes(policy, env)
            avg_disc_ret,avg_ret,std_ret = analyze_eval_episodes(eval_episodes,discount=discount)
            avg_disc_rets.append(avg_disc_ret)
            avg_rets.append(avg_ret)
            # std_rets=std_ret
            env.close()
        logger.log(f'evaluating the model on target env')
        tgt_episodes = obtain_evaluation_episodes(policy, tgt_env)
        avg_disc_ret, avg_ret, std_ret = analyze_eval_episodes(tgt_episodes,
                                                               discount=discount)

        logger.log(f'results for model in {p}: ')
        logger.log(f'validation score: Average Discounted Return = {np.mean(avg_disc_rets)}, Average Returns = {np.mean(avg_rets)}')
        logger.log(f'evaluation on target env: Average Discounted Return = {avg_disc_ret},Average Return = {avg_ret}, Std Return = {std_ret}')

        results_dict.update({p:[np.mean(avg_disc_rets),np.mean(avg_rets),
                                avg_disc_ret,avg_ret,std_ret]})
    logger.log('~'*30)
    logger.log('results summary:')
    logger.log(f'{results_dict}')

    val_results = [res[1] for p,res in results_dict.items()]
    tst_results = [res[3] for p,res in results_dict.items()]
    spearman_coef = stats.spearmanr(val_results,tst_results)
    logger.log(f'Spearman correlation: {spearman_coef}')
    logger.log('done.')

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
    options = {'name':'walker2d_mod_val'}
    walker2d_mod_val(options, args=args)

    sys.path.remove(proj_root_dir)
