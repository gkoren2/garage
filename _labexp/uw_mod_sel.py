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
from garage.experiment import Snapshotter
from garage.torch import NonLinearity,set_gpu_mode,global_device
from torch.nn import functional as F
import cloudpickle
from garage import rollout
from tqdm import tqdm


DATA_FOLDER = os.path.join(os.path.expanduser('~'),'labexp_data')

def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--policies', type=str, nargs='+',
                        help='path and snapshot numbers of the candidate policies',
                        required=True)
    parser.add_argument('-d','--val_dataset_path', type=str, nargs='+',
                        help='path iteration [start sample] [end sample]',
                        required=True)
    parser.add_argument('--intc',action='store_true',help='use internal critic')
    parser.add_argument('-c', '--trained_critic_path', help='Path to a pretrained critic to continue training',
                        default='', type=str)

    parser.add_argument('--gpuid',type=int,default=0,help='gpu id or -1 for cpu')
    # parser.add_argument('--num_experiments', help='number of experiments', default=1,type=int)
    # parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    # parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    # parser.add_argument('-i', '--trained_agent', help='Path to a pretrained agent to continue training',
    #                     default='', type=str)
    # parser.add_argument('-n', '--n_timesteps', help='Overwrite the number of timesteps', default=-1,type=int)
    # parser.add_argument('--log_interval', help='Override log interval (default: -1, no change)', default=-1,type=int)
    args = parser.parse_args()
    return args

##########################################
#region infrastructures - MCDO models

class MultiHeadedMLPModuleDropout(nn.Module):
    """MultiHeadedMLPModule Model with Dropout after each layer

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 n_heads,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False,
                 dropout_prob=0):
        super().__init__()

        self._layers = nn.ModuleList()

        output_dims = self._check_parameter_for_output_layer(
            'output_dims', output_dims, n_heads)
        output_w_inits = self._check_parameter_for_output_layer(
            'output_w_inits', output_w_inits, n_heads)
        output_b_inits = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits, n_heads)
        output_nonlinearities = self._check_parameter_for_output_layer(
            'output_nonlinearities', output_nonlinearities, n_heads)

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))
            if dropout_prob>0:
                hidden_layers.add_module('dropout',nn.Dropout(p=dropout_prob))

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(prev_size, output_dims[i])
            output_w_inits[i](linear_layer.weight)
            output_b_inits[i](linear_layer.bias)
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module('non_linearity',
                                        NonLinearity(output_nonlinearities[i]))

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_heads (int): number of head

        Returns:
            list: list of variables (length of n_heads)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_heads

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'n_heads ({}), but {} provided.')
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]


class MLPModuleDropout(MultiHeadedMLPModuleDropout):
    """MLP Model with Dropout

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 dropout_prob=0):
        super().__init__(1, input_dim, output_dim, hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization,dropout_prob)

        self._output_dim = output_dim

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value

        """
        return super().forward(input_value)[0]

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim


class ContinuousMLPQFunctionDropout(MLPModuleDropout):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MLPModuleDropout.__init__(self,
                           input_dim=self._obs_dim + self._action_dim,
                           output_dim=1,
                           **kwargs)

    # pylint: disable=arguments-differ
    # def forward(self, observations, actions):
    #     """Return Q-value(s).
    #
    #     Args:
    #         observations (np.ndarray): observations.
    #         actions (np.ndarray): actions.
    #
    #     Returns:
    #         torch.Tensor: Output value
    #     """
    #     return super().forward(torch.cat([observations, actions], 1))
#endregion
##########################################
#region Uncertainty Weighted Critic
class UncWgtCritic:
    def __init__(self,env_spec,
                 device,
                 hidden_dims=[1024, 1024],
                 discount=0.99,
                 n_epochs=1000,
                 batch_size=128,
                 critic_target_update_freq=100,
                 critic_tau=0.01,
                 lr=0.001,
                 layer_norm = False,
                 dropout_prob=0.2,
                 activation='relu',
                 log_interval=1000):
        self.env_spec = env_spec
        self.lr = lr
        self.dropout_prob= dropout_prob
        self.log_interval=log_interval
        self.n_epochs=n_epochs
        self.discount=discount
        self.device=device
        self.batch_size = batch_size

        if activation=='relu':
            act_layer=nn.ReLU
        elif activation=='tanh':
            act_layer=nn.Tanh
        else:
            act_layer=nn.Identity

        self.critic = ContinuousMLPQFunctionDropout(self.env_spec,
                                                    hidden_sizes=hidden_dims,
                                                    hidden_nonlinearity=act_layer,
                                                    layer_normalization=layer_norm,
                                                    dropout_prob=dropout_prob).to(device)

        self.critic_target = ContinuousMLPQFunctionDropout(self.env_spec,
                                                           hidden_sizes=hidden_dims,
                                                           hidden_nonlinearity=act_layer,
                                                           layer_normalization=layer_norm,
                                                           dropout_prob=dropout_prob).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target_update_freq = critic_target_update_freq
        self.critic_tau = critic_tau

        self.training = True    # a flag to indicate whether we're in training/eval mode

    def train(self, training=True):
        '''
        change the operational mode :
        training = True - means we're in training mode.
        training = False - means we're in evaluation mode.
        '''
        self.training = training
        self.critic.train(training)


    def _update_on_batch(self,obs,action, reward, next_obs, done,action_pred,step):
        with torch.no_grad():
            next_obs_action_pred = torch.cat([next_obs, action_pred], dim=1)
            target_Q = self.critic_target(next_obs_action_pred)
            target_Q = reward + ((1-done) * self.discount * target_Q)

        # get current Q estimates
        obs_action = torch.cat([obs, action], dim=1)
        current_Q = self.critic(obs_action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        if step % self.log_interval == 0:
        #     L.log('train_critic/loss', critic_loss, step)
        #     logger.log(f'step {step}: train_critic/loss {critic_loss}')
            print(f'step {step}: train_critic/loss {critic_loss}')

        #     tabular.record('train_critic/loss',critic_loss)
        #     logger.log(tabular)
        #     logger.dump_all(step)
        #     tabular.clear()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _soft_update_target(self):
        tau=self.critic_tau
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def fit(self,dataset,policy):
        # extract the relevant items from the experience buffer
        obs, action, reward, next_obs, done = tuple([v for k,v in dataset.items()])
        train_set_size = len(obs)
        n_steps = (self.n_epochs * train_set_size) // self.batch_size

        # move the policy to the device
        policy = policy.to(self.device)
        print(f'training for {n_steps} steps')
        for step in range(n_steps):
            # sample from the train set
            idxs = np.random.randint(0,train_set_size,size=self.batch_size)
            obs_tensor = torch.FloatTensor(obs[idxs]).to(self.device)
            next_obs_tensor = torch.FloatTensor(next_obs[idxs]).to(self.device)
            reward_tensor = torch.FloatTensor(reward[idxs]).to(self.device)
            action_tensor = torch.FloatTensor(action[idxs]).to(self.device)
            done_tensor = torch.FloatTensor(done[idxs]).to(self.device)
            action_pred_tensor = torch.FloatTensor(policy.get_actions(next_obs_tensor)[0]).to(self.device)

            self._update_on_batch(obs_tensor, action_tensor, reward_tensor, next_obs_tensor, done_tensor,
                                  action_pred_tensor, step)

            #update the target
            if step % self.critic_target_update_freq == 0:
                self._soft_update_target()

        return self

    def predict(self,obs,action,n_sim=1000):
        self.critic.train()
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        obs_action = torch.cat([obs_tensor, action_tensor], dim=1)
        Q_preds=np.array([self.critic(obs_action).data.cpu().numpy() for _ in range(n_sim)]).squeeze()
        q_mean = np.mean(Q_preds,axis=0)
        q_std = np.std(Q_preds,axis=0)
        return q_mean,q_std
#endregion
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

    logger.log(f'NumEpisodes: {len(returns)}')
    logger.log(f'AverageDiscountedReturn: {average_discounted_return}')
    logger.log(f'AverageReturn: {np.mean(undiscounted_returns)}')
    logger.log(f'StdReturn: {np.std(undiscounted_returns)}' )
    # logger.log('MaxReturn', np.max(undiscounted_returns))
    # logger.log('MinReturn', np.min(undiscounted_returns))
    # logger.log('TerminationRate', np.mean(termination))
    # if success:
    #     logger.log('SuccessRate', np.mean(success))
    return

#endregion
##########################################

def eval_policy_with_internal_critic(policy,dataset,critic,device):
    logger.log('doing offline evaluation on evaluation dataset using internal critic ...')
    obs=dataset['observation']
    batch_size = 10000
    q_all=[]
    n_obs = len(obs)
    for l in tqdm(range(0,n_obs,batch_size)):
        obs_tensor = torch.FloatTensor(obs[l:l+batch_size]).to(device)
        policy_act_tensor = torch.FloatTensor(policy.get_actions(obs_tensor)[0]).to(device)
        q = critic(obs_tensor,policy_act_tensor).data.cpu().numpy()
        q_all.append(q)
    q_all = np.array(q_all).flatten()
    logger.log(f'average q_mean = {q_all.mean()} ')
    # statistic summary : the average (over all states) of uncertainty weighted q values
    value=np.mean(q_all)
    return value




def eval_policy_with_uw_critic(policy,dataset,critic,device):
    logger.log('doing offline evaluation on evaluation dataset using uncertainty weighted critic ...')
    obs=dataset['observation']
    batch_size = 10000
    q_mean_all=[]
    q_std_all=[]
    n_obs = len(obs)
    for l in tqdm(range(0,n_obs,batch_size)):
        obs_tensor = torch.FloatTensor(obs[l:l+batch_size]).to(device)
        policy_act = policy.get_actions(obs_tensor)[0]
        q_mean,q_std = critic.predict(obs[l:l+batch_size],policy_act,n_sim=1)
        q_mean_all.append(q_mean)
        q_std_all.append(q_std)
    q_mean_all = np.array(q_mean_all).flatten()
    q_std_all = np.array(q_std_all).flatten()
    logger.log(f'average q_mean = {q_mean_all.mean()} , average q_std = {q_std_all.mean()} ')
    # statistic summary : the average (over all states) of uncertainty weighted q values
    value=np.mean(q_mean_all-q_std_all)
    return value


def load_or_train_critic(policy,dataset,env_spec,device,critic_dir):
    critic_file_name = os.path.join(critic_dir, 'critic.pkl')
    if os.path.exists(critic_file_name):
        logger.log(f'loading critic from {critic_file_name}')
        with open(critic_file_name, 'rb') as file:
            critic = cloudpickle.load(file)
    else:
        logger.log('creating and training a critic')
        critic = UncWgtCritic(env_spec,device,n_epochs=5,hidden_dims=[1024,1024],dropout_prob=0.)
        critic.fit(dataset,policy)
        # save the critic
        with open(critic_file_name, 'wb') as file:
            cloudpickle.dump(critic, file)
    return critic


@wrap_experiment(snapshot_mode='last')
def unc_wgt_policy_sel(ctxt=None,args=None):
    # args = parse_cmd_line()
    if args.gpuid>=0 and torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=args.gpuid)
    else:
        set_gpu_mode(False)
    device = global_device()

    # exp_dir = os.path.join(exp_dir,'uw_mod_sel'+ '-' + time.strftime("%d-%m-%Y_%H-%M-%S"))

    # snp_folder = os.path.join(exp_dir, 'sac_half_cheetah_vel')
    # policies_itr = {'poor': 40, 'bad': 60, 'good': 160, 'expert': 460}
    # working on gpu15
    policy_snp_folder = args.policies[0]
    dataset_snp_folder = args.val_dataset_path[0]
    # policies_itr = {'poor': 60, 'bad': 80, 'good': 120, 'expert': 460}

    snapshotter = Snapshotter()
    # policies = {k: snapshotter.load(snp_folder, itr=itr)['algo'].policy for
    #             k, itr in policies_itr.items()}

    policy_itr = int(args.policies[1])
    dataset_itr = int(args.val_dataset_path[1])    # s for source, t for target
    logger.log(f'Evaluating policy from {policy_snp_folder} itr_{policy_itr} on dataset {dataset_snp_folder} itr_{dataset_itr}')
    policy_snp = snapshotter.load(os.path.expanduser(policy_snp_folder), itr=policy_itr)
    policy = policy_snp['algo'].policy
    data_snapshot = snapshotter.load(os.path.expanduser(dataset_snp_folder), itr=dataset_itr)
    env = data_snapshot['env']
    replay_buffer = data_snapshot['algo'].replay_buffer
    # need to leave only the valid samples in the dataset
    n_trans_stored = replay_buffer.n_transitions_stored
    start_trans=0
    if len(args.val_dataset_path) > 2:   # we're given a segment from the buffer
        start_trans = int(args.val_dataset_path[2])
    end_trans = n_trans_stored
    if len(args.val_dataset_path) > 3:   # we're given a segment from the buffer
        end_trans = int(args.val_dataset_path[3])
    logger.log(f'selecting transitions {start_trans} to {end_trans} for the evaluation')
    dataset = {k: v[start_trans:end_trans] for k, v in replay_buffer._buffer.items()}
    critic_path = args.trained_critic_path or ctxt.snapshot_dir
    if args.intc:   # use internal critic
        logger.log('using internal critic')
        critic = policy_snp['algo']._target_qf1.to(device)
        value = eval_policy_with_internal_critic(policy,dataset,critic,device)
    else: # use uw critic
        logger.log('using uw critic')
        critic = load_or_train_critic(policy,dataset,env.spec,device,critic_path)
        value = eval_policy_with_uw_critic(policy, dataset, critic, device)

    logger.log(f'the value of policy {policy_itr} on dataset {dataset_itr} is {value}')

    logger.log('evaluating the policy on the target env for 100 episodes')
    discount = policy_snp['algo']._discount
    eval_episodes = obtain_evaluation_episodes(policy, env)
    analyze_eval_episodes(eval_episodes,discount=discount)
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
    options = {'name':'uw_mod_sel'}
    unc_wgt_policy_sel(options, args=args)

    sys.path.remove(proj_root_dir)
