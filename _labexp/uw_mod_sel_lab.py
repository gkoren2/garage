import os
import sys
proj_root_dir = os.path.dirname(os.path.realpath('.'))   # project root dir
if proj_root_dir not in sys.path:
    sys.path.insert(0,proj_root_dir)
######################
import argparse
import time
import json
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
from sklearn.mixture import GaussianMixture
import pickle as pkl

DATA_FOLDER = os.path.join(os.path.expanduser('~'),'labexp_data')
DATA_PATH = '/data/users/gkoren2/rad_data/emb_data/'

def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--policies', type=str,
                        help='path to json with list of policies',
                        required=True)

    parser.add_argument('--n_epochs',type=int,help='number of epochs to train the external critic',default=200000)
    parser.add_argument('--uwc',action='store_true',help='use external critic trained on source')
    parser.add_argument('--intc',action='store_true',help='use internal critic')
    parser.add_argument('--ope', action='store_true',
                        help='if set, train external critic on target dataset')
    parser.add_argument('--gmm', action='store_true',
                        help='use gmm score')
    parser.add_argument('-pc', '--trained_critic_path', help='Path to a pretrained critic',
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

    def __init__(self, obs_dim,action_dim, **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self._obs_dim = obs_dim
        self._action_dim = action_dim

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
    def __init__(self,obs_dim,action_dim,
                 device,
                 hidden_dims=[1024, 1024],
                 discount=0.99,
                 n_epochs=1000,
                 batch_size=128,
                 critic_target_update_freq=1,
                 critic_tau=0.005,
                 lr=0.001,
                 layer_norm = False,
                 dropout_prob=0.2,
                 activation='relu',
                 n_sim=1000,
                 log_interval=10):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.dropout_prob= dropout_prob
        self.log_interval=log_interval
        self.n_epochs=n_epochs
        self.discount=discount
        self.device=device
        self.batch_size = batch_size
        self.n_sim = n_sim
        if activation=='relu':
            act_layer=nn.ReLU
        elif activation=='tanh':
            act_layer=nn.Tanh
        else:
            act_layer=nn.Identity

        self.critic = ContinuousMLPQFunctionDropout(self.obs_dim,self.action_dim,
                                                    hidden_sizes=hidden_dims,
                                                    hidden_nonlinearity=act_layer,
                                                    layer_normalization=layer_norm,
                                                    dropout_prob=dropout_prob).to(device)

        self.critic_target = ContinuousMLPQFunctionDropout(self.obs_dim,self.action_dim,
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
            print(f'step {step}: train_critic/loss {critic_loss}, target_Q={torch.mean(target_Q)}')

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

    def fit(self,dataset,n_epochs=None):
        self.n_epochs = n_epochs or self.n_epochs
        # extract the relevant items from the experience buffer
        obs, action, reward, next_obs, done, action_pred = tuple([v for k,v in dataset.items()])
        train_set_size = len(obs)
        # n_steps = (self.n_epochs * train_set_size) // self.batch_size
        n_batches = self.n_epochs
        # move the policy to the device
        print(f'training for {n_batches} batches * 1 gradient step per batch')
        for step in range(n_batches):
            # sample from the train set
            idxs = np.random.randint(0,train_set_size,size=self.batch_size)
            obs_tensor = torch.FloatTensor(obs[idxs]).to(self.device)
            next_obs_tensor = torch.FloatTensor(next_obs[idxs]).to(self.device)
            reward_tensor = torch.FloatTensor(reward[idxs]).to(self.device)
            action_tensor = torch.FloatTensor(action[idxs]).to(self.device)
            done_tensor = torch.FloatTensor(done[idxs]).to(self.device)
            action_pred_tensor = torch.FloatTensor(action_pred[idxs]).to(self.device)

            self._update_on_batch(obs_tensor, action_tensor, reward_tensor, next_obs_tensor, done_tensor,
                                  action_pred_tensor, step)

            #update the target
            if step % self.critic_target_update_freq == 0:
                self._soft_update_target()

        return self

    def predict(self,obs,action,n_sim=None):
        n_samples=n_sim or self.n_sim
        self.critic.train()
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        obs_action = torch.cat([obs_tensor, action_tensor], dim=-1)
        Q_preds=np.array([self.critic(obs_action).data.cpu().numpy() for _ in range(n_samples)]).squeeze()
        q_mean = np.mean(Q_preds,axis=0)
        q_std = np.std(Q_preds,axis=0)
        return q_mean,q_std
#endregion
##########################################
#region tools
from garage.np import discount_cumsum
def analyze_eval_episodes(batch, discount, critics=None, device=None):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    if critics is None:
        critics = {}
    returns = []
    q0s=[]
    undiscounted_returns = []
    termination = []
    success = []
    s0_arr=[]
    a0_arr=[]
    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))
        s0_arr.append(eps.observations[0])
        a0_arr.append(eps.actions[0])
    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    # prepare for critics evaluations
    s0_arr = np.vstack(s0_arr)
    a0_arr = np.vstack(a0_arr)
    critics_results={}
    for cid,critic in critics.items():
        if isinstance(critic,GaussianMixture):    # gmm
            q0s=critic.score(s0_arr)
        elif hasattr(critic,'predict'):       # external critic
            q0s=critic.predict(s0_arr,a0_arr)
        else:   # internal critic
            obs_tensor = torch.FloatTensor(s0_arr).to(device)
            act_tensor = torch.FloatTensor(a0_arr).to(device)
            q0s = critic(obs_tensor, act_tensor).data.cpu().numpy()
        critics_results.update({cid:np.mean(q0s)})
    logger.log(f'NumEpisodes: {len(returns)}')
    logger.log(f'Actual AverageDiscountedReturn: {average_discounted_return}')
    logger.log(f'Critics V^(s0): {critics_results}')
    # logger.log(f'AverageReturn: {np.mean(undiscounted_returns)}')
    # logger.log(f'StdReturn: {np.std(undiscounted_returns)}')

    return

#endregion
##########################################

def eval_policy_with_int_critic(policy,dataset,critic,device):
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


def eval_policy_with_ext_critic(dataset,critic,device):
    logger.log('doing offline evaluation on evaluation dataset using uncertainty weighted critic ...')
    obs=dataset['observation']
    action_pred=dataset['action_pred']
    batch_size = 10000
    q_mean_all=[]
    q_std_all=[]
    n_obs = len(obs)
    for l in tqdm(range(0,n_obs,batch_size)):
        q_mean,q_std = critic.predict(obs[l:l+batch_size],action_pred[l:l+batch_size])
        q_mean_all.append(q_mean)
        q_std_all.append(q_std)
    q_mean_all = np.array(q_mean_all).flatten()
    q_std_all = np.array(q_std_all).flatten()
    logger.log(f'average q_mean = {q_mean_all.mean()} , average q_std = {q_std_all.mean()} ')
    # statistic summary : the average (over all states) of uncertainty weighted q values
    value=np.mean(q_mean_all-q_std_all)
    return value


'''
def fit_gmm_and_score(sim_embedding_dataset, real_embedding_dataset, sample_seq_with_length=None):
    def get_random_sequences_no_replacement(array_size, seq_length):
        arr = list(range(array_size))
        random.shuffle(arr)
        ret_list = []

        while True:
            ret_list.append(arr[:seq_length])
            del arr[:seq_length]
            if len(arr) < seq_length:
                break

        return ret_list

    gmm = GaussianMixture(n_components=2, covariance_type='diag').fit(sim_embedding_dataset)

    if sample_seq_with_length is None:
        sample_seq_with_length = real_embedding_dataset.shape[0]

    sequences = get_random_sequences_no_replacement(real_embedding_dataset.shape[0], sample_seq_with_length)
    real_score = [gmm.score(real_embedding_dataset[seq]) for seq in sequences]

    return real_score

'''

def train_eval_gmm(src_dataset,tgt_dataset):
    src_embeddings = src_dataset['observation']
    tgt_embeddings = tgt_dataset['observation']
    gmm = GaussianMixture(n_components=2, covariance_type='diag').fit(
        src_embeddings)
    value = gmm.score(tgt_embeddings)
    return value,gmm




def load_or_train_critic(dataset,device,critic_file_name,n_epochs=120000):
    if os.path.exists(critic_file_name):
        logger.log(f'loading critic from {critic_file_name}')
        with open(critic_file_name, 'rb') as file:
            critic = cloudpickle.load(file)
    else:
        logger.log('creating and training a critic')
        obs_dim = np.prod(dataset['observation'].shape[1:])
        action_dim = np.prod(dataset['action'].shape[1:])
        critic = UncWgtCritic(obs_dim,action_dim,device,
                              hidden_dims=[256,256],
                              batch_size=256,
                              dropout_prob=0.2,
                              # dropout_prob=0,
                              lr=3e-4,
                              critic_target_update_freq=1000,
                              critic_tau=.99,
                              n_sim=1000,
                              log_interval=1000)
        critic.fit(dataset,n_epochs=n_epochs)
        # save the critic
        with open(critic_file_name, 'wb') as file:
            cloudpickle.dump(critic, file)
    return critic


@wrap_experiment(snapshot_mode='last')
def model_sel_lab(ctxt=None,args=None):
    # args = parse_cmd_line()
    if args.gpuid>=0 and torch.cuda.is_available():
        set_gpu_mode(True,gpu_id=args.gpuid)
    else:
        set_gpu_mode(False)
    device = global_device()


    try:
        with open(args.policies) as f:
            variant=json.load(f)
    except FileNotFoundError:
        logger.log(f'Error opening config at {args.policies}')
        return
    eval_results_s=dict()
    eval_results_s0 = dict()

    models_datasets = variant['models']
    for i,model in enumerate(models_datasets):
        model_sim_name=os.path.basename(model)
        model_name = os.path.splitext(model_sim_name)[0].split("_sim")[0]
        logger.log('='*30 + f'model {i+1}/{len(models_datasets)}:{model_name}' +'='*30)
        emb_name = model_name.split('_')[0]
        src_ds = model
        tgt_ds = os.path.join(DATA_PATH,f'{emb_name}_real.pkl')
        tgt_ds0 = os.path.join(DATA_PATH, f'{emb_name}_real_s0.pkl')
        with open(src_ds,'rb') as fp:
            src_data = pkl.load(fp)
            src_dataset = dict(observation=np.concatenate([src_data['embedding'],src_data['proprio']],axis=1),
                               action=src_data['action'],
                               reward=src_data['reward'],
                               next_observation=np.concatenate([src_data['next_embedding'],src_data['next_proprio']],axis=1),
                               terminal=1.0-src_data['not_done'],
                               action_pred=src_data['action_pred'])

        with open(tgt_ds,'rb') as fp:
            tgt_data = pkl.load(fp)
            tgt_dataset = dict(observation=np.concatenate([tgt_data['embedding'],tgt_data['proprio']],axis=1),
                               action_pred=tgt_data['action_pred'])

        with open(tgt_ds0,'rb') as fp:
            tgt_data0 = pkl.load(fp)
            tgt_dataset_s0 = dict(observation=np.concatenate([tgt_data0['embedding'],tgt_data0['proprio']],axis=1),
                               action_pred=tgt_data0['action_pred'])

        logger.log(f'creating critic and training on {model_sim_name}')
        # todo: load the sim file and prepare the data for the critic
        critic_file_name = model.replace('_sim.pkl','_uwc.pkl')
        uwc_critic = load_or_train_critic(src_dataset, device,
                                      critic_file_name, args.n_epochs)

        value = eval_policy_with_ext_critic(tgt_dataset, uwc_critic, device)
        value_s0 = eval_policy_with_ext_critic(tgt_dataset_s0, uwc_critic,
                                            device)

        eval_results_s.update({model_name:value})
        eval_results_s0.update({model_name:value_s0})
    output_fname = model.replace('_sim.pkl','_uwc_vs_do-0.2.pkl')
    logger.log(f'saving results dict model:v(s) in {output_fname}')
    with open(output_fname,'wb') as fp:
        pkl.dump(eval_results_s,fp)

    output_fname = model.replace('_sim.pkl','_uwc_vs0_do-0.2.pkl')
    logger.log(f'saving results dict model:v(s0) in {output_fname}')
    with open(output_fname,'wb') as fp:
        pkl.dump(eval_results_s0,fp)




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
    options = {'name':'mod_sel_lab'}
    model_sel_lab(options, args=args)

    sys.path.remove(proj_root_dir)
