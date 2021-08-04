import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from utils.utils import maybe_cuda

import numpy as np
import torch
from torch import distributions

from RL.pytorch_util import build_mlp
from RL.actor.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            #self.logits_na.to(ptu.device)
            self.logits_na = maybe_cuda(self.logits_na)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, )
            )
            #self.mean_net.to(ptu.device)
            self.mean_net = maybe_cuda(self.mean_net)
            #self.logstd.to(ptu.device)
            self.logstd = maybe_cuda(self.logstd)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            #self.baseline.to(ptu.device)
            self.baseline = maybe_cuda(self.baseline)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################


    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        observation_tensor = torch.tensor(observation, dtype=torch.float)#.to(ptu.device)
        observation_tensor = maybe_cuda(observation_tensor)
        action_distribution = self.forward(observation_tensor)
        action = action_distribution.sample().cpu().numpy()
        return action

    def train_batch(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):

        action_distribution = self.forward(state_batch)
        log_prob = action_distribution.log_prob(action_batch)
        loss = torch.sum(log_prob*reward_batch)




        # rl_loss = torch.nn.SmoothL1Loss()(predict_q, td_target )



        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.q_function.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        return loss


    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            dist = distributions.Categorical(logits=self.logits_na(observation))
        else:
            dist = distributions.Normal(
                self.mean_net(observation),
                torch.exp(self.logstd)[None],
            )
        return dist



#####################################################
#####################################################

#
# class MLPPolicyAC(MLPPolicy):
#     def update(self, observations, actions, adv_n=None):
#         # TODO: update the policy and return the loss
#         loss = TODO
#         return loss.item()
