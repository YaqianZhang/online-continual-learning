from RL.pytorch_util import  build_mlp,build_lstm

from RL.dqn_utils import cl_exploration_schedule,critic_lr_schedule
import torch
from utils.utils import maybe_cuda
from RL.lstm import LSTM_critic

class MAB_critic_class(object):
    def __init__(self,params,action_num,ob_dim,training_steps,RL_agent):
        self.RL_agent = RL_agent
        self.total_training_steps = training_steps
        self.params = params
        self.action_num = action_num
        self.ob_dim = ob_dim
        self.gamma = 1
        self.grad_norm_clipping = 10

        self.initialize_critic(params,self.action_num,self.ob_dim)


    def initialize_critic(self, params, action_num, ob_dim):
        self.q_values = torch.zeros(action_num)











    def train_batch(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch,training_steps):
        self.q_values[action_batch] +=0.1* ( self.q_values[action_batch] -reward_batch)











