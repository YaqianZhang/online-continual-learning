from RL.pytorch_util import  build_mlp

from RL.dqn_utils import cl_exploration_schedule,critic_lr_schedule
import torch
from utils.utils import maybe_cuda
from RL.critic.critic import critic_class

class task_critic_class(critic_class):
    def __init__(self,params,action_num,ob_dim,training_steps,RL_agent):
        self.RL_agent = RL_agent
        self.total_training_steps = training_steps
        self.params = params
        self.action_num = 2
        self.ob_dim = ob_dim
        self.gamma = 1
        self.grad_norm_clipping = 10



        self.initialize_critic(params,self.action_num,self.ob_dim)


    def initialize_critic(self, params, action_num, ob_dim):

        self.rl_lr = critic_lr_schedule(self.total_training_steps, self.params.critic_lr_type)
        self.n_layers = params.critic_nlayer
        self.size = params.critic_layer_size


        self.loss = torch.nn.SmoothL1Loss()

        self.rl_wd = self.params.critic_wd  # 1 * 10 ** (-6)  # -4
        self.build_task_critic(action_num, ob_dim)




    def build_task_critic(self,action_dim,state_dim,output_dim=20):

        self.q_func = build_mlp(
            state_dim+action_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )

        self.q_func = maybe_cuda(self.q_func, self.params.cuda)

    def compute_q(self, obs, action):
        q_values = self.q_func([obs,action])
        return q_values



    def get_parameters(self):

        return list(self.q_func.parameters())+ \
               list(self.actor.parameters())










