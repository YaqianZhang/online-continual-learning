from RL.pytorch_util import  build_mlp

from RL.dqn_utils import cl_exploration_schedule,critic_lr_schedule
import torch
from utils.utils import maybe_cuda
from RL.critic.critic import critic_class

class actor_critic_class(critic_class):
    def __init__(self,params,action_num,ob_dim,training_steps,RL_agent):
        self.RL_agent = RL_agent
        self.total_training_steps = training_steps
        self.params = params
        if(params.RL_type == "RL_2ratioMemIter"):
            if(params.mem_iter_max != params.mem_iter_min):
                self.action_num = 2
            else:
                self.action_num = 1

        else:
            self.action_num = 1
        self.ob_dim = ob_dim
        self.gamma = 1
        self.grad_norm_clipping = 10



        self.initialize_critic(params,self.action_num,self.ob_dim)
        self.init_actor_network(self.action_num, self.ob_dim)

        self.rl_opt = torch.optim.Adam(self.get_parameters(),
                                  lr=self.params.critic_lr,
                                  )

        self.rl_actor_opt = torch.optim.Adam(self.get_actor_parameters(),
                                  lr=self.params.critic_lr,
                                  )

    def init_actor_network(self,action_dim, ob_dim):
        self.actor = build_mlp(
            ob_dim,
            action_dim,
            n_layers=self.n_layers,
            size=self.size,
            output_activation="sigmoid",
        )
        self.actor= maybe_cuda(self.actor, self.params.cuda)

    def get_actor_parameters(self):

        return list(self.actor.parameters())




    def initialize_critic(self, params, action_num, ob_dim):

        self.rl_lr = critic_lr_schedule(self.total_training_steps, self.params.critic_lr_type)
        self.n_layers = params.critic_nlayer
        self.size = params.critic_layer_size


        self.loss = torch.nn.SmoothL1Loss()

        self.rl_wd = self.params.critic_wd  # 1 * 10 ** (-6)  # -4
        self.build_task_critic(action_num, ob_dim)





    def build_task_critic(self,action_dim,state_dim,output_dim=20):

        self.q_function = build_mlp(
            state_dim+action_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )

        self.q_function = maybe_cuda(self.q_function, self.params.cuda)

    def compute_q(self, obs, action):
        input=torch.cat((obs,action),dim=1).float()

        q_values = self.q_function(input).reshape([-1])
        return q_values



    def get_parameters(self):

        return list(self.q_function.parameters())+ \
               list(self.actor.parameters())

    def train_batch(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, training_steps):

        if self.params.episode_type == "multi-step":

            with torch.no_grad():
                # q_s = self.q_function_target(next_state_batch)
                q_s_target = self.compute_q(next_state_batch, self.q_function_target)
                q_s = self.compute_q(next_state_batch, self.q_function)

                if (self.params.double_DQN):

                    max_a = torch.max(q_s_target, axis=1)[1]
                    max_a_na = torch.nn.functional.one_hot(max_a, num_classes=q_s.shape[1])
                    td_target = reward_batch + self.gamma * torch.sum(q_s * max_a_na, axis=1) * (1 - done_batch)

                else:

                    td_target = reward_batch + self.gamma * torch.max(q_s, axis=1)[0] * (1 - done_batch)
        else:
            td_target = reward_batch

        td_target = td_target.float()
        # print("train batch",)

        n = state_batch.shape[0]
        # print(self.RL_agent.ER_batchsize, n)

        predict_q = self.compute_q(state_batch, action_batch).float()

        #predict_q = q_values[torch.arange(n), action_batch]

        td_target = maybe_cuda(td_target)



        assert predict_q.shape == td_target.shape

        rl_loss = torch.nn.functional.mse_loss(predict_q, td_target, reduction="mean")

        # rl_loss = torch.nn.SmoothL1Loss()(predict_q, td_target )

        # rl_opt = torch.optim.Adam(self.get_parameters(),
        #                           lr=self.rl_lr.value(training_steps),
        #                           weight_decay=self.rl_wd)

        self.rl_opt.zero_grad()
        rl_loss.backward()
        torch.nn.utils.clip_grad_value_(self.get_parameters(), self.grad_norm_clipping)
        if (self.params.RL_agent_update_flag):
            self.rl_opt.step()
        # print("train RL, loss", rl_loss.item())



        #### perform actor update
        actions = self.actor(state_batch)
        actor_loss = -torch.sum(self.compute_q(state_batch,actions))
        #print("train RL actor, loss", actor_loss.item())

        self.rl_actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.get_actor_parameters(), self.grad_norm_clipping)

        self.rl_actor_opt.step()

        return rl_loss










