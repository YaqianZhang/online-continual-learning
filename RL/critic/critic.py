from RL.pytorch_util import  build_mlp

from RL.dqn_utils import cl_exploration_schedule,critic_lr_schedule
import torch
from utils.utils import maybe_cuda

class critic_class(object):
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

        self.rl_lr = critic_lr_schedule(self.total_training_steps, self.params.critic_lr_type)
        self.n_layers = params.critic_nlayer
        self.size = params.critic_layer_size




        self.loss = torch.nn.SmoothL1Loss()

        # if (self.params.episode_type == "batch"):
        #     self.rl_lr = 0.0005
        #     self.rl_wd = 0.0001
        # else:
        # self.rl_lr = self.params.critic_lr(self.params.) #5 * 10 ** (-6)  # -4
        self.rl_wd = self.params.critic_wd  # 1 * 10 ** (-6)  # -4

        self.q_function = build_mlp(
            ob_dim,
            action_num,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.q_function = maybe_cuda(self.q_function, params.cuda)

        if (self.params.critic_use_model):
            self.load_critic_model(self.q_function)

        self.q_function_target = build_mlp(
            ob_dim,
            action_num,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.q_function_target = maybe_cuda(self.q_function_target, params.cuda)
        if (self.params.critic_use_model):
            self.update_q_target()



        return self.q_function, self.q_function_target

    def load_critic_model(self, model):
        print("!!! load pre-trained model")
        PATH= "results/59054/RLER1_MIR_ran_mIter3_testBch100_RL2rmemIter_33_test_loss_0.0_new_old6mn_nxtBtch_recent2_misc4_nc_20_10000_cifar100_RLmodel"
        #PATH = "results/19036/ER_ran_ran_testBch100_RL_2ratioMemIter_multi-step_7_dim_random_numRuns50_20_5000_cifar100_RLmodel"
        # "results/19037/ER_ran_ran_testBch100_RL_2ratioMemIter_multi-step_7_dim_random_critic32_2_" \
        # "ERbch50_Done50_crtBchSize50_numRuns5_orderRnd_20_5000_cifar100_RLmodel"
        # "ER_ran_ran_testBch100_RL_2ratioMemIter_multi-step_7_dim_random_numRuns10_20_5000_cifar100_RLmodel"

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint)
    def initialize_q(self):
        print("initialize q")
        self.initialize_critic(self.params, self.action_num, self.ob_dim)
        # self.training_steps = 0

        # network = self.q_function
        # for layer in network.children():
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        #
        # network = self.q_function_target
        # for layer in network.children():
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()

    def update_q_target(self):

        for target_param, param in zip(
                self.q_function_target.parameters(), self.q_function.parameters()
        ):
            target_param.data.copy_(param.data)


    def train_batch(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch,training_steps):

        if self.params.reward_type[:10] == "multi-step":

            with torch.no_grad():
                q_s = self.q_function_target(next_state_batch)

                q_max = torch.max(q_s, axis=1)[0]

                td_target = reward_batch + self.gamma * torch.max(q_s, axis=1)[0] * (1 - done_batch)

        else:
            td_target = reward_batch

        td_target = td_target.float()


        n = state_batch.shape[0]
        #print(self.RL_agent.ER_batchsize, n)

        predict_q = self.q_function(state_batch)[torch.arange(n), action_batch].float()
        td_target = maybe_cuda(td_target)

        assert predict_q.shape == td_target.shape

        rl_loss = torch.nn.functional.mse_loss(predict_q, td_target, reduction="mean")

        # rl_loss = torch.nn.SmoothL1Loss()(predict_q, td_target )

        rl_opt = torch.optim.Adam(self.q_function.parameters(),
                                  lr=self.rl_lr.value(training_steps),
                                  weight_decay=self.rl_wd)

        rl_opt.zero_grad()
        rl_loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_function.parameters(), self.grad_norm_clipping)
        rl_opt.step()
        #print("train RL, loss", rl_loss.item())
        return rl_loss




