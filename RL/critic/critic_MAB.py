from RL.pytorch_util import  build_mlp

from RL.dqn_utils import critic_lr_schedule
import torch
from utils.utils import maybe_cuda
from unused.lstm import LSTM_critic

class critic_class_MAB(object):
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
        if (self.params.q_function_type == "mlp"):
            self.q_function = build_mlp(
                ob_dim,
                action_num,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.q_function_target = build_mlp(
                ob_dim,
                action_num,
                n_layers=self.n_layers,
                size=self.size,
            )
        elif(self.params.q_function_type == "lstm"):


            input_size = ob_dim  # number of features
            hidden_size = self.params.critic_layer_size #200  # number of features in hidden state
            num_layers = self.params.critic_nlayer #2  # number of stacked lstm layers

            num_classes = action_num  # number of output classes
            if (self.params.state_feature_type[-5:] == "4time"):
                seq_len = 4
                input_size = int(ob_dim /4)
            else:
                seq_len = 1

            self.q_function = LSTM_critic(num_classes, input_size, hidden_size, num_layers,seq_len)
            self.q_function_target = LSTM_critic(num_classes, input_size, hidden_size, num_layers, seq_len)
                           # our lstm class
            # self.q_function = build_lstm(
            #     input_dim=ob_dim,
            # hidden_dim=self.size,
            # output_size=1,
            # n_layers=self.n_layers,
            # seq_len=4)
            # self.q_function_target = build_lstm(
            #     input_dim=ob_dim,
            # hidden_dim=self.size,
            # output_size=1,
            # n_layers=self.n_layers,
            # seq_len=4)
        else:
            raise NotImplementedError ("undefined q function type",self.params.q_function_type)

        self.q_function = maybe_cuda(self.q_function, params.cuda)
        self.rl_opt = torch.optim.Adam(self.get_parameters(),
                                  lr=self.params.critic_lr,
                                  )

        if (self.params.critic_use_model):
            self.load_critic_model(self.q_function)


        self.q_function_target = maybe_cuda(self.q_function_target, params.cuda)
        if (self.params.critic_use_model):
            self.update_q_target()



        return self.q_function, self.q_function_target
    def reset_parameters(self):
        torch.nn.init.kaiming_normal(self.q_function.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def load_critic_model(self, model):
        print("!!! load pre-trained model")
        prefix="RLER1_MIR_ran_mIter3_splitno_testBch100_RL2rmemIter_33_test_loss_0.0_new_old6mn_org_qstart100" \
               "_erb50_exp_nxtBtch_0.50.1_recent2_wd-6_test_bug_nc_20_10000_cifar100_RLmodel"
        PATH = "results/59054/"+prefix
        #PATH= "results/59054/RLER1_MIR_ran_mIter3_testBch100_RL2rmemIter_33_test_loss_0.0_new_old6mn_nxtBtch_recent2_misc4_nc_20_10000_cifar100_RLmodel"
        #PATH = "results/19036/ER_ran_ran_testBch100_RL_2ratioMemIter_multi-step_7_dim_random_numRuns50_20_5000_cifar100_RLmodel"
        # "results/19037/ER_ran_ran_testBch100_RL_2ratioMemIter_multi-step_7_dim_random_critic32_2_" \
        # "ERbch50_Done50_crtBchSize50_numRuns5_orderRnd_20_5000_cifar100_RLmodel"
        # "ER_ran_ran_testBch100_RL_2ratioMemIter_multi-step_7_dim_random_numRuns10_20_5000_cifar100_RLmodel"

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint)
    def get_parameters(self):
        return self.q_function.parameters()
    def compute_q(self,obs,func):
        # obs=obs.float().cuda()
        # func = func.float()
        #print(obs.device,func.device)
        #assert False
        if(self.params.q_function_type  =="lstm"):
            if(self.params.state_feature_type[-5:]=="4time"):
                seq_obs = torch.reshape(obs, (obs.shape[0], 4, -1))
            else:

                seq_obs = torch.reshape(obs,(obs.shape[0],1,obs.shape[1]))


            return func.forward(seq_obs)
        else:
            return func(obs)
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


