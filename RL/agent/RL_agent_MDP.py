

import torch
import numpy as np
from RL.agent.RL_agent_base import RL_base_agent
from utils.utils import maybe_cuda
from RL.pytorch_util import  build_mlp
from RL.RL_buffer import RL_ExperienceReplay

class RL_memIter_agent(RL_base_agent):
    def __init__(self, params):
        super().__init__(params)

        self.action_design_space  = self.initialize_actor(params)
        self.action_num = len(self.action_design_space)  # params.mem_iter_max - params.mem_iter_min +1 ## memITer 0,1,2


        self.ob_dim = self.initialize_state(params.state_feature_type)
        self.initialize_critic(params,self.action_num,self.ob_dim)

        self.mse_training_loss =[]
        self.real_reward_list=[]
        self.real_action_list = []

    def initialize_actor(self,params):
        self.action_start = params.mem_iter_min
        if(params.RL_type == "RL_ratio"):
            #self.action_design_space = np.linspace(0.1,2,10)
            self.action_design_space = np.linspace(0, 2, 10)
            print("action space",self.action_design_space)

        elif (params.RL_type == "RL_memIter"):
            self.action_design_space = np.arange(params.mem_iter_min,params.mem_iter_max+1)

        elif(params.RL_type  == "RL_ratioMemIter"):
            self.mem_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
            self.ratio_design_space = [0.0,0.1,0.5,1.0,1.5]

            self.action_design_space = []
            for i in range(len(self.mem_design_space)):
                for j in range(len(self.ratio_design_space)):
                    self.action_design_space.append((self.mem_design_space[i],self.ratio_design_space[j]))
        elif(params.RL_type == "DormantRL"):
            self.action_design_space = []

        else:
            raise NotImplementedError("Undefined action space for RL type", params.RL_type)


        return self.action_design_space

    def initialize_state(self,state_feature_type):
        ######### state ###############
        if (state_feature_type == "4_dim" or state_feature_type == "4_loss"):
            ob_dim = 4  ## train_incoming_acc, train_mem_acc, test_mem_acc,batch id (task id?)
        elif (state_feature_type == "3_dim" or state_feature_type == "3_loss"):
            ob_dim = 3
        elif (state_feature_type == "6_dim"):
            ob_dim = 6
        elif (state_feature_type == "same_batch"):
            ob_dim = 6
        elif (state_feature_type == "same_batch_7_dim"):
            ob_dim = 7
        elif (state_feature_type == "7_dim"):
            ob_dim = 7
        else:
            raise NotImplementedError("state type is not defined" + state_feature_type)
        print("obs_dim", ob_dim)
        return ob_dim

    def initialize_critic(self,params,action_num,ob_dim):
        self.n_layers = params.critic_nlayer
        self.size = params.critic_layer_size
        self.ER_batchsize = params.ER_batch_size
        if (self.params.episode_type == "batch"):
            self.rl_lr = 0.0005
            self.rl_wd = 0.0001
        else:
            self.rl_lr = 5 * 10 ** (-4)  # -4
            self.rl_wd = 1 * 10 ** (-4)  # -4

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
        self.training_steps = 0
        self.update_q_target_freq = 100
        self.ExperienceReplayObj = RL_ExperienceReplay(params)
        self.RL_training_iters = params.critic_training_iters
        return self.q_function,self.q_function_target


    def sample_action(self, state):

        rnd = torch.tensor(1).float().uniform_(0, 1 ).item()
        if(rnd<self.epsilon): ## take random action
            #action =torch.randint(0,high=121,size=1)## unrepetitive
            action = torch.zeros(1).long().random_(0, self.action_num)
        else: ## take greedy action
            #q_values
            with torch.no_grad():
                q_values = self.q_function(state)[0,:]

                q_value_max = torch.max(q_values)
                num_max =torch.sum(q_values == q_value_max)

                prob = (q_values == q_value_max)/num_max

                action = np.random.choice(np.arange(self.action_num), 1, p=prob)[0]
                #print(action.shape)

            # sorted_action= torch.sort(self.q_function(state),axis = 1,descending=True)[1]## select action with largest Q values
            # action=sorted_action[0,0]

        self.real_action_list.append(action)
        return  action
    def from_action_to_memIter(self,action):
        return self.action_design_space[action]
    def from_action_to_ratio(self,action):
        return self.action_design_space[action]
    def from_action_to_ratio_memIter(self,action):
        i,j = self.action_design_space[action]

        return i,j


    def initialize_q(self):

        network = self.q_function
        for layer in network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        network = self.q_function_target
        for layer in network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def update_q_target(self):
        print("double q update",self.training_steps)
        for target_param, param in zip(
                self.q_function_target.parameters(), self.q_function.parameters()
        ):
            target_param.data.copy_(param.data)


    def update_agent(self,reward,state,action,next_state,done,):
        if(self.params.RL_type =="DormantRL"):
            return
        self.real_reward_list.append(reward)
        self.training_steps += 1
        ## DQN
        gamma = 0.99
        ## add <state, action, reward, next_state> into memory
        self.ExperienceReplayObj.store(state,action,reward,next_state)


        ## sample batch from replay memory to train network


        if(self.ExperienceReplayObj.can_sample() == False):
            return
        for i in range(self.RL_training_iters):
            state_batch, action_batch, reward_batch, next_state_batch = self.ExperienceReplayObj.sample(self.ER_batchsize)
            state_batch = torch.from_numpy(state_batch)
            action_batch = torch.from_numpy(action_batch)
            reward_batch = torch.from_numpy(reward_batch)
            next_state_batch = torch.from_numpy(next_state_batch)


            # backprogration
            if(self.params.reward_type == "multi-step"):
                if(done):
                    td_target = reward_batch
                else:
                    with torch.no_grad():
                        q_s = self.q_function_target(next_state_batch)

                        q_max = torch.max(q_s,axis = 1)[0]

                        td_target = reward_batch +gamma * torch.max(q_s,axis = 1)[0]

            else:
                td_target = reward_batch


            td_target = td_target.float()

            predict_q= self.q_function(state_batch)[torch.arange(self.ER_batchsize),action_batch].float()


            rl_loss = torch.nn.functional.mse_loss(predict_q,td_target,reduction="mean")
            rl_opt = torch.optim.Adam(self.q_function.parameters(),
                                     lr=self.rl_lr,
                                     weight_decay=self.rl_wd)

            rl_opt.zero_grad()
            rl_loss.backward()
            rl_opt.step()


        self.mse_training_loss.append(rl_loss.item())
        # if(rl_loss.item()>20):
        #     print(state_batch,predict_q,td_target)
        #     assert False


        ## double q network

        if(self.training_steps % self.update_q_target_freq == 0):
            self.update_q_target()


    def save_RL_stats(self,prefix):
        arr = np.array(self.mse_training_loss)
        np.save(prefix + "mse_loss.npy", arr)

        arr = np.array(self.real_reward_list)
        np.save(prefix + "real_reward_list.npy", arr)

        arr = np.array(self.real_action_list)
        np.save(prefix + "real_action_list.npy", arr)

        ## save model
        #torch.save(self.q_function.state_dict(), prefix + "RLmodel")

        #todo save q functions
