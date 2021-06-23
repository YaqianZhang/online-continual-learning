import torch
from utils.utils import maybe_cuda


import numpy as np

class RL_agent(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.epsilon = 0.1
        self.alpha = 0.9 # Q update parameters
        self.selected_num = 1



        self.action_num= params.action_size
        self.action_len = maybe_cuda(torch.LongTensor(1).fill_(self.action_num))

        #self.action_num = params.action_size * self.params.action_size
        #self.action_len = maybe_cuda(torch.LongTensor(1).fill_(self.action_num))
        self.q_function = maybe_cuda(torch.FloatTensor(self.action_num).fill_(1))

        self.current_action = torch.zeros(1).long().random_(0, self.action_num) ## todo:set initial current action to be MIR
        self.action_list= []
    def save_q(self,prefix):

        arr = self.q_function.detach().cpu().numpy()

        np.save(prefix + "q_function.npy", arr)
        #np.save(prefix+"last_action.npy",self.current_action.detach().cpu().numpy())

    def save_action(self,prefix):
        arr = np.array(self.action_list)
        np.save(prefix + "action_list.npy", arr)
        print("save action")


    def sample_action(self):
        rnd = torch.tensor(1).float().uniform_(0, 1 ).item()
        if(rnd<self.epsilon): ## take random action
            #action =torch.randint(0,high=121,size=1)## unrepetitive
            action = torch.zeros(1).long().random_(0, self.action_num)
        else: ## take greedy action
            #q_values
            action = torch.sort(self.q_function,descending=True)[1][:self.selected_num] ## select action with largest Q values
        self.current_action = action
        self.action_list.append(action.cpu())
        return action



    def update_agent(self,reward):
        #if(reward<0):return
        action_num = self.current_action
        self.q_function[action_num] += self.alpha *(reward - self.q_function[action_num])

    def initialize_q(self):
        self.q_function = maybe_cuda(torch.FloatTensor(self.action_num).fill_(1))


class RL_agent_2dim(RL_agent):
    def __init__(self,params):
        super().__init__(params)
        self.action_num = params.action_size * self.params.action_size
        self.action_len = maybe_cuda(torch.LongTensor(1).fill_(self.action_num))


class RL_memIter_agent(RL_agent):
    def __init__(self, params):
        super().__init__()

        self.action_num= 5 ## memITer
        self.action_len = maybe_cuda(torch.LongTensor(1).fill_(self.action_num))


        #self.q_function = maybe_cuda(torch.FloatTensor(self.action_num).fill_(1)) ## todo Q network

        self.current_action = torch.zeros(1).long().random_(0, self.action_num) ## todo:set initial current action to be MIR
        self.action_list= []

    # def sample_action(self,state):
    #     action = self.greedy_policy(state)
    #     return action
    #
    # def greedy_policy(self,state):
    #     rnd = torch.tensor(1).float().uniform_(0, 1 ).item()
    #     if(rnd<self.epsilon): ## take random action
    #         actions =torch.randint(0,high=self.action_len,size=self.selected_num)## unrepetitive
    #     else: ## take greedy action
    #         #q_values
    #         actions = torch.sort(self.q_function)[1][:self.selected_num]
    #     return actions
