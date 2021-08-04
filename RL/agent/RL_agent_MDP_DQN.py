

import torch
import numpy as np
from RL.agent.RL_agent_MDP import RL_memIter_agent
from utils.utils import maybe_cuda
from RL.critic.critic import critic_class
from RL.dqn_utils import cl_exploration_schedule,critic_lr_schedule




class RL_DQN_agent(RL_memIter_agent):
    def __init__(self, params):
        super().__init__(params)
        self.update_q_target_freq = 1000
        self.epsilon = None




        self.exploration = cl_exploration_schedule(self.total_training_steps,para=self.params.rl_exp_type)
        self.critic = critic_class(params, self.action_num, self.ob_dim,self.total_training_steps,RL_agent=self)


        # if (params.reward_type[:10] == "multi-step"):
        #     self.critic_training_start = params.critic_training_start  # 1000
        # else:
        #     self.critic_training_start = params.ER_batch_size * 4
        self.critic_training_start = params.critic_training_start

    def take_greedy_action(self,state):


        with torch.no_grad():
            state = maybe_cuda(state)
            q_values = self.critic.q_function(state)[0, :]
            action = torch.argmax(q_values).detach().cpu().numpy()

            # q_value_max = torch.max(q_values)
            # #print(q_values)
            # num_max = torch.sum(q_values == q_value_max)
            #
            # prob = (q_values == q_value_max) / num_max
            # prob = prob.cpu().numpy()
            # if (np.sum(prob) != 1):
            #     print(q_value_max)
            #     print(prob)
            #
            # action = np.random.choice(np.arange(self.action_num), 1, p=prob)[0]
        return action



    def from_action_to_replay_para(self,action):
        return self.action_design_space[action]


    def sample_action(self, state):
        if(self.params.RL_type == "DormantRL" or self.params.RL_type == "NoRL"):
            return None
        self.epsilon = self.exploration.value(self.training_steps)
        rnd = torch.tensor(1).float().uniform_(0, 1).item()
        if(self.params.critic_use_model):
            if (rnd < self.epsilon ):  ## take random action

                action = np.random.randint(0, self.action_num)  # torch.zeros(1).long().random_(0, self.action_num)
                self.greedy = "random"
            else:  ## take greedy action

                action = self.take_greedy_action(state)
                self.greedy = "greedy"
        else:
            if(rnd<self.epsilon or (self.training_steps < self.critic_training_start)): ## take random action
                #action =torch.randint(0,high=121,size=1)## unrepetitive
                action = np.random.randint(0,self.action_num)#torch.zeros(1).long().random_(0, self.action_num)
                self.greedy="random"
            else: ## take greedy action
                #q_values
                self.greedy="greedy"
                action = self.take_greedy_action(state)

        self.real_action_list.append(action)
        return  action







    def update_agent(self,reward,state,action,next_state,done,):
        if(self.params.RL_type =="DormantRL"):
            return
        self.real_reward_list.append(reward)
        self.training_steps += 1
        ## DQN

        # ## add <state, action, reward, next_state> into memory
        self.ExperienceReplayObj.store(state,action,reward,next_state,done)



        ## sample batch from replay memory to train network


        if(self.ExperienceReplayObj.can_sample() == False):
            return
        if(self.training_steps < self.critic_training_start): return
        for i in range(self.RL_training_iters):
            state_batch, action_batch, reward_batch, next_state_batch,done_batch = self.ExperienceReplayObj.sample(self.ER_batchsize)
            state_batch = maybe_cuda(torch.from_numpy(state_batch))
            #print(action_batch)
            action_batch = maybe_cuda(torch.from_numpy(action_batch))
            reward_batch = maybe_cuda(torch.from_numpy(reward_batch))
            next_state_batch = maybe_cuda(torch.from_numpy(next_state_batch))
            done_batch = maybe_cuda(torch.from_numpy(done_batch))
            rl_loss = self.critic.train_batch(state_batch,action_batch,reward_batch,next_state_batch,done_batch,self.training_steps)




        self.mse_training_loss.append(rl_loss.item())
    

        if(self.params.reward_type[:10] == "multi-step" and  self.training_steps % self.update_q_target_freq == 0):
            self.critic.update_q_target()


    def save_RL_stats(self,prefix):
        arr = np.array(self.mse_training_loss)
        np.save(prefix + "mse_loss.npy", arr)

        arr = np.array(self.real_reward_list)
        np.save(prefix + "real_reward_list.npy", arr)

        arr = np.array(self.real_action_list)
        np.save(prefix + "real_action_list.npy", arr)

        # save model
        torch.save(self.critic.q_function.state_dict(), prefix + "RLmodel")

        #todo save q functions

