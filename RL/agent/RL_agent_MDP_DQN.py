

import torch
import numpy as np
from RL.agent.RL_agent_MDP import RL_memIter_agent
from utils.utils import maybe_cuda
from RL.critic.critic import critic_class
from RL.critic.task_critic import task_critic_class
from RL.critic.actor_critic import actor_critic_class
from RL.dqn_utils import cl_exploration_schedule,critic_lr_schedule




class RL_DQN_agent(RL_memIter_agent):
    def __init__(self, params):
        super().__init__(params)
        self.update_q_target_freq = params.update_q_target_freq #1000
        self.epsilon = None
        self.batch_num = None
        self.predicted_q=[]
        self.real_q=[]
        self.greedy_action=[]
        self.select_batch_num=[]
        self.out_range = False




        self.exploration = cl_exploration_schedule(self.total_training_steps,para=self.params.rl_exp_type)
        if(params.critic_type == "task_critic"):
            self.critic = task_critic_class(params, self.action_num, self.ob_dim, self.total_training_steps, RL_agent=self)

        elif(params.critic_type == "critic"):

            self.critic = critic_class(params, self.action_num, self.ob_dim,self.total_training_steps,RL_agent=self)
        elif (params.critic_type == "actor_critic"):

            self.critic = actor_critic_class(params, self.action_num, self.ob_dim, self.total_training_steps, RL_agent=self)

        else:
            raise NotImplementedError("not defined critic type",params.critic_type)

        # if (params.reward_type[:10] == "multi-step"):
        #     self.critic_training_start = params.critic_training_start  # 1000
        # else:
       #self.critic_training_start = params.ER_batch_size * 4
        self.critic_training_start = params.critic_training_start

    def take_greedy_action(self,state):
        if(self.params.state_feature_type == "new_old5_4time"):
            state = self.ExperienceReplayObj.append_state(state)



        with torch.no_grad():
            #print("take greedy action",self.RL_agent_update_steps )
            state = maybe_cuda(state)
            output = self.critic.compute_q(state,self.critic.q_function)
            q_values= output[0, :]
            self.predicted_q.append(torch.max(q_values).detach().cpu().numpy())




            #action = torch.argmax(q_values).detach().cpu().numpy()

            # if(state[0][0]==0):
            #     print(state)
            #     print("Q_VALUES",q_values)
            #     print(action)
            #     assert False

            q_value_max = torch.max(q_values)
            #print(q_values)
            num_max = torch.sum(q_values == q_value_max)
            if(num_max <1):
                print(state,output,q_value_max,q_values,num_max)
                for name, param in self.critic.q_function.named_parameters():

                    print(name, param.data)
                assert False


            prob = (q_values == q_value_max) / num_max
            prob = prob.cpu().numpy()
            if (np.sum(prob) != 1):
                print(q_value_max)
                print(prob)

            action = np.random.choice(np.arange(self.action_num), 1, p=prob)[0]
        return action



    def from_action_to_replay_para(self,action):
        return self.action_design_space[action]


    def clip_action(self,value,max_value,min_value):
        if(value>max_value):
            return max_value, True
        if(value<min_value):
            return min_value, True
        return value, False

    def sample_continuous_action(self,state):
        state = maybe_cuda(state)
        mu = self.critic.actor(state)
        sigma = [0.01,0.5]
        ratio,iter = np.random.normal(mu, sigma, 1)
        print(sigma,ratio,iter)



        assert False
        ratio, ratio_out_range = self.clip_action(ratio, 0.1,1.5)
        ratio,iter_out_range = self.clip_action(iter,0,5)

        if ratio_out_range or iter_out_range:
            self.out_range = True
        else:
            self.out_range = False

        return ratio,iter





    def sample_action(self, state):
        if(self.params.RL_type == "DormantRL" or self.params.RL_type == "NoRL"):
            return None
        if (self.params.save_prefix == "non_stationary"):
            self.epsilon = self.exploration.value(self.batch_num)
        else:
            self.epsilon = self.exploration.value(self.RL_running_steps)

        if(self.params.actor_type == "random"):
            self.epsilon = 1




        rnd = torch.tensor(1).float().uniform_(0, 1).item()


        if(self.params.critic_use_model):
            if (rnd < self.epsilon ):  ## take random action

                action = np.random.randint(0, self.action_num)  # torch.zeros(1).long().random_(0, self.action_num)
                self.greedy = "random"
            else:  ## take greedy action

                action = self.take_greedy_action(state)
                self.greedy = "greedy"
        else:
            if(rnd<self.epsilon or self.RL_agent_update_steps<30):# (self.RL_running_steps < self.critic_training_start)): ## take random action
                #action =torch.randint(0,high=121,size=1)## unrepetitive
                action = np.random.randint(0,self.action_num)#torch.zeros(1).long().random_(0, self.action_num)
                self.greedy="random"
            else: ## take greedy action
                #q_values
                self.greedy="greedy"
                action = self.take_greedy_action(state)

        self.real_action_list.append(action)
        return  action
    def take_random_action(self):
        return np.random.randint(0,self.action_num)

    def take_base_action(self):
        return self.base_action







    def update_agent(self,reward,state,action,next_state,done,):
        if(self.params.RL_type =="DormantRL"):
            return

        ## DQN

        # ## add <state, action, reward, next_state> into memory
        #self.ExperienceReplayObj.store(state,action,reward,next_state,done)



        ## sample batch from replay memory to train network


        if(self.ExperienceReplayObj.can_sample() == False):
            return

        if(self.RL_running_steps < self.critic_training_start): return
        if(self.params.RL_agent_update_flag):
            self.RL_agent_update_steps += 1

        for i in range(self.RL_training_iters):

            state_batch, action_batch, reward_batch, next_state_batch,done_batch = self.ExperienceReplayObj.sample(self.ER_batchsize)
            state_batch = maybe_cuda(torch.from_numpy(state_batch))
            #print(action_batch)
            action_batch = maybe_cuda(torch.from_numpy(action_batch))
            reward_batch = maybe_cuda(torch.from_numpy(reward_batch))

            next_state_batch = maybe_cuda(torch.from_numpy(next_state_batch))
            done_batch = maybe_cuda(torch.from_numpy(done_batch))

            if (self.params.save_prefix != "non_stationary"):
                rl_loss = self.critic.train_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                                                  self.RL_running_steps)


            else:
                rl_loss = self.critic.train_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch,
                                                  self.batch_num)




        self.mse_training_loss.append(rl_loss.item())

        if(self.params.episode_type == "multi-step" and  (self.RL_agent_update_steps  % self.params.update_q_target_freq == 0)):
            self.critic.update_q_target()


    def save_RL_stats(self,prefix):
        arr = np.array(self.mse_training_loss)
        np.save(prefix + "mse_loss.npy", arr)

        arr = np.array(self.real_reward_list)
        np.save(prefix + "real_reward_list.npy", arr)

        arr = np.array(self.real_q)
        np.save(prefix + "real_q.npy", arr)
        arr = np.array(self.greedy_action)
        np.save(prefix + "greedy_action.npy", arr)

        arr = np.array(self.select_batch_num)
        np.save(prefix + "select_batch_num.npy", arr)

        arr = np.array(self.predicted_q)
        np.save(prefix + "predicted_q.npy", arr)



        arr = np.array(self.real_action_list)
        np.save(prefix + "real_action_list.npy", arr)

        # save model
        torch.save(self.critic.q_function.state_dict(), prefix + "RLmodel")

        #todo save q functions

