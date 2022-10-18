

import torch
import numpy as np
from RL.agent.RL_agent_MDP_DQN_hp import RL_DQN_agent_hp
from utils.utils import maybe_cuda
from RL.critic.critic import critic_class
from RL.critic.task_critic import task_critic_class
from RL.critic.actor_critic import actor_critic_class
from RL.dqn_utils import cl_exploration_schedule


class RL_MAB_agent(RL_DQN_agent_hp):
    def __init__(self, params,action_num,state_dim,RL_replay=None):
        super().__init__(params,action_num,state_dim,RL_replay)
        self.update_q_target_freq = params.update_q_target_freq #1000
        self.epsilon = None
        self.batch_num = None
        self.predicted_q=[]
        self.real_q=[]
        self.greedy_action=[]
        self.select_batch_num=[]
        self.action_num=action_num
        self.out_range = False




        self.exploration = cl_exploration_schedule(self.total_training_steps,para=self.params.rl_exp_type)

        self.q_history=[] #
        for i in range(action_num):
            self.q_history.append([])
    def compute_q_values(self):
        q=[]
        for i in range(self.action_num):
            if(len(self.q_history[i])==0):
                rwd = 10
            else:
                rwd = np.mean(self.q_history[i][-self.params.MAB_reward_len:]) ## todo:try other q update methods
            q.append(rwd)
        return q

    ## override
    def _take_greedy_action(self,state):
        #print("greedy")
        q_values =self.compute_q_values()



        q_value_max = np.max(q_values)
        #print(q_values)
        num_max = np.sum(q_values == q_value_max)


        prob = (q_values == q_value_max) / num_max
        #prob = prob.cpu().numpy()
        if (np.sum(prob) != 1):
            print(q_value_max)
            print(prob)

        action = np.random.choice(np.arange(self.action_num), 1, p=prob)[0]
        return action





    def update(self,i,state,action,reward,next_state):
        if(action==None or reward ==None):
            return
        self.real_reward_list.append(reward)
        self.RL_running_steps += 1
        self.RL_agent_update_steps += 1

        self.q_history[action].append(reward)

    # def update_agent(self,reward,state,action,next_state,done,):
    #     ## update Q values
    #     if(self.params.RL_type =="DormantRL"):
    #         return
    #
    #     self.q_history[action].append(reward)







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

