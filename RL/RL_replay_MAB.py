#
#
import numpy as np
import torch
from RL.agent.RL_agent_MDP_DQN_hp import RL_DQN_agent_hp
from RL.RL_replay_base import RL_replay
class RL_replay_MAB(RL_replay):

    def __init__(self,params,):
        super(RL_replay_MAB, self).__init__( params)
        self.q_values = torch.zeros(self.action_num)



    def update_RL_agent(self):

        self.q_values[self.action] +=  0.1 * (self.reward - self.q_values[self.action])
        # if(self.state == None):
        #     return
        #
        #
        #
        #
        # i = self.train_stats['batch_num']  # stats_dict['batch_num'] ## next stat
        # if ((i+1 ) % self.params.done_freq == 0):
        #     done = 1
        # else:
        #     done = 0
        #
        #
        #
        #
        # self.RL_agent.real_reward_list.append(self.reward)
        # self.RL_agent.RL_running_steps += 1
        # self.RL_agent.ExperienceReplayObj.store(self.state, self.action, self.reward, self.next_state, done)
        # self.RL_agent.update_agent(self.reward, self.state,
        #                            self.action, self.next_state, done)  ## update RL agent






    def make_replay_decision(self,): ## action

        eps = 0.1
        if(np.random.rand(1)<eps):
            self.action = np.random.randint(0,self.action_num,1)[0]
            #print("!!random",self.action)
        else:
            b = self.q_values.numpy()
            self.action = np.random.choice(np.flatnonzero(b == b.max()))

            #print("!!greedy",self.action)

        selected_action= self._get_replay_para(self.action)
        self.RL_agent.real_action_list.append(self.action)
        return selected_action















