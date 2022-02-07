#
#
import numpy as np
import torch
from RL.agent.RL_agent_MDP_DQN_hp import RL_DQN_agent_hp
from utils.utils import maybe_cuda
from RL.RL_env_design import RL_env
from RL.RL_replay_base import RL_replay
class RL_replay_MAB(RL_replay):
    ## bridge RL and CL agent


    def __init__(self,params,close_loop):
        super(RL_replay_MAB,self).__init__(params,close_loop)
        self.RL_agent = RL_DQN_agent_(params,action_num,ob_dim,RL_replay=self)







    def make_replay_decision_update_RL(self,i): ## action
        if(self.close_loop_CL.CL_agent.task_seen == self.params.start_task):
            return None


        if (i <= self.params.RL_start_batchstep):
            # if(self.task_seen ==0):
            replay_para = {'mem_iter': self.params.mem_iters,
                           'mem_ratio': self.params.task_start_mem_ratio,
                           "randaug_M":self.params.randaug_M,
                           'incoming_ratio': self.params.task_start_incoming_ratio, }

            print("###",replay_para)
            return replay_para
        if(self.close_loop_CL.test_stats == None or self.close_loop_CL.train_stats == None ):
            return None



        self.RL_agent.update( i,self.state,self.action,self.reward,self.next_state)




        self.action = self.RL_agent.sample_action(self.state) #self.sample_action(self.state) ## dormant RL
        selected_action= self._get_replay_para(self.action)
        return selected_action










