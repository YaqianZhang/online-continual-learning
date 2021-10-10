import torch
import numpy as np

from RL.env.RL_env_base import Base_RL_env
class RL_env_MDP(Base_RL_env):
    def __init__(self,params,model,RL_agent,CL_agent):
        super().__init__(params,model,RL_agent,CL_agent)



        if(self.params.RL_type == "DormantRL"):
            self.basic_mem_iters = self.params.mem_iters
            self.basic_i_ratio = self.params.incoming_ratio
            self.basic_m_ratio = self.params.mem_ratio
        else:
            self.basic_mem_iters  = 1
            self.basic_i_ratio = 1
            self.basic_m_ratio = 1
        self.initialize()

    def get_basic_replay_para(self):
        replay_para = {'mem_iter': self.basic_mem_iters,
                       'mem_ratio': self.basic_m_ratio,
                       'incoming_ratio': self.basic_i_ratio}
        return replay_para

    def initialize(self):

        self.RL_mem_iters =0
        self.RL_incoming_ratio = 1
        self.RL_mem_ratio = 1

        # self.state = None
        # self.action = None
        # self.reward = None






    def check_episode_done(self,stats_dict,):
        if(self.params.dynamics_type == "within_batch"):
            if(stats_dict["mini_iter"]==(self.params.mem_iters-1)):
                done = 1
            else:
                done = 0


        else:
            i = self.CL_agent.incoming_batch ['batch_num'] #stats_dict['batch_num']
            if ((i + 1) % self.params.done_freq == 0):
                done = 1
            else:
                done = 0
        return done

    def get_replay_action(self,action):

        if(self.params.RL_type == "RL_memIter"):
            self.RL_mem_iters  = self.RL_agent.from_action_to_replay_para(action)
        elif(self.params.RL_type == "RL_ratio"):
            self.RL_incoming_ratio = self.RL_agent.from_action_to_replay_para(action)
        elif(self.params.RL_type == "RL_ratioMemIter"):
            self.RL_mem_iters,self.RL_incoming_ratio = self.RL_agent.from_action_to_replay_para(action)
        elif(self.params.RL_type == "RL_actor"):
            self.RL_incoming_ratio = action
            self.RL_mem_ratio = 1
            self.RL_mem_iters=1
        elif(self.params.RL_type == "RL_2ratioMemIter" or self.params.RL_type == "RL_ratio_1para"):
            self.RL_mem_iters,self.RL_incoming_ratio,self.RL_mem_ratio = self.RL_agent.from_action_to_replay_para(action)
            # elif(self.params.RL_type == "RL_adpRatio"):
            #
            #     ratio_list=[]
            #     for i in range(len(batch_y)):
            #         task_id = self.get_task_id(batch_y) ## todo:zyq task_id
            #         action = self.RL_agent.sample_action(state,task_id)
            #         ratio = self.RL_agent.action_design_space[action]
            #         ratio_list.append(ratio)
            #     self.add_mem_ratio = ratio_list
            #
            #     action = self.RL_agent.sample_action(state, task_id_new) ## todo:zyq task_id
            #     self.self.add_incoming_ratio = self.RL_agent.action_design_space[action]
            #     #self.self.add_incoming_ratio,
            #     self.action = 0

        elif (self.params.RL_type == "DormantRL"):
            pass
        else:
            raise NotImplementedError("Undefined RL type in set replay action", self.params.RL_type)

        replay_para = {'mem_iter': self.RL_mem_iters,
                       'mem_ratio': self.RL_mem_ratio,
                       'incoming_ratio': self.RL_incoming_ratio, }

        return replay_para



    def step(self,action,virtual=False,use_set_mem=False):
        self.CL_agent.replay_para = self.get_replay_action(action)
        #end_stats = self.CL_agent.replay_and_evaluate(replay_para)
        if(virtual):
            end_stats = self.CL_agent.virtual_joint_training(self.CL_agent.replay_para, TEST=True,use_set_mem=use_set_mem)
        else:
            end_stats = self.CL_agent.joint_training(self.CL_agent.replay_para, TEST=True,use_set_mem=use_set_mem)
        return end_stats
















