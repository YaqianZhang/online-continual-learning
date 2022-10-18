#
#
import numpy as np
import torch
from RL.agent.RL_agent_MDP_DQN_hp import RL_DQN_agent_hp
from RL.agent.RL_agent_MAB import RL_MAB_agent
from RL.agent.RL_agent_MDP_stop import RL_MDP_agent_stop
from utils.utils import maybe_cuda
from RL.RL_env_design import RL_env
class RL_replay(object):
    ## bridge RL and CL agent

    def __init__(self,params,close_loop):
        self.params = params

        #self.RL_env = RL_env

        #RL_memIter_agent(params)

        #self.memoryManager = memoryManager

        self.close_loop_CL = close_loop
        self.RL_env = RL_env(params,RL_replay=self)
        if(self.params.stop_state_type == "4-dim"):
            ob_dim = 4 ## train stats #self.RL_env.get_state_dim()
        elif (self.params.stop_state_type == "2-dim"):
            ob_dim = 2
        else:
            ob_dim = 5
        self.ob_dim = ob_dim
        action_num = 2 #stop / continue   ##self.RL_env.get_action_dim()
        if(self.params.RL_type == "RL_MDP"):
            self.RL_agent = RL_DQN_agent_hp(params, action_num, ob_dim, RL_replay=self)
        elif(self.params.RL_type == "RL_MAB"):
            self.RL_agent = RL_MAB_agent(params,action_num,ob_dim, RL_replay=self)
        elif(self.params.RL_type =="RL_MDP_stop"):

            self.RL_agent = RL_MDP_agent_stop(params, action_num, ob_dim, RL_replay=self)
        else:
            self.RL_agent = None
            #raise NotImplementedError(self.params.RL_type)
        self.state = None
        self.action = None
        self.reward = None
        self.total_reward = 0
        self.return_list = []
        self._init_replay_para()
        self.prev_train_stats = None



    def _init_replay_para(self):
        self.RL_mem_iters = self.params.mem_iters
        self.RL_mem_ratio = self.params.mem_ratio
        self.RL_incoming_ratio = self.params.incoming_ratio


    def set_immediate_reward(self,train_stats, STOP_FLAG):
        if(self.params.immediate_reward == "penalty"):
            iteration_penalty = self.params.iter_penalty
            if(STOP_FLAG==1):
                reward = 0
            else:
                reward = -iteration_penalty
        elif(self.params.immediate_reward == "train_acc"):
            if(self.prev_train_stats != None):
                reward = train_stats["acc_mem"] - self.prev_train_stats["acc_mem"]
            else:
                reward = train_stats["acc_mem"]  ### the first iteration
            self.prev_train_stats = train_stats
        elif(self.params.immediate_reward == "debug"):
            reward = 1
        elif(self.params.immediate_reward == "mem_inc_ratio"):
            reward = - np.abs(train_stats["loss_mem"]-train_stats["loss_incoming"])
        self.reward = reward
        return reward


    def set_end_reward(self,train_stats):

        if(self.params.immediate_reward == "mem_inc_ratio"):
            reward = - np.abs(train_stats["loss_mem"]-train_stats["loss_incoming"])
        else:
            test_acc, test_loss = self.close_loop_CL.compute_testmem_loss()
            reward = test_acc
        self.reward = reward
        return reward
        #self.reward = self.RL_env.get_reward(self.close_loop_CL.test_stats,self.close_loop_CL.test_stats_prev)

        # if(mem_iter != None and self.reward != None):
        #
        #     self.reward -= self.params.reward_rg*mem_iter
        # self.close_loop_CL.test_stats_prev = self.close_loop_CL.test_stats
        # if(self.RL_agent.greedy == "greedy"):
        #     self.RL_agent.real_q.append(self.reward)
        #     self.RL_agent.greedy_action.append(self.action)

    def train_agent(self):
        self.RL_agent.update_agent()


    # ## action
    # def _from_action_to_replay_para(self,action):
    #     return self.RL_env.action_design_space[action]
    # def _get_replay_para(self, action):
    #     self.raugM = 1
    #     if (self.params.critic_type == "actor_critic"):
    #         if (self.params.hp_action_space == "aug_iter"):
    #
    #
    #             aug = (action[0]-0.1)/1.4*30
    #             replay_para = {'mem_iter': action[1],
    #                            'mem_ratio': self.RL_mem_ratio,
    #                            'incoming_ratio': self.RL_mem_ratio,
    #                            'randaug_M':aug}
    #         elif (self.params.hp_action_space == "ratio_iter"):
    #             replay_para = {'mem_iter': action[1],
    #                            'mem_ratio': self.RL_mem_ratio,
    #                            'incoming_ratio': action[0], }
    #
    #         elif (self.params.hp_action_space == "iter"):
    #             replay_para = {'mem_iter': action[0],
    #                            'mem_ratio': self.RL_mem_ratio,
    #                            'incoming_ratio': self.RL_incoming_ratio }
    #         elif(self.params.hp_action_space == "ratio"):
    #             replay_para = {'mem_iter': self.RL_mem_iters,
    #                            'mem_ratio': self.RL_mem_ratio,
    #                            'incoming_ratio': action[0], }
    #         return replay_para
    #     else:
    #
    #         if (self.params.hp_action_space == "iter"):
    #             self.RL_mem_iters = self._from_action_to_replay_para(action)
    #         elif (self.params.hp_action_space == "ratio"):
    #             self.RL_incoming_ratio = self._from_action_to_replay_para(action)
    #         elif (self.params.hp_action_space == "ratio_iter"):
    #             self.RL_mem_iters, self.RL_incoming_ratio, self.RL_mem_ratio = self._from_action_to_replay_para(
    #             action)
    #
    #         elif (self.params.hp_action_space == "aug_iter"):
    #
    #             self.RL_mem_iters, act, self.RL_mem_ratio = self._from_action_to_replay_para(
    #                 action)
    #             self.raugM = int((act - 0.1) / 1.4 * 30)
    #             self.RL_incoming_ratio=1
    #
    #
    #         elif (self.params.RL_type == "DormantRL"):
    #             pass
    #         else:
    #             raise NotImplementedError("Undefined RL type in set replay action", self.params.RL_type)
    #
    #         replay_para = {'mem_iter': self.RL_mem_iters,
    #                        'mem_ratio': self.RL_mem_ratio,
    #                        'incoming_ratio': self.RL_incoming_ratio,
    #                        'randaug_M':self.raugM}
    #
    #         return replay_para


    def store(self,state,action,reward,next_state,done):
        if(next_state == None):
            next_state = torch.from_numpy(np.array([-1,]*self.ob_dim, dtype=np.float32).reshape([1,-1]))  ## make buffer data type consistent

        self.RL_agent.store(state,action,reward,next_state,done)

    def get_state(self,train_stats,j):
        if(self.params.stop_state_type == "4-dim"):
            state = [train_stats['acc_incoming'],
                 train_stats['acc_mem'],
                 train_stats['loss_incoming'],
                 train_stats['loss_mem'],

                 ]
        elif (self.params.stop_state_type == "2-dim"):
                state = [train_stats['acc_incoming'],
                         train_stats['acc_mem'],
                         # train_stats['loss_incoming'],
                         # train_stats['loss_mem'],

                         ]
        else:
            state = [train_stats['acc_incoming'],
                     train_stats['acc_mem'],
                     train_stats['loss_incoming'],
                     train_stats['loss_mem'],
                     j

                     ]

        return torch.from_numpy(np.array(state,dtype=np.float32).reshape([1, -1]) ) ## important state dim: 1* statedim


    def make_stop_decision(self,state):
        ## compute state from train_stats
        # train_stats = {'acc_incoming': acc_incoming,
        #                'acc_mem': acc_mem,
        #                "loss_incoming": incoming_loss.item(),
        #                "loss_mem": mem_loss.item(),
        #                "batch_num": i,


        self.action = self.RL_agent.sample_action(state) #self.sample_action(self.state) ## dormant RL
        return self.action
        #output the sampled action: stop, not stop

    # def make_replay_decision_update_RL(self,i): ## action
    #     if(self.close_loop_CL.CL_agent.task_seen == self.params.start_task):
    #         return None
    #
    #
    #     if (i <= self.params.RL_start_batchstep):
    #         # if(self.task_seen ==0):
    #         replay_para = {'mem_iter': self.params.mem_iters,
    #                        'mem_ratio': self.params.task_start_mem_ratio,
    #                        "randaug_M":self.params.randaug_M,
    #                        'incoming_ratio': self.params.task_start_incoming_ratio, }
    #
    #         print("###",replay_para)
    #         return replay_para
    #     if(self.close_loop_CL.test_stats == None or self.close_loop_CL.train_stats == None ):
    #         return None
    #
    #     self.next_state = self.RL_env.get_state(self.close_loop_CL.train_stats,
    #                                       self.close_loop_CL.test_stats,
    #                                       self.close_loop_CL.weighted_test_stats,
    #                                       self.close_loop_CL.class_acc_stats,
    #                                       self.close_loop_CL.class_loss_stats,)
    #
    #     self.RL_agent.update( i,self.state,self.action,self.reward,self.next_state)
    #
    #     self.state = self.next_state
    #
    #
    #     self.action = self.RL_agent.sample_action(self.state) #self.sample_action(self.state) ## dormant RL
    #     selected_action= self._get_replay_para(self.action)
    #     return selected_action



