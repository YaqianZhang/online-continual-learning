#
#
import numpy as np
import torch
from RL.agent.RL_agent_MDP_DQN_hp import RL_DQN_agent_hp
from RL.agent.RL_agent_MAB import RL_MAB_agent
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
        ob_dim = self.RL_env.get_state_dim()
        action_num = self.RL_env.get_action_dim()
        if(self.params.RL_type == "RL_MDP"):
            self.RL_agent = RL_DQN_agent_hp(params, action_num, ob_dim, RL_replay=self)
        elif(self.params.RL_type == "RL_MAB"):
            self.RL_agent = RL_MAB_agent(params,action_num,ob_dim, RL_replay=self)

        self.state = None
        self.action = None
        self.reward = None
        self.total_reward = 0
        self.return_list = []
        self._init_replay_para()



    def _init_replay_para(self):
        self.RL_mem_iters = self.params.mem_iters
        self.RL_mem_ratio = self.params.mem_ratio
        self.RL_incoming_ratio = self.params.incoming_ratio





    def set_reward(self,mem_iter = 0):
        self.reward = self.RL_env.get_reward(self.close_loop_CL.test_stats,self.close_loop_CL.test_stats_prev)

        if(mem_iter != None and self.reward != None):

            self.reward -= self.params.reward_rg*mem_iter
        self.close_loop_CL.test_stats_prev = self.close_loop_CL.test_stats
        if(self.RL_agent.greedy == "greedy"):
            self.RL_agent.real_q.append(self.reward)
            self.RL_agent.greedy_action.append(self.action)



    ## action
    def _from_action_to_replay_para(self,action):
        return self.RL_env.action_design_space[action]
    def _get_replay_para(self, action):
        self.raugM = 1
        if (self.params.critic_type == "actor_critic"):
            if (self.params.hp_action_space == "aug_iter"):


                aug = (action[0]-0.1)/1.4*30
                replay_para = {'mem_iter': action[1],
                               'mem_ratio': self.RL_mem_ratio,
                               'incoming_ratio': self.RL_mem_ratio,
                               'randaug_M':aug}
            elif (self.params.hp_action_space == "ratio_iter"):
                replay_para = {'mem_iter': action[1],
                               'mem_ratio': self.RL_mem_ratio,
                               'incoming_ratio': action[0], }

            elif (self.params.hp_action_space == "iter"):
                replay_para = {'mem_iter': action[0],
                               'mem_ratio': self.RL_mem_ratio,
                               'incoming_ratio': self.RL_incoming_ratio }
            elif(self.params.hp_action_space == "ratio"):
                replay_para = {'mem_iter': self.RL_mem_iters,
                               'mem_ratio': self.RL_mem_ratio,
                               'incoming_ratio': action[0], }
            return replay_para
        else:

            if (self.params.hp_action_space == "iter"):
                self.RL_mem_iters = self._from_action_to_replay_para(action)
            elif (self.params.hp_action_space == "ratio"):
                self.RL_incoming_ratio = self._from_action_to_replay_para(action)
            elif (self.params.hp_action_space == "ratio_iter"):
                self.RL_mem_iters, self.RL_incoming_ratio, self.RL_mem_ratio = self._from_action_to_replay_para(
                action)

            elif (self.params.hp_action_space == "aug_iter"):

                self.RL_mem_iters, act, self.RL_mem_ratio = self._from_action_to_replay_para(
                    action)
                self.raugM = int((act - 0.1) / 1.4 * 30)
                self.RL_incoming_ratio=1


            elif (self.params.RL_type == "DormantRL"):
                pass
            else:
                raise NotImplementedError("Undefined RL type in set replay action", self.params.RL_type)

            replay_para = {'mem_iter': self.RL_mem_iters,
                           'mem_ratio': self.RL_mem_ratio,
                           'incoming_ratio': self.RL_incoming_ratio,
                           'randaug_M':self.raugM}

            return replay_para




    # def sample_action(self,state):
    #     # if(self.params.critic_type == "actor_critic"):
    #     #     return self.RL_agent.sample_continuous_action(state)
    #     # else:
    #     return


    def make_stop_decision(self,train_stats):
        ## compute state from train_stats
        # train_stats = {'acc_incoming': acc_incoming,
        #                'acc_mem': acc_mem,
        #                "loss_incoming": incoming_loss.item(),
        #                "loss_mem": mem_loss.item(),
        #                "batch_num": i,
        state = [train_stats['acc_incoming'],
                 train_stats['acc_mem'],
                 train_stats['loss_incoming'],
                 train_stats['loss_mem'],
                 ]

        self.action = self.RL_agent.sample_action(self.state) #self.sample_action(self.state) ## dormant RL
        #output the sampled action: stop, not stop

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

        self.next_state = self.RL_env.get_state(self.close_loop_CL.train_stats,
                                          self.close_loop_CL.test_stats,
                                          self.close_loop_CL.weighted_test_stats,
                                          self.close_loop_CL.class_acc_stats,
                                          self.close_loop_CL.class_loss_stats,)

        self.RL_agent.update( i,self.state,self.action,self.reward,self.next_state)

        self.state = self.next_state


        self.action = self.RL_agent.sample_action(self.state) #self.sample_action(self.state) ## dormant RL
        selected_action= self._get_replay_para(self.action)
        return selected_action


    ############# state


    ## old state space for SCR mem iter
    # def get_state(self,train_stats,test_stats):
    #
    #     if(self.params.scr_memIter):
    #         if(self.params.scr_memIter_state_type == "3dim"):
    #             list_data = [test_stats['test_loss'],test_stats['test_acc'], train_stats['loss_mem']]
    #         elif(self.params.scr_memIter_state_type == "4dim"):
    #             list_data = [train_stats['batch_num'],test_stats['test_loss'], test_stats['test_acc'], train_stats['loss_mem']]
    #         elif (self.params.scr_memIter_state_type == "6dim"):
    #             list_data = [train_stats['batch_num'], test_stats['test_loss'], test_stats['test_acc'],
    #                      train_stats['loss_mem'],train_stats['incoming_loss'],train_stats['acc_incoming']]
    #         elif (self.params.scr_memIter_state_type == "7dim"):
    #             list_data = [train_stats['batch_num'], test_stats['test_loss'], test_stats['test_acc'],
    #                          train_stats['loss_mem'], train_stats['softmax_loss'],train_stats['incoming_loss'], train_stats['acc_incoming']]
    #
    #         else:
    #             raise NotImplementedError("unseed scr_memIter_state_type")
    #     else:
    #         list_data = [train_stats['acc_mem'], train_stats['loss_mem']]
    #     state = np.array(list_data, dtype=np.float32).reshape([1, -1])
    #     state = torch.from_numpy(state)
    #     return state

    # def _get_state_scr(self, train_stats, test_stats):
    #     list_data = [train_stats['loss_mem'], train_stats['batch_num']]
    #     state = np.array(list_data, dtype=np.float32).reshape([1, -1])
    #     state = torch.from_numpy(state)
    #     return state





#

    # def virtual_explore(self,state):
    #
    #     #### virtual explore
    #     for i in range(self.params.virtual_update_times):
    #         action = self.RL_agent.take_random_action()  ## dormant RL
    #
    #         if (action != None):
    #             end_stats = self.RL_env.step(action,virtual=True)  ## perform replay
    #         else:
    #             end_stats = stats_dict
    #
    #         reward = self.RL_env.get_reward(end_stats, stats_dict, )
    #         self.RL_agent.ExperienceReplayObj.store(state, action, reward, state, done)














