#
#
import numpy as np
import torch
from RL.agent.RL_agent_MDP_DQN_hp import RL_DQN_agent_hp
from utils.utils import maybe_cuda
class RL_replay(object):

    def __init__(self,params,close_loop):
        self.params = params

        #self.RL_env = RL_env

        #RL_memIter_agent(params)

        #self.memoryManager = memoryManager

        self.close_loop_CL = close_loop





        self.state = None
        self.action = None
        self.reward = None
        self.total_reward = 0
        self.return_list = []



        ob_dim = self.init_state()



        self.action_design_space = self.init_action_space(self.params)
        self.action_num = len(self.action_design_space)
        self.RL_agent = RL_DQN_agent_hp(params,self.action_num,ob_dim)
        self.init_replay_para()

    def init_state(self):
        if(self.params.scr_memIter):
            if (self.params.scr_memIter_action_type == "8"):
                self.para_list=[1,2,3,4,5,6,7,8]
            elif (self.params.scr_memIter_action_type == "4"):
                self.para_list=[2,4,6,8]
            else:
                raise NotImplementedError("unseed scr_memIter_action_type")
            if(self.params.scr_memIter_state_type == "3dim"):
                ob_dim = 3
            elif(self.params.scr_memIter_state_type == "4dim"):
                ob_dim = 4
            elif(self.params.scr_memIter_state_type == "6dim"):
                ob_dim = 6
            elif(self.params.scr_memIter_state_type == "7dim"):
                ob_dim = 7
            else:
                raise NotImplementedError("unseed scr_memIter_state_type")

        ob_dim_dict = {

            ### memiter
            "train_test4": 4,


            "new_old5_overall": 6,
            "weighted_new_old8": 8,
            "new_old10":10,
            "new_old_train7":7

        }
        state_feature_type = self.params.state_feature_type
        if (state_feature_type in ob_dim_dict.keys()):
            if (self.params.dynamics_type == "within_batch"):
                ob_dim =  ob_dim_dict[state_feature_type] + 1
            else:

                ob_dim =  ob_dim_dict[state_feature_type]
        else:
            raise NotImplementedError("state type is not defined" + state_feature_type)
        if(self.params.critic_type =="task_critic"):
            ob_dim += self.params.num_tasks
        return ob_dim

    def init_replay_para(self):
        self.RL_mem_iters = self.params.mem_iters
        self.RL_mem_ratio = self.params.mem_ratio
        self.RL_incoming_ratio = self.params.incoming_ratio

    def init_action_space(self, params): ## determined by RL_type and action_design type
        # lr_list_dict = {
        #     "basic": [0.01, 0.1, 0.2],
        #     "4lr": [0.001, 0.01, 0.1, 0.2],
        #     "5lr": [0.001, 0.01, 0.1, 0.2, 0.5],
        #     "scr": [0.001, 0.01, 0.1, 0.5],
        # }
        # self.para_list = lr_list_dict[self.params.online_hyper_lr_list_type]
        self.action_start = params.mem_iter_min
        if (params.RL_type == "RL_ratio"):
            # self.action_design_space = np.linspace(0.1,2,10)
            self.action_design_space = np.linspace(0, 2, 10)
            # print("action space", self.action_design_space)

        elif (params.RL_type == "RL_memIter"):
            self.action_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)

        elif (params.RL_type == "RL_ratioMemIter"):
            self.mem_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
            self.ratio_design_space = [0.0, 0.1, 0.5, 1.0, 1.5]

            self.action_design_space = []
            for i in range(len(self.mem_design_space)):
                for j in range(len(self.ratio_design_space)):
                    self.action_design_space.append((self.mem_design_space[i], self.ratio_design_space[j]))
        elif (params.RL_type == "RL_actor"):
            self.agent_params = {}
            self.agent_params['ac_dim'] = 1

            self.action_design_space = []
        elif (params.RL_type == "RL_2ratioMemIter"):
            self.mem_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
            # self.mem_ratio_design_space = [0.0,0.1,0.5,1.0,1.5]
            # self.incoming_ratio_design_space = [0.0, 0.1, 0.5, 1.0, 1.5]
            if (params.action_space_type == "posneu"):
                self.action_design_space = [(3, 0.1, 0.1),
                                            (3, 0.1, 0.5),
                                            (3, 0.1, 1),
                                            (3, 0.5, 0.5),
                                            (3, 0.5, 1),
                                            (3, 1, 1)]
                return self.action_design_space
            else:
                if (params.action_space_type == "sparse"):
                    self.mem_ratio_design_space = [0.1, 0.5, 1.0, ]
                    self.incoming_ratio_design_space = [0.1, 0.5, 1.0, ]
                elif (params.action_space_type == "ionly"):
                    self.mem_ratio_design_space = [1.0, ]
                    self.incoming_ratio_design_space = [0.1, 0.5, 1.0, 1.5]
                    self.base_action = 2
                elif (params.action_space_type == "ionly_dense"):
                    self.mem_ratio_design_space = [1.0, ]
                    self.incoming_ratio_design_space = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.2, 1.5]
                    self.base_action = 5
                elif (params.action_space_type == "monly_dense"):
                    self.incoming_ratio_design_space = [1.0, ]
                    self.mem_ratio_design_space = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.2, 1.5]
                elif (params.action_space_type == "medium"):
                    self.mem_ratio_design_space = [0.01, 0.1, 0.5, 1.0, ]
                    self.incoming_ratio_design_space = [0.01, 0.1, 0.5, 1.0, ]
                elif (params.action_space_type == "upper"):
                    self.mem_ratio_design_space = [0.1, 0.5, 1.0, 1.5]
                    self.incoming_ratio_design_space = [0.1, 0.5, 1.0, 1.5]

                elif (params.action_space_type == "dense"):
                    self.mem_ratio_design_space = [0.1, 0.25, 0.5, 0.75, 1.0, ]
                    self.incoming_ratio_design_space = [0.1, 0.25, 0.5, 0.75, 1.0, ]

                self.action_design_space = []
                if (params.dynamics_type == "same_batch"):
                    self.action_design_space.append((0, 0, 0))
                for i in range(0, len(self.mem_design_space)):
                    for j in range(len(self.mem_ratio_design_space)):
                        for k in range(len(self.incoming_ratio_design_space)):
                            self.action_design_space.append(
                                (self.mem_design_space[i], self.incoming_ratio_design_space[k],
                                 self.mem_ratio_design_space[j]))
        elif (params.RL_type == "RL_adpRatio"):
            self.action_design_space = [0.01, 0.1, 0.5, 1.0, ]
        elif (params.RL_type == "RL_ratio_1para"):

            self.mem_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
            # self.mem_ratio_design_space = [0.0,0.1,0.5,1.0,1.5]
            # self.incoming_ratio_design_space = [0.0, 0.1, 0.5, 1.0, 1.5]
            self.mem_ratio_design_space = np.linspace(0.1, 0.9, 9)

            self.action_design_space = []
            # self.action_design_space.append((0,0,0))
            for i in range(0, len(self.mem_design_space)):
                for j in range(len(self.mem_ratio_design_space)):
                    # for k in range(len(self.incoming_ratio_design_space)):
                    self.action_design_space.append((self.mem_design_space[i], 1 - self.mem_ratio_design_space[j],
                                                     self.mem_ratio_design_space[j]))



        elif (params.RL_type == "DormantRL" or params.RL_type == "NoRL"):
            self.action_design_space = []

        else:
            raise NotImplementedError("Undefined action space for RL type", params.RL_type)

        return self.action_design_space


    ### reward
    def _get_reward(self,stats,stats_prev):
        if(stats == None or stats_prev == None):
            return None
        if(self.params.reward_type == "test_loss"):
            return -stats['test_loss']
        elif (self.params.reward_type == "test_loss_rlt"):
            if(stats_prev != None):
                return -stats['test_loss'] + stats_prev['test_loss']
            else:
                return -stats['test_loss']

        else:
            return stats['test_acc']

    def set_reward(self,mem_iter = 0):
        self.reward = self._get_reward(self.close_loop_CL.test_stats,self.close_loop_CL.test_stats_prev)

        if(mem_iter != None and self.reward != None):

            self.reward -= self.params.reward_rg*mem_iter
        self.close_loop_CL.test_stats_prev = self.close_loop_CL.test_stats
        if(self.RL_agent.greedy == "greedy"):
            self.RL_agent.real_q.append(self.reward)
            self.RL_agent.greedy_action.append(self.action)



    ## action
    def from_action_to_replay_para(self,action):
        return self.action_design_space[action]
    def get_replay_para(self, action):
        self.raugM = 1
        if (self.params.critic_type == "actor_critic"):
            if (self.params.RL_type == "RL_2ratioMemIter"):
                if (self.params.mem_iter_max != self.params.mem_iter_min):
                    if(self.params.randaug):

                        if(self.params.save_prefix == "r30test"):
                            aug = (action[0]-0.1)/1.4*30
                        else:
                            aug = int((action[0]-0.1)/1.4*30)
                        replay_para = {'mem_iter': action[1],
                                       'mem_ratio': self.RL_mem_ratio,
                                       'incoming_ratio': self.RL_mem_ratio,
                                       'randaug_M':aug}
                    else:

                        replay_para = {'mem_iter': action[1],
                                       'mem_ratio': self.RL_mem_ratio,
                                       'incoming_ratio': action[0], }
                    return replay_para
                else:
                    replay_para = {'mem_iter': self.RL_mem_iters,
                                   'mem_ratio': self.RL_mem_ratio,
                                   'incoming_ratio': action[0], }



                    return replay_para
            else:
                replay_para = {'mem_iter': action[0],
                               'mem_ratio': self.RL_mem_ratio,
                               'incoming_ratio': self.RL_incoming_ratio }
                return replay_para



        if (self.params.RL_type == "RL_memIter"):
            self.RL_mem_iters = self.RL_agent.from_action_to_replay_para(action)
        elif (self.params.RL_type == "RL_ratio"):
            self.RL_incoming_ratio = self.RL_agent.from_action_to_replay_para(action)
        elif (self.params.RL_type == "RL_ratioMemIter"):
            self.RL_mem_iters, self.RL_incoming_ratio = self.RL_agent.from_action_to_replay_para(action)
        elif (self.params.RL_type == "RL_actor"):
            self.RL_incoming_ratio = action
            self.RL_mem_ratio = 1
            self.RL_mem_iters = 1
        elif (self.params.RL_type == "RL_2ratioMemIter" or self.params.RL_type == "RL_ratio_1para"):


            if (self.params.randaug):
                self.RL_mem_iters, act, self.RL_mem_ratio = self.RL_agent.from_action_to_replay_para(
                    action)
                self.raugM = int((act - 0.1) / 1.4 * 30)
                self.RL_incoming_ratio=1

            else:
                self.RL_mem_iters, self.RL_incoming_ratio, self.RL_mem_ratio = self.RL_agent.from_action_to_replay_para(
                    action)



        elif (self.params.RL_type == "DormantRL"):
            pass
        else:
            raise NotImplementedError("Undefined RL type in set replay action", self.params.RL_type)

        replay_para = {'mem_iter': self.RL_mem_iters,
                       'mem_ratio': self.RL_mem_ratio,
                       'incoming_ratio': self.RL_incoming_ratio,
                       'randaug_M':self.raugM}

        return replay_para




    def sample_action(self,state):
        # if(self.params.critic_type == "actor_critic"):
        #     return self.RL_agent.sample_continuous_action(state)
        # else:
        return self.RL_agent.sample_action(state)





    def make_replay_decision(self,i): ## action
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
        if(self.close_loop_CL.test_stats == None):
            return None

        self.next_state = self._get_state(self.close_loop_CL.train_stats,
                                          self.close_loop_CL.test_stats,
                                          self.close_loop_CL.weighted_test_stats,
                                          self.close_loop_CL.class_acc_stats,
                                          self.close_loop_CL.class_loss_stats,)

        self.update_RL_agent(i)

        self.state = self.next_state


        self.action = self.sample_action(self.state) ## dormant RL
        selected_action= self.get_replay_para(self.action)
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



    def _get_state(self,train_stats,test_stats,weighted_test_stats, class_acc_stats,class_loss_stats,
                   state_feature_type=None):
        stats_dict=test_stats

        if (state_feature_type == None):
            state_feature_type = self.params.state_feature_type

        if (state_feature_type == "weighted_new_old8"):
            list_data = [
                weighted_test_stats["weighted_mem_acc"],
                weighted_test_stats[ "weighted_incoming_acc"],
            weighted_test_stats["weighted_mem_loss"],
            weighted_test_stats["weighted_incoming_loss"],
            weighted_test_stats["mem_num_unseen"],
            weighted_test_stats["incoming_num_unseen"],
                          stats_dict['test_acc'],
                         stats_dict['test_loss']
                         ]
        elif(state_feature_type == "new_old10"):
            list_data = [
                weighted_test_stats["weighted_mem_acc"],
                weighted_test_stats[ "weighted_incoming_acc"],
            weighted_test_stats["weighted_mem_loss"],
            weighted_test_stats["weighted_incoming_loss"],
            weighted_test_stats["mem_num_unseen"],
            weighted_test_stats["incoming_num_unseen"],
                          stats_dict['test_acc'],
                         stats_dict['test_loss'],
                train_stats["acc_incoming"],
                train_stats["loss_incoming"],
                         ]

        elif(state_feature_type == "new_old_train7"):
            list_data = [

                weighted_test_stats[ "weighted_incoming_acc"],
            weighted_test_stats["weighted_incoming_loss"],
            weighted_test_stats["incoming_num_unseen"],
                          stats_dict['test_acc'],
                         stats_dict['test_loss'],
                train_stats["acc_incoming"],
                train_stats["loss_incoming"],
                         ]




        elif (state_feature_type == "new_old5_overall" or state_feature_type == "new_old5_4time"):
            list_data = [stats_dict["test_loss_old"],
                         stats_dict["test_acc_old"],
                         stats_dict["test_loss_new"],
                         stats_dict["test_acc_new"],
                          stats_dict['test_acc'],
                         stats_dict['test_loss']
                         ]
        elif (state_feature_type == "train_test4"):
            list_data = [
                         train_stats["acc_incoming"],
                         train_stats["loss_incoming"],
                          stats_dict['test_acc'],
                         stats_dict['test_loss']
                         ]

        else:
            raise NotImplementedError("undefined state type", state_feature_type)


        if(self.params.critic_type == "task_critic"):
            task_vec = self.one_hot(self.close_loop_CL.CL_agent.task_seen)
            list_data = list_data + task_vec

        state = np.array(list_data, dtype=np.float32).reshape([1, -1])
        state = torch.from_numpy(state)
        return state
    def one_hot(self,task_id):
        self.task_vec = [0]*self.params.num_tasks
        self.task_vec[task_id]=1
        return self.task_vec

############ update
    def update_RL_agent(self,i):
        if(self.state == None):
            return



        if ((i+1 ) % self.params.done_freq == 0):
            done = 1
        else:
            done = 0

        if(self.RL_agent.out_range == True):
            self.reward = -1


        self.RL_agent.real_reward_list.append(self.reward)
        self.RL_agent.RL_running_steps += 1
        self.RL_agent.ExperienceReplayObj.store(self.state, self.action, self.reward, self.next_state, done)
        self.RL_agent.update_agent(self.reward, self.state,
                                   self.action, self.next_state, done)  ## update RL agent



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














