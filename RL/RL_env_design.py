
import torch
import numpy as np

class RL_env(object):
    ## take a state and action, output state and reward
    ## implement state design and reward design
    def __init__(self,params,RL_replay=None):
        self.params = params
        self.RL_replay = RL_replay
        self.action_design_space = self.set_action_space(params)
    def get_state_dim(self):
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
            "train_test2": 2,


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


    def get_state(self,train_stats,test_stats,weighted_test_stats, class_acc_stats,class_loss_stats,
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
        elif (state_feature_type == "train_test2"):
            list_data = [
                         train_stats["loss_mem"],
                         stats_dict['test_loss']
                         ]

        else:
            raise NotImplementedError("undefined state type", state_feature_type)


        if(self.params.critic_type == "task_critic"):
            task_vec = self._one_hot(self.RL_replay.close_loop_CL.CL_agent.task_seen)
            list_data = list_data + task_vec

        state = np.array(list_data, dtype=np.float32).reshape([1, -1])
        state = torch.from_numpy(state)
        return state
    ### reward
    def get_reward(self,stats,stats_prev):
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
    def get_action_dim(self):
        return len(self.action_design_space)

    def set_action_space(self, params):  ## determined by RL_type and action_design type
        # lr_list_dict = {
        #     "basic": [0.01, 0.1, 0.2],
        #     "4lr": [0.001, 0.01, 0.1, 0.2],
        #     "5lr": [0.001, 0.01, 0.1, 0.2, 0.5],
        #     "scr": [0.001, 0.01, 0.1, 0.5],
        # }
        # self.para_list = lr_list_dict[self.params.online_hyper_lr_list_type]
        self.action_start = params.mem_iter_min
        if (self.params.hp_action_space == "ratio"):
            # self.action_design_space = np.linspace(0.1,2,10)
            self.action_design_space = np.linspace(0, 2, 10)
            # print("action space", self.action_design_space)

        elif (self.params.hp_action_space == "iter"):
            self.action_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)

        elif (self.params.hp_action_space == "ratio_iter"):
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


        elif (params.RL_type == "DormantRL" or params.RL_type == "NoRL"):
            self.action_design_space = []

        else:
            raise NotImplementedError("Undefined action space for RL type", params.RL_type)

        return self.action_design_space


    def _one_hot(self,task_id):
        self.task_vec = [0]*self.params.num_tasks
        self.task_vec[task_id]=1
        return self.task_vec





