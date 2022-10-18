from RL.RL_buffer import RL_ExperienceReplay



class RL_memIter_agent(object):
    def __init__(self, params):
        #super().__init__(params)
        self.params = params

        self.epsilon = 0.1
        self.alpha = 0.9 # Q update parameters
        self.selected_num = 1

        self.greedy = "None"


        # self.action_design_space  = self.initialize_actor(params)
        # self.action_num = len(self.action_design_space)  # params.mem_iter_max - params.mem_iter_min +1 ## memITer 0,1,2


        #self.ob_dim = self.initialize_state(params.state_feature_type)


        self.mse_training_loss =[]
        self.real_reward_list=[]
        self.real_action_list = []

        #self.critic=critic_class(params,self.action_num,self.ob_dim)

        #training
        training_steps_dict = {"cifar100":5000,
                          "core50":10000,
                          "cifar10":5000,
                               "clrs25":1500,
                               "mini_imagenet":5000}




        self.total_training_steps = params.num_runs * training_steps_dict[params.data]  # params['train_steps_total'] #None #
        if(self.params.save_prefix == "non_stationary"):
            self.total_training_steps = int(self.total_training_steps)/params.num_tasks
            print("!!!total training steps",self.total_training_steps)
        if(self.params.RL_type == "RL_MDP_stop"):
            self.total_training_steps = self.total_training_steps * 3
        self.RL_running_steps = 0
        self.RL_agent_update_steps=0
        self.RL_training_iters = params.critic_training_iters

        self.ExperienceReplayObj = RL_ExperienceReplay(params)
        self.ER_batchsize = params.ER_batch_size

    # def initialize_actor(self, params):
    #     self.action_start = params.mem_iter_min
    #     if (params.RL_type == "RL_ratio"):
    #         # self.action_design_space = np.linspace(0.1,2,10)
    #         self.action_design_space = np.linspace(0, 2, 10)
    #         #print("action space", self.action_design_space)
    #
    #     elif (params.RL_type == "RL_memIter"):
    #         self.action_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
    #
    #     elif (params.RL_type == "RL_ratioMemIter"):
    #         self.mem_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
    #         self.ratio_design_space = [0.0, 0.1, 0.5, 1.0, 1.5]
    #
    #         self.action_design_space = []
    #         for i in range(len(self.mem_design_space)):
    #             for j in range(len(self.ratio_design_space)):
    #                 self.action_design_space.append((self.mem_design_space[i], self.ratio_design_space[j]))
    #     elif(  params.RL_type == "RL_actor"):
    #         self.agent_params = {}
    #         self.agent_params['ac_dim']=1
    #
    #         self.action_design_space = []
    #     elif (params.RL_type == "RL_2ratioMemIter"  ):
    #         self.mem_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
    #         # self.mem_ratio_design_space = [0.0,0.1,0.5,1.0,1.5]
    #         # self.incoming_ratio_design_space = [0.0, 0.1, 0.5, 1.0, 1.5]
    #         if(params.action_space_type == "posneu"):
    #             self.action_design_space=[(3,0.1,0.1),
    #                                       (3,0.1,0.5),
    #                                       (3,0.1,1),
    #                                       (3,0.5,0.5),
    #                                       (3,0.5,1),
    #                                       (3,1,1)]
    #             return self.action_design_space
    #         else:
    #             if (params.action_space_type == "sparse"):
    #                 self.mem_ratio_design_space = [0.1, 0.5, 1.0, ]
    #                 self.incoming_ratio_design_space = [0.1, 0.5, 1.0, ]
    #             elif (params.action_space_type == "ionly"):
    #                 self.mem_ratio_design_space = [ 1.0, ]
    #                 self.incoming_ratio_design_space = [0.1, 0.5, 1.0,1.5 ]
    #                 self.base_action = 2
    #             elif (params.action_space_type == "ionly_dense"):
    #                 self.mem_ratio_design_space = [ 1.0, ]
    #                 self.incoming_ratio_design_space = [0.1,0.2,0.3, 0.5,0.75, 1.0,1.2,1.5 ]
    #                 self.base_action = 5
    #             elif (params.action_space_type == "monly_dense"):
    #                 self.incoming_ratio_design_space = [ 1.0, ]
    #                 self.mem_ratio_design_space = [0.1,0.2,0.3, 0.5,0.75, 1.0,1.2,1.5 ]
    #             elif (params.action_space_type == "medium"):
    #                 self.mem_ratio_design_space = [0.01,0.1, 0.5, 1.0, ]
    #                 self.incoming_ratio_design_space = [0.01,0.1, 0.5, 1.0, ]
    #             elif (params.action_space_type == "upper"):
    #                 self.mem_ratio_design_space = [0.1, 0.5, 1.0, 1.5]
    #                 self.incoming_ratio_design_space = [0.1, 0.5, 1.0, 1.5]
    #
    #             elif (params.action_space_type == "dense"):
    #                 self.mem_ratio_design_space = [0.1, 0.25, 0.5, 0.75, 1.0, ]
    #                 self.incoming_ratio_design_space = [0.1, 0.25, 0.5, 0.75, 1.0, ]
    #
    #
    #             self.action_design_space = []
    #             if (params.dynamics_type == "same_batch"):
    #                 self.action_design_space.append((0, 0, 0))
    #             for i in range(0, len(self.mem_design_space)):
    #                 for j in range(len(self.mem_ratio_design_space)):
    #                     for k in range(len(self.incoming_ratio_design_space)):
    #                         self.action_design_space.append((self.mem_design_space[i], self.incoming_ratio_design_space[k],
    #                                                          self.mem_ratio_design_space[j]))
    #     elif (params.RL_type == "RL_adpRatio"):
    #         self.action_design_space = [0.01, 0.1, 0.5, 1.0, ]
    #     elif (params.RL_type == "RL_ratio_1para"):
    #
    #         self.mem_design_space = np.arange(params.mem_iter_min, params.mem_iter_max + 1)
    #         # self.mem_ratio_design_space = [0.0,0.1,0.5,1.0,1.5]
    #         # self.incoming_ratio_design_space = [0.0, 0.1, 0.5, 1.0, 1.5]
    #         self.mem_ratio_design_space = np.linspace(0.1, 0.9, 9)
    #
    #         self.action_design_space = []
    #         # self.action_design_space.append((0,0,0))
    #         for i in range(0, len(self.mem_design_space)):
    #             for j in range(len(self.mem_ratio_design_space)):
    #                 # for k in range(len(self.incoming_ratio_design_space)):
    #                 self.action_design_space.append((self.mem_design_space[i], 1 - self.mem_ratio_design_space[j],
    #                                                  self.mem_ratio_design_space[j]))
    #
    #
    #
    #     elif (params.RL_type == "DormantRL"):
    #         self.action_design_space = []
    #
    #     else:
    #         raise NotImplementedError("Undefined action space for RL type", params.RL_type)
    #
    #     return self.action_design_space

    # def initialize_state(self,state_feature_type):
    #     ######### state ###############
    #     ob_dim_dict={
    #
    #         ### memiter
    #         "train_test4": 4,
    #
    #
    #         "4_dim":4,
    #         "4_loss":4,
    #         "3_dim":3,
    #         "3_loss":3,
    #         "6_dim":6,
    #         "7_dim":7,
    #         "8_dim":8,
    #         "task_dim":8,
    #         "same_batch":6,
    #         "sam_batch_7_dim":7,
    #         "new_old2":2,
    #
    #         "new_old3": 3,
    #         "new_old4": 4,
    #         "new_old_old": 3,
    #         "new_old_old4": 4,
    #         "new_old_old4_noi": 3,
    #         "new_old5": 5,
    #         "weighted_new_old8": 8,
    #         "new_old10":10,
    #         "new_old_train7": 7,
    #         "input_new_old5":5+160,
    #         "new_old5_overall":5,
    #         "new_old6_overall_train":6,
    #         "new_old7_overall_train_income":7,
    #         "new_old7_overall_train": 7,
    #         "new_old5_scale": 5,
    #         "new_old5_4time":20,
    #         "new_old5_incoming":6,
    #         "new_old5_task": 5+self.params.num_tasks,
    #         "new_old5t": 5,
    #         "new_old6": 6,
    #         "new_old6m": 6,
    #         "new_old6mn": 6,
    #         "new_old6mn_org":6,
    #         "new_old6mnt": 7,
    #         "new_old6mn_incoming": 7,
    #         "new_old7": 7,
    #         "new_old9": 9,
    #         "new_old11": 1,
    #     }
    #     if(state_feature_type in ob_dim_dict.keys()):
    #         if(self.params.dynamics_type == "within_batch"):
    #             return ob_dim_dict[state_feature_type]+1
    #         else:
    #
    #             return ob_dim_dict[state_feature_type]
    #     else:
    #         raise NotImplementedError("state type is not defined" + state_feature_type)






