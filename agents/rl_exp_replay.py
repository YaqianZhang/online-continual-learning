
from torch.utils import data


from agents.exp_replay_with_feedback import ExperienceReplay_eval
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter

from RL.RL_trainer import RL_trainer
from RL.env.RL_env_MDP import RL_env_MDP
from RL.agent.RL_agent_MDP_DQN import RL_DQN_agent
from RL.agent.RL_pg_agent import RL_pg_agent



class RL_ExperienceReplay(ExperienceReplay_eval):
    def __init__(self, model, opt, params):
        super(RL_ExperienceReplay, self).__init__(model, opt, params)


        self.setup_RL_agent(model, params)
        self.start_RL = False
        self.start_task = 1



    def setup_RL_agent(self,model,params):
        if(params.RL_type == "RL_adpRatio"):
            self.adaptive_ratio = True
        if (params.RL_type != "NoRL" ):

            if (params.RL_type in [ "RL_ratio_1para","RL_adpRatio","RL_ratio","RL_memIter","RL_ratioMemIter","DormantRL","RL_2ratioMemIter"]):
                self.RL_agent = RL_DQN_agent(params)#RL_memIter_agent(params)
                self.RL_env = RL_env_MDP(params, model,self.RL_agent,self)

                self.RL_trainer = RL_trainer(params, self.RL_env, self.RL_agent, )
            elif (params.RL_type in [ "RL_actor"]):
                self.RL_agent = RL_pg_agent(params)#RL_memIter_agent(params)
                self.RL_env = RL_env_MDP(params, model,self.RL_agent,self)

                self.RL_trainer = RL_trainer(params, self.RL_env, self.RL_agent, )
            else:
                raise NotImplementedError("undefined RL_type")



    def check_start_RL(self,task_seen):

        if (task_seen == 0 or self.params.RL_type == "NoRL"):
            self.start_RL = False
        else:
            self.start_RL = self.memory_manager.test_memory_ready()


    def RL_joint_training(self, task_seen,i):

        self.check_start_RL(task_seen)
        if (self.params.dynamics_type == "same_batch"):
            self.replay_para = self.RL_env.get_basic_replay_para()

            self.stats = self.joint_training(self.replay_para,TEST=True)

            if (self.start_RL and self.stats != None):
                self.stats = self.RL_trainer.RL_training_step(self.stats,  task_seen)

        elif (self.params.dynamics_type == "next_batch"):
            if (self.start_RL and self.stats != None  and i>self.params.RL_start_batchstep and ("correct_cnt_mem_new" in self.stats.keys())):

                self.stats = self.RL_trainer.RL_training_step(self.stats,  task_seen)

            else:
                print("not RL batch num",i)
                if i<= self.params.RL_start_batchstep:

                    if(self.params.RL_type == "DormantRL"):
                        self.replay_para = {'mem_iter': self.params.mem_iters,
                                            'mem_ratio': self.params.mem_ratio,
                                            'incoming_ratio': self.params.incoming_ratio, }
                    else:

                        self.replay_para = {'mem_iter': self.params.mem_iters,
                                            'mem_ratio': self.params.task_start_mem_ratio,
                                            'incoming_ratio': self.params.task_start_incoming_ratio, }
                        if (self.params.save_prefix == "tsbug"):
                            self.RL_trainer.state = None
                            self.RL_trainer.action = 3
                else:


                    self.replay_para  = {'mem_iter': self.params.mem_iters,
                                   'mem_ratio': self.params.mem_ratio,
                                   'incoming_ratio': self.params.incoming_ratio,}
                #self.stats = self.replay_and_evaluate(self, replay_para)
                self.stats = self.joint_training(self.replay_para, TEST=True)
                print(self.stats)
        elif (self.params.dynamics_type == "within_batch"):

            if (self.start_RL and self.stats != None and i > 0 and ("correct_cnt_mem_new" in self.stats.keys())):

                self.replay_para['mem_iter'] = 1

                for mini_iter in range(self.params.mem_iters):
                    self.stats["mini_iter"]=mini_iter
                    self.stats = self.RL_trainer.RL_training_step(self.stats, task_seen,)
            else:
                self.replay_para = {'mem_iter': self.params.mem_iters,
                                    'mem_ratio': self.params.mem_ratio,
                                    'incoming_ratio': self.params.incoming_ratio, }

                self.stats = self.joint_training(self.replay_para, TEST=True)


        else:
            raise NotImplementedError("undefined dynamics type", self.params.dynamics_type)
        return self.stats

    def compute_incoming_influence(self):
        return self.memory_manager.compute_incoming_influence(self.incoming_batch['batch_x'],self.incoming_batch['batch_y'])


    def train_learner(self, x_train, y_train,labels=None):
        print("new task!!!!!!!!!!!!!!!")

        if(self.params.episode_type == "batch"):
            self.RL_agent.initialize_q()
        ## reset q function
        self.memory_manager.reset_new_old()
        if(self.params.critic_ER_type == "recent3" or self.params.critic_ER_type == "recent4"):
            self.RL_agent.ExperienceReplayObj.reset_RL_buffer()

        self.task_seen_so_far += 1


        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        self.losses_batch = AverageMeter()
        self.losses_mem = AverageMeter()
        self.acc_batch = AverageMeter()
        self.acc_mem = AverageMeter()
        self.batch_num = len(train_loader)



        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # fetch incoming batch data
                if(self.params.RL_type != "NoRL"):
                    self.RL_agent.batch_num = i
                    self.RL_agent.greedy = None
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x, batch_y = self.memory_manager.update_before_training(  batch_x, batch_y)




                self.incoming_batch={
                    "batch_x":batch_x,
                    "batch_y":batch_y,
                    "batch_num":i
                }
                if (self.task_seen_so_far == self.start_task ):  ## no test data
                    self.replay_para = {'mem_iter': self.params.mem_iters,
                                        'mem_ratio': 1,
                                        'incoming_ratio': 1, }

                    stats_dict = self.joint_training(self.replay_para)
                    self.log_stats_list(stats_dict)
                elif(self.params.RL_type == "NoRL"): ## no test data
                    self.replay_para = {'mem_iter': self.params.mem_iters,
                                   'mem_ratio': self.params.mem_ratio,
                                   'incoming_ratio': self.params.incoming_ratio, }

                    stats_dict = self.joint_training( self.replay_para)
                    self.log_stats_list(stats_dict)


                else:
                    stats_dict = self.RL_joint_training(self.task_seen,i)
                    self.log_stats_list(stats_dict)
                    self.log_test_stats_list(stats_dict)
                    self.mem_iter_list.append(self.RL_env.RL_mem_iters)
                    self.incoming_ratio_list.append(self.RL_env.RL_incoming_ratio)
                    self.mem_ratio_list.append(self.RL_env.RL_mem_ratio)
                self.memory_manager.update_memory(batch_x, batch_y)

                if (i) % 40 == 0 and self.verbose:
                    if(self.params.RL_type != "NoRL"):
                        print(self.task_seen, self.memory_manager.test_buffer.current_index, self.start_RL,
                               )
                        print(i,"reward ",self.RL_trainer.reward,"RL run steps:",self.RL_agent.RL_running_steps,"q update",self.RL_agent.RL_agent_update_steps,"eps:",self.RL_agent.epsilon)
                        print(self.replay_para,"action:",self.RL_agent.greedy,self.RL_trainer.action,)
                        # print(
                        #     #" MemIter:",self.RL_env.basic_mem_iters,"+",self.RL_env.RL_mem_iters,
                        #       # " iratio:",self.RL_env.RL_incoming_ratio,
                        #       # " mratio:",self.RL_env.RL_mem_ratio,
                        #       )
                    # if(stats_dict!= None):
                    #     #print(stats_dict['batch_num'])
                    #     print(
                    #         '==>>> it: {},  '
                    #         'running mem acc: {:.3f}'
                    #         'running train loss: {:.3f}, '
                    #             .format(i,stats_dict["loss_mem_value"],stats_dict["loss_incoming_value"],
                    #
                    #                     )
                    #     )
                    #     if ("correct_cnt_mem_new" in stats_dict):
                    #         print(
                    #             '==>>> it: {},  '
                    #                 'old mem acc: {:.3f}  num {:d} '
                    #                 'new mem acc: {:.3f} num {:d}  '
                    #                 .format(i,
                    #                         stats_dict["loss_mem_old"],
                    #                         stats_dict["old_task_num"],
                    #                         stats_dict["loss_mem_new"],
                    #                         stats_dict["new_task_num"],
                    #                         )
                    #         )


            if(self.params.use_tmp_buffer):
                self.memory_manager.tmp_buffer.update_true_buffer()

        self.after_train()











