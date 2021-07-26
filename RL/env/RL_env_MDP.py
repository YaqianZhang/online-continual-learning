import torch
import numpy as np

from RL.env.RL_env_base import Base_RL_env
class RL_env_MDP(Base_RL_env):
    def __init__(self,params,model,testBuffer,RL_agent,CL_agent):
        super().__init__(params,model,testBuffer,RL_agent,CL_agent)
        self.return_list = []

        if(self.params.RL_type == "DormantRL"):
            self.basic_mem_iters = self.params.mem_iters
            self.basic_i_ratio = self.params.incoming_ratio
            self.basic_m_ratio = self.params.mem_ratio
        else:
            self.basic_mem_iters  = 1
            self.basic_i_ratio = 1
            self.basic_m_ratio = 1
        self.initialize()

    def initialize(self):
        self.state = None
        self.action = None
        self.add_mem_iters =0
        self.add_incoming_ratio = 0
        self.add_mem_ratio = 0
        self.reward = None
        self.start_RL = False
        self.stats = None

        self.total_reward = 0


    def get_state(self, stats_dict, i, state_feature_type=None,task_seen=None):


        [correct_cnt_incoming, correct_cnt_mem,
         loss_incoming_value, loss_mem_value, correct_cnt_test_mem,loss_test_value, ] = [
            stats_dict['correct_cnt_incoming'],stats_dict['correct_cnt_mem'],
        stats_dict['loss_incoming_value'],stats_dict['loss_mem_value'],
        stats_dict['correct_cnt_test_mem'],stats_dict['loss_test_value']]

        if (state_feature_type == None):
            state_feature_type = self.params.state_feature_type
        if (state_feature_type == "4_dim"):
            list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, i]
        elif (state_feature_type == "3_dim"):
            list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem]
        elif (state_feature_type == "4_loss"):
            list_data = [loss_incoming_value, loss_mem_value, loss_test_value, i]
        elif (state_feature_type == "3_loss"):
            list_data = [loss_incoming_value, loss_mem_value, loss_test_value]
        elif (state_feature_type == "6_dim"):
            list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
                         loss_incoming_value, loss_mem_value, loss_test_value]
        elif (state_feature_type == "7_dim"):
            list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
                         loss_incoming_value, loss_mem_value, loss_test_value, i]
        elif (state_feature_type == "8_dim"):
            list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
                         loss_incoming_value, loss_mem_value, loss_test_value, i,task_seen]

        elif(state_feature_type == "task_dim"):
            # if("loss_mem_old" not in stats_dict):
            #     return None

            list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
                         loss_incoming_value, loss_mem_value, loss_test_value,
                         stats_dict["loss_mem_old"],stats_dict["loss_mem_new"]]


        else:
            raise NotImplementedError("undefined state type")
        state = np.array(list_data, dtype=np.float32).reshape([1, -1])
        state = torch.from_numpy(state)
        return state


    def set_replay_action(self,state):




        if(self.params.RL_type == "RL_memIter"):
            self.action = self.RL_agent.sample_action(state)
            self.add_mem_iters  = self.RL_agent.from_action_to_replay_para(self.action)
        elif(self.params.RL_type == "RL_ratio"):
            self.action = self.RL_agent.sample_action(state)
            self.add_incoming_ratio = self.RL_agent.from_action_to_replay_para(self.action)
        elif(self.params.RL_type == "RL_ratioMemIter"):
            self.action = self.RL_agent.sample_action(state)
            self.add_mem_iters,self.add_incoming_ratio = self.RL_agent.from_action_to_replay_para(self.action)
        elif(self.params.RL_type == "RL_2ratioMemIter"):
            self.action = self.RL_agent.sample_action(state)
            self.add_mem_iters,self.add_incoming_ratio,self.add_mem_ratio = self.RL_agent.from_action_to_replay_para(self.action)
        elif (self.params.RL_type == "DormantRL"):
            pass
        else:
            raise NotImplementedError ("Undefined RL type in set replay action",self.params.RL_type)

        return self.action, self.add_mem_iters, self.add_incoming_ratio,self.add_mem_ratio

    def check_start_RL(self,task_seen):


        if (task_seen == 0 or self.params.RL_type == "NoRL"):
            self.start_RL = False
        else:
            self.start_RL = self.test_buffer.current_index >= 50#self.params.test_mem_batchSize




    def RL_training_step(self,stats_dict, i, state, reward, action,
                         batch_x, batch_y,losses_batch, acc_batch, losses_mem, acc_mem,er_agent=None,task_seen=None):
        next_state = self.get_state(stats_dict, i, task_seen=task_seen)
        if (state != None and reward != None and action != None):
            if((i+1) %self.params.done_freq==0):
                done = 1
                #print(i,"!!! episode finish",self.total_reward,self.episode_start_test_loss)
                print("___________________________________")
                self.return_list.append(self.total_reward)
                self.total_reward = 0
            else:
                done = 0
                self.total_reward += reward




            self.RL_agent.update_agent(reward, state,
                                       action, next_state, done)  ## todo next state and done
        state = next_state


        action, action_mem_iter, action_incoming_ratio,action_mem_ratio = self.set_replay_action(state)
        end_stats=stats_dict
        if (action != None and action_mem_iter > 0):
            next_stats_dict = er_agent.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,
                                             iters=action_mem_iter, mem_ratio=action_mem_ratio,incoming_ratio=action_incoming_ratio)
            next_stats_dict = er_agent.compute_test_accuracy(next_stats_dict)
            end_stats = next_stats_dict



        reward = self.get_reward(end_stats,   stats_dict,i,action, action_mem_iter, action_incoming_ratio,action_mem_ratio)


        return state,action,reward,end_stats



    def RL_joint_training(self,i,batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,task_seen,
                         er_agent=None):

        self.check_start_RL(task_seen)
        if (self.params.dynamics_type == "same_batch"):
            batch_x, batch_y = self.update_memory_before(batch_x, batch_y)



            stats_dict= er_agent.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,
                                              iters=self.basic_mem_iters ,incoming_ratio=self.basic_i_ratio,
                                                mem_ratio=self.basic_m_ratio)
            stats_dict = er_agent.compute_test_accuracy(stats_dict)
            self.stats = stats_dict
            # if(self.stats != None):
            #     if(self.params.state_feature_type == "task_dim" and ("loss_mem_old" not in stats_dict)):
            #         self.start_RL =  False ## unable to compute memory_old memory_new

            if (self.start_RL and self.stats != None ):

                self.state, self.action, self.reward, self.stats = self.RL_training_step(self.stats, i, self.state, self.reward, self.action, batch_x, batch_y,
                                                                      losses_batch, acc_batch, losses_mem, acc_mem,er_agent,task_seen)

        elif (self.params.dynamics_type == "next_batch"):
            if (self.start_RL and self.stats != None):
                self.state, self.action, self.reward, self.stats = self.RL_training_step(self.stats, i, self.state, self.reward, self.action, batch_x, batch_y,
                                                                      losses_batch, acc_batch, losses_mem, acc_mem,er_agent,task_seen)
            else:
                stats_dict = er_agent.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,
                                                     iters=self.params.mem_iters,
                                                     incoming_ratio=self.params.incoming_ratio,
                                                     mem_ratio=self.params.mem_ratio)
                stats_dict = er_agent.compute_test_accuracy(stats_dict)
                self.stats =stats_dict
        else:
            raise NotImplementedError("undefined dynamics type",self.params.dynamics_type)
        return  self.stats




    def update_memory_before(self,batch_x,batch_y):
        test_size = int(batch_x.shape[0] * 0.2)
        # print("save batch to test buffer and buffer",test_size)
        self.test_buffer.update(batch_x[:test_size], batch_y[:test_size])
        return batch_x[test_size:], batch_y[test_size:] #todo save the sample that not used in test memory into training memory





