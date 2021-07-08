import torch
import numpy as np

from RL.env.RL_env_base import Base_RL_env
class RL_env_MDP(Base_RL_env):
    def __init__(self,params,model,testBuffer,RL_agent,CL_agent):
        super().__init__(params,model,testBuffer,RL_agent,CL_agent)
        self.state = None
        self.action = None
        self.mem_iters =params.mem_iters
        self.incoming_ratio = params.incoming_ratio
        self.reward = None
        self.start_RL = False
        self.stats = None

    def get_state(self, stats_dict, i, state_feature_type=None):

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

        else:
            raise NotImplementedError("undefined state type")
        state = np.array(list_data, dtype=np.float32).reshape([1, -1])
        state = torch.from_numpy(state)
        return state


    def set_replay_action(self,state):


        if(self.params.RL_type == "RL_memIter"):
            self.action = self.RL_agent.sample_action(state)
            self.mem_iters  = self.RL_agent.from_action_to_memIter(self.action)
        elif(self.params.RL_type == "RL_ratio"):
            self.action = self.RL_agent.sample_action(state)
            self.incoming_ratio = self.RL_agent.from_action_to_ratio(self.action)
        elif(self.params.RL_type == "RL_ratioMemIter"):
            self.action = self.RL_agent.sample_action(state)
            self.mem_iters,self.incoming_ratio = self.RL_agent.from_action_to_ratio_memIter(self.action)
        elif (self.params.RL_type == "DormantRL"):
            pass
        else:
            raise NotImplementedError ("Undefined RL type in set replay action",self.params.RL_type)

        return self.action, self.mem_iters, self.incoming_ratio

    def check_start_RL(self,task_seen):

        if (task_seen == 0 or self.params.RL_type == "NoRL"):
            self.start_RL = False
        else:
            self.start_RL = self.test_buffer.current_index >= self.params.test_mem_batchSize




    def RL_training_step(self,stats_dict, i, state, reward, action,
                         batch_x, batch_y,losses_batch, acc_batch, losses_mem, acc_mem,er_agent=None):
        next_state = self.get_state(stats_dict, i, )
        if (state != None and reward != None and action != None):
            done = (i % 50 == 0)
            self.RL_agent.update_agent(reward, state,
                                       action, next_state, done)  ## todo next state and done
        state = next_state


        action, action_mem_iter, action_ratio = self.set_replay_action(state)
        end_stats=stats_dict
        if (action != None and action_mem_iter > 0):
            next_stats_dict = er_agent.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,
                                             iters=action_mem_iter, ratio=action_ratio)
            next_stats_dict = er_agent.compute_test_accuracy(next_stats_dict)
            end_stats = next_stats_dict


        reward = self.get_reward(end_stats,   stats_dict)

        return state,action,reward,end_stats



    def RL_joint_training(self,i,batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,task_seen,
                         er_agent=None):
        self.check_start_RL(task_seen)
        if (self.params.test_mem_type == "before"):
            batch_x, batch_y = self.update_memory_before(batch_x, batch_y)
        if (self.params.dynamics_type == "same_batch"):
            stats_dict= er_agent.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,
                                              iters=self.params.mem_iters, ratio=self.incoming_ratio)
            stats_dict = er_agent.compute_test_accuracy(stats_dict)
            self.stats = stats_dict



            if (self.start_RL and self.stats != None):
                self.state, self.action, self.reward, self.stats = self.RL_training_step(self.stats, i, self.state, self.reward, self.action, batch_x, batch_y,
                                                                      losses_batch, acc_batch, losses_mem, acc_mem,er_agent)

        elif (self.params.dynamics_type == "next_batch"):
            if (self.start_RL and self.stats != None):
                self.state, self.action, self.reward, self.stats = self.RL_training_step(self.stats, i, self.state, self.reward, self.action, batch_x, batch_y,
                                                                      losses_batch, acc_batch, losses_mem, acc_mem,er_agent)
            else:
                stats_dict = er_agent.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,
                                                     iters=self.params.mem_iters, ratio=self.incoming_ratio)
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





