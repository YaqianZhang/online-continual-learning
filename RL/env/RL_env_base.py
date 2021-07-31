import torch
import numpy as np
from utils.buffer.buffer_utils import random_retrieve
from utils.utils import maybe_cuda
import torch.nn.functional as F
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector


class Base_RL_env(object):
    def __init__(self, params,model,RL_agent,CL_agent):
        super().__init__()
        self.model = model
        #self.memoryManager = memoryManager
        self.RL_agent = RL_agent
        self.CL_agent = CL_agent

        self.params = params
        self.reward_type = params.reward_type

        self.reward_list = []
        self.task_reward_list=[]

    def get_state(self, stats_dict, state_feature_type=None,task_seen=None):

        i = stats_dict['batch_num']
        # [correct_cnt_incoming, correct_cnt_mem,
        #  loss_incoming_value, loss_mem_value, correct_cnt_test_mem,loss_test_value, ] = [
        #     stats_dict['correct_cnt_incoming'],stats_dict['correct_cnt_mem'],
        # stats_dict['loss_incoming_value'],stats_dict['loss_mem_value'],
        # stats_dict['correct_cnt_test_mem'],stats_dict['loss_test_value']]
        #
        if (state_feature_type == None):
            state_feature_type = self.params.state_feature_type
        # if (state_feature_type == "4_dim"):
        #     list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, i]
        # elif (state_feature_type == "3_dim"):
        #     list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem]
        # elif (state_feature_type == "4_loss"):
        #     list_data = [loss_incoming_value, loss_mem_value, loss_test_value, i]
        # elif (state_feature_type == "3_loss"):
        #     list_data = [loss_incoming_value, loss_mem_value, loss_test_value]
        # elif (state_feature_type == "6_dim"):
        #     list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
        #                  loss_incoming_value, loss_mem_value, loss_test_value]
        # elif (state_feature_type == "7_dim"):
        #     list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
        #                  loss_incoming_value, loss_mem_value, loss_test_value, i]
        # elif (state_feature_type == "8_dim"):
        #     list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
        #                  loss_incoming_value, loss_mem_value, loss_test_value, i,task_seen]

        if(state_feature_type == "new_old2"):
            list_data = [
                         stats_dict["loss_mem_old"],stats_dict["loss_mem_new"]]
            # stats_dict.update({'correct_cnt_mem_old': correct_cnt_mem_old,
            #               'correct_cnt_mem_new': correct_cnt_mem_new,
            #               "loss_mem_old": loss_mem_old,
            #               "loss_mem_new": loss_mem_new,
            #                    "old_task_num":len(y_old),
            #                    "new_task_num":len(y_new)})
        elif(state_feature_type == "new_old4"):
            list_data = [stats_dict["correct_cnt_mem_old"],stats_dict["correct_cnt_mem_new"],
                         stats_dict["loss_mem_old"],stats_dict["loss_mem_new"]
                         ]
        elif (state_feature_type == "new_old5"):
            list_data = [i,stats_dict["correct_cnt_mem_old"], stats_dict["correct_cnt_mem_new"],
                     stats_dict["loss_mem_old"], stats_dict["loss_mem_new"]
                     ]

        elif (state_feature_type == "new_old7"):
            list_data = [i, stats_dict["correct_cnt_mem_old"], stats_dict["correct_cnt_mem_new"],
                         stats_dict["loss_mem_old"], stats_dict["loss_mem_new"],
                         stats_dict["correct_cnt_incoming"],stats_dict["loss_incoming_value"],
                         ]


        elif (state_feature_type == "new_old5t"):
            list_data = [task_seen,stats_dict["correct_cnt_mem_old"], stats_dict["correct_cnt_mem_new"],
                     stats_dict["loss_mem_old"], stats_dict["loss_mem_new"]
                     ]
        elif (state_feature_type == "new_old6"):
            list_data = [i,task_seen,stats_dict["correct_cnt_mem_old"], stats_dict["correct_cnt_mem_new"],
                     stats_dict["loss_mem_old"], stats_dict["loss_mem_new"]
                     ]

        elif (state_feature_type == "new_old9"):
            list_data = [i,stats_dict["correct_cnt_mem_old"], stats_dict["correct_cnt_mem_new"],
                     stats_dict["loss_mem_old"], stats_dict["loss_mem_new"],
                         correct_cnt_mem,loss_mem_value,correct_cnt_test_mem,loss_test_value,
                     ]
        elif (state_feature_type == "new_old11"):
            list_data = [i,stats_dict["correct_cnt_mem_old"], stats_dict["correct_cnt_mem_new"],
                     stats_dict["loss_mem_old"], stats_dict["loss_mem_new"],
                         correct_cnt_mem,loss_mem_value,correct_cnt_test_mem,loss_test_value,
                         correct_cnt_incoming,loss_incoming_value,
                     ]

        elif(state_feature_type == "task_dim"):
            # if("loss_mem_old" not in stats_dict):
            #     return None

            list_data = [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem, \
                         loss_incoming_value, loss_mem_value, loss_test_value,
                         stats_dict["loss_mem_old"],stats_dict["loss_mem_new"]]


        else:
            raise NotImplementedError("undefined state type",state_feature_type)
        state = np.array(list_data, dtype=np.float32).reshape([1, -1])
        state = torch.from_numpy(state)
        return state



    def get_reward(self,next_stats,prev_stats,):
        i= next_stats['batch_num']

        # correct_cnt_test_mem_prev = prev_stats["correct_cnt_test_mem"]
        # [correct_cnt_incoming, correct_cnt_mem,correct_cnt_test_mem,] = [
        #     next_stats['correct_cnt_incoming'],next_stats['correct_cnt_mem'], next_stats['correct_cnt_test_mem'],]

        # if(i==0):
        #     self.episode_start_test_acc = 100*correct_cnt_test_mem
        #     self.episode_start_test_loss = 100 * next_stats['loss_test_value']
        #     print("### i = 0 episode start",self.episode_start_test_acc)

        if(self.params.reward_type == "incoming_acc"):
            reward = correct_cnt_incoming#next_state[0,0].numpy()
        elif(self.params.reward_type == "mem_acc"):
            reward = correct_cnt_mem#next_state[0, 1].numpy()
        elif(self.params.reward_type == "test_acc"):

            reward = next_stats['correct_cnt_test_mem']
        elif (self.params.reward_type == "test_acc_rlt"):
            reward = (correct_cnt_test_mem - correct_cnt_test_mem_prev)
        elif(self.params.reward_type == "test_loss"):
            reward = -next_stats['loss_test_value']
        elif(self.params.reward_type == "test_loss_rlt"):
            reward = prev_stats['loss_test_value']-next_stats['loss_test_value']

        elif(self.params.reward_type == "acc_diff"):
            reward = - np.abs(next_stats['correct_cnt_mem']-next_stats['correct_cnt_test_mem']) \
                     - np.abs(next_stats['correct_cnt_mem'] - next_stats['correct_cnt_incoming'])

        elif(self.params.reward_type == "scaled"):
            #return np.log(correct_cnt_test_mem+1)
            reward = 100*correct_cnt_test_mem
        elif(self.params.reward_type == "real_reward"):
            reward =  100*self.CL_agent.evaluator.evaluate_model(self.model,self.CL_agent.task_seen)
        elif(self.params.reward_type == "multi-step"):
            #reward =  100*(correct_cnt_test_mem-correct_cnt_test_mem_prev)
            reward = next_stats['loss_test_value']-prev_stats['loss_test_value']
        elif(self.params.reward_type == "multi-step-0"):
            if((i+1) %self.params.done_freq==0):

                reward =  100*(correct_cnt_test_mem)
            else:
                reward = 0
        elif(self.params.reward_type == "multi-step-0-rlt"):
            if((i+1) %self.params.done_freq==0):

                reward =  100*(correct_cnt_test_mem)-self.episode_start_test_acc
                self.episode_start_test_acc = 100 * correct_cnt_test_mem
                print("___________________________________")
                print("### ",str(i)," episode start", self.episode_start_test_acc)
            else:
                reward = 0
        elif(self.params.reward_type == "multi-step-0-rlt-loss"):
            if((i+1) %self.params.done_freq==0):

                reward =  100*(next_stats['loss_test_value'])-self.episode_start_test_loss

                self.episode_start_test_loss = 100 * next_stats['loss_test_value']
                print("___________________________________")
                print("### ",str(i)," episode start", self.episode_start_test_loss)
            else:
                reward = 0
        else:
            raise NotImplementedError("not implemented reward error")

        if (self.params.reward_test_type == "reverse"):
            reward = -reward
        # elif (self.params.reward_test_type == "relative"):
        #     reward = self.get_reward(next_stats[:3])  # -correct_cnt_test_mem*100
        elif (self.params.reward_test_type == "None"):
            pass
        else:
            raise NotImplementedError("undefined reward test type ", self.params.reward_test_type)
        reward = reward - self.params.reward_rg * np.abs(next_stats['loss_mem_value']-next_stats['loss_test_value'])#action_mem_iter
        return reward

    def update_task_reward(self):
        if(len(self.reward_list) == 0): return
        self.task_reward_list.append(self.reward_list[-1])
    def save_task_reward(self,prefix):

        arr = np.array(self.task_reward_list)
        np.save(prefix + "task_reward_list.npy", arr)








    # def compute_pre_test_loss(self,buffer):
    #     grad_dims = []
    #     for param in buffer.model.parameters():
    #         grad_dims.append(param.data.numel())
    #     grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
    #     model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
    #     ## compute loss on test batch
    #     if (not self.test_batch_x == None):
    #         logits = model_temp.forward(self.test_batch_x)
    #         self.pre_loss_test = F.cross_entropy(logits, self.test_batch_y, reduction='none')
    #
    #         #_, pred_label = torch.max(logits, 1)
    #         # print(pred_label) ## TODO: never predict classes not seen
    #         #self.pre_loss_test = (pred_label == self.test_batch_y).sum().item() / self.test_batch_y.size(0)
    #
    #
    # def reward_post_loss(self,):
    #     with torch.no_grad():
    #         logits = self.model.forward(self.test_batch_x)
    #         post_loss = F.cross_entropy(logits, self.test_batch_y, reduction='none')
    #         reward = self.pre_loss_test-post_loss
    #
    #     return reward

    # def pre_sgd_loss(self,test_buffer,buffer):
    #     ## test memory batch
    #     if (self.params.reward_type == "relative"):
    #
    #         self.get_test_batch(test_buffer)
    #         self.compute_pre_test_loss(buffer)
    #
    # def get_test_batch(self,test_memory):
    #     if(test_memory.current_index==0):
    #         print("Test memory is empty")
    #         return -1
    #     self.test_batch_x,self.test_batch_y = random_retrieve(test_memory,self.params.test_mem_batchSize)







