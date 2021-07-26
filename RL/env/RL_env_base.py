import torch
import numpy as np
from utils.buffer.buffer_utils import random_retrieve
from utils.utils import maybe_cuda
import torch.nn.functional as F
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector


class Base_RL_env(object):
    def __init__(self, params,model,testBuffer,RL_agent,CL_agent):
        super().__init__()
        self.model = model
        self.test_buffer = testBuffer
        self.RL_agent = RL_agent
        self.CL_agent = CL_agent

        self.params = params
        self.reward_type = params.reward_type

        self.reward_list = []
        self.task_reward_list=[]



    def get_reward(self,next_stats,prev_stats,i,action, action_mem_iter, action_incoming_ratio,action_mem_ratio):


        #reward = next_state[0,1].numpy() # todo: RL q test
        correct_cnt_test_mem_prev = prev_stats["correct_cnt_test_mem"]
        [correct_cnt_incoming, correct_cnt_mem,correct_cnt_test_mem,] = [
            next_stats['correct_cnt_incoming'],next_stats['correct_cnt_mem'], next_stats['correct_cnt_test_mem'],]

        if(i==0):
            self.episode_start_test_acc = 100*correct_cnt_test_mem
            self.episode_start_test_loss = 100 * next_stats['loss_test_value']
            print("### i = 0 episode start",self.episode_start_test_acc)

        if(self.params.reward_type == "incoming_acc"):
            reward = correct_cnt_incoming#next_state[0,0].numpy()
        elif(self.params.reward_type == "mem_acc"):
            reward = correct_cnt_mem#next_state[0, 1].numpy()
        elif(self.params.reward_type == "test_acc"):
            reward = correct_cnt_test_mem
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
            reward =  100*(correct_cnt_test_mem-correct_cnt_test_mem_prev)
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







