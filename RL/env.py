import torch
import torch.nn.functional as F
from utils.buffer.buffer_utils import random_retrieve, get_grad_vector
import copy
from utils.utils import maybe_cuda
import numpy as np
from utils.buffer.buffer_utils import random_retrieve


class RL_env(object):
    def __init__(self, params,model):
        super().__init__()
        self.model = model

        self.params = params
        self.reward_type = params.reward_type
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
        self.size1 = params.action_size
        self.action_space_replay=maybe_cuda(torch.linspace(-2,2,steps=self.size1))
        self.action_space_timestamp = maybe_cuda(torch.linspace(-2, 2, steps=self.size1))
        self.reward_list = []
        self.task_reward_list=[]
        self.test_batch_x=None
        self.test_batch_y=None
        self.pre_loss_test =None

    def pre_sgd_loss(self,test_buffer,buffer):
        ## test memory batch
        if (self.params.reward_type == "relative"):

            self.get_test_batch(test_buffer)
            self.compute_pre_test_loss(buffer)

    # def compute_acc_test_buffer(self,test_buffer):
    #    # reward = None
    #     if (self.params.reward_type == "relative"):
    #         # if (self.RL_env.pre_loss_test == None):
    #         #     pass
    #         # else:
    #         reward = torch.mean(self.reward_post_loss()).cpu()
    #
    #     else:
    #         reward = self.acc_test_buffer(test_buffer)
    #     #if not (reward == None):
    #     self.reward_list.append(reward)
    #     return reward

    def acc_test_buffer(self,test_memory,):

        if(test_memory.current_index==0):
            print("Test memory is empty")
            return None

        with torch.no_grad():
            batch_x,batch_y = random_retrieve(test_memory,self.params.test_mem_batchSize)
            logits = self.model.forward(batch_x)
            _, pred_label = torch.max(logits, 1)
            #print(pred_label) ## TODO: never predict classes not seen
            correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

        return correct_cnt

    def get_reward(self,stats,evaluator,model,task_seen,correct_cnt_test_mem_prev):
        #reward = next_state[0,1].numpy() # todo: RL q test
        [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem] = stats
        if(self.params.reward_type == "incoming_acc"):
            reward = correct_cnt_incoming#next_state[0,0].numpy()
        elif(self.params.reward_type == "mem_acc"):
            reward = correct_cnt_mem#next_state[0, 1].numpy()
        elif(self.params.reward_type == "test_acc"):

            reward = correct_cnt_test_mem#next_state[0, 2].numpy()
        elif(self.params.reward_type == "scaled"):
            #return np.log(correct_cnt_test_mem+1)
            reward = 100*correct_cnt_test_mem
        elif(self.params.reward_type == "real_reward"):
            reward =  100*evaluator.evaluate_model(model,task_seen)
        elif(self.params.reward_type == "multi-step"):
            reward =  100*(correct_cnt_test_mem-correct_cnt_test_mem_prev)
        else:
            raise NotImplementedError("not implemented reward error")

        if (self.params.reward_test_type == "reverse"):
            reward = -reward
        elif (self.params.reward_test_type == "relative"):
            reward = self.RL_env.get_reward(stats[:3])  # -correct_cnt_test_mem*100
        elif (self.params.reward_test_type == "None"):
            pass
        else:
            raise NotImplementedError("undefined reward test type ", self.params.reward_test_type)
        return reward






    # def compute_state(self,buffer,selected_mem_indices):
    #     ## MIR loss
    #     ## replay times
    #     ## last replay timestamp
    #     MIR_loss_ranking = self.compute_MIR_loss(buffer,selected_mem_indices)
    #     replay_times_ranking = self.compute_replay_times(buffer,selected_mem_indices)
    #     replay_timestamp_ranking = self.compute_replay_timestamp(buffer,selected_mem_indices)
    #     state = torch.cat((MIR_loss_ranking,replay_times_ranking,replay_timestamp_ranking))
    #     return state

    def compute_replay_times(self,buffer,selected_mem_indices):
        replay_times_list = buffer.buffer_replay_times[selected_mem_indices]
        ranking = torch.argsort(replay_times_list, descending=True)
        return ranking
    def compute_replay_timestamp(self,buffer,selected_mem_indices):
        replay_timestamp = buffer.buffer_replay_times[selected_mem_indices]
        ranking = torch.argsort(replay_timestamp,descending=True)
        return ranking

    def compute_pre_test_loss(self,buffer):
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        ## compute loss on test batch
        if (not self.test_batch_x == None):
            logits = model_temp.forward(self.test_batch_x)
            self.pre_loss_test = F.cross_entropy(logits, self.test_batch_y, reduction='none')

            #_, pred_label = torch.max(logits, 1)
            # print(pred_label) ## TODO: never predict classes not seen
            #self.pre_loss_test = (pred_label == self.test_batch_y).sum().item() / self.test_batch_y.size(0)

    def compute_MIR_loss(self,buffer,selectede_mem_indices):
        sub_x = buffer.buffer_img[selectede_mem_indices]
        sub_y = buffer.buffer_label[selectede_mem_indices]
        sub_x = maybe_cuda(sub_x)
        sub_y = maybe_cuda(sub_y)
        grad_dims = []
        for param in buffer.model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(buffer.model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(buffer.model, grad_vector, grad_dims)
        ## compute loss on test batch

        with torch.no_grad():
            logits_pre = buffer.model.forward(sub_x)
            logits_post = model_temp.forward(sub_x)
            pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
            post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
            scores = post_loss - pre_loss
            ranking = torch.argsort(scores, descending=True)


            # if(not self.test_batch_x == None):
            #
            #     logits = model_temp.forward(self.test_batch_x)
            #     self.pre_loss_test = F.cross_entropy(logits, self.test_batch_y, reduction='none')

        return ranking
    def from_action_to_coef(self,action):

        idx1 = torch.floor_divide(action, self.size1)#action/self.size1
        idx2 = action - idx1 * self.size1
        coef1 = self.action_space_replay[idx1]
        coef2 = self.action_space_timestamp[idx2]

        #print("selected action", action,coef1,coef2)

        return coef1, coef2

    def from_action_to_indices(self,action,buffer,selected_mem_indices):
        coef2 = self.action_space_timestamp[action]
        MIR_loss_ranking = maybe_cuda(self.compute_MIR_loss(buffer,selected_mem_indices))
        #replay_times_ranking = maybe_cuda(self.compute_replay_times(buffer,selected_mem_indices))
        replay_timestamp_ranking = maybe_cuda(self.compute_replay_timestamp(buffer,selected_mem_indices))
        idex = MIR_loss_ranking +  coef2*replay_timestamp_ranking
        idx = torch.sort(idex)[1][:self.num_retrieve]

        #assert False
        return idx
    def get_test_batch(self,test_memory):
        if(test_memory.current_index==0):
            print("Test memory is empty")
            return -1
        self.test_batch_x,self.test_batch_y = random_retrieve(test_memory,self.params.test_mem_batchSize)
    def reward_post_loss(self,):
        with torch.no_grad():

            logits = self.model.forward(self.test_batch_x)
            post_loss = F.cross_entropy(logits, self.test_batch_y, reduction='none')
            reward = self.pre_loss_test-post_loss

        return reward


    def update_task_reward(self):
        if(len(self.reward_list) == 0): return
        self.task_reward_list.append(self.reward_list[-1])
    def save_reward(self,prefix):

        arr = np.array(self.reward_list)

        np.save(prefix + "reward_list.npy", arr)

        arr = np.array(self.task_reward_list)
        np.save(prefix + "task_reward_list.npy", arr)
        print("save reward")


    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.learning_rate * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1


class RL_env_2dim(RL_env):
    def __init__(self, params,model):
        super().__init__(params,model)

    def from_action_to_indices(self,action,buffer,selected_mem_indices):
        coef1,coef2 = self.from_action_to_coef(action)
        MIR_loss_ranking = maybe_cuda(self.compute_MIR_loss(buffer,selected_mem_indices))
        replay_times_ranking = maybe_cuda(self.compute_replay_times(buffer,selected_mem_indices))*coef1
        replay_timestamp_ranking = maybe_cuda(self.compute_replay_timestamp(buffer,selected_mem_indices))*coef2

        idex = MIR_loss_ranking + coef1*replay_times_ranking + coef2*replay_timestamp_ranking
        idx = torch.sort(idex)[1][:self.num_retrieve]
        return idx
