import torch

import torch.nn.functional as F
from utils.buffer.buffer_utils import get_grad_vector
import copy
from utils.utils import maybe_cuda

from RL.env.RL_env_base import Base_RL_env

class RL_env_memControl(Base_RL_env):
    def __init__(self,params,model,testBuffer):
        super.__init__(params,model,testBuffer)
        self.subsample = params.subsample
        self.num_retrieve = params.eps_mem_batch
        self.size1 = params.action_size
        self.action_space_replay=maybe_cuda(torch.linspace(-2,2,steps=self.size1))
        self.action_space_timestamp = maybe_cuda(torch.linspace(-2, 2, steps=self.size1))
        self.test_batch_x=None
        self.test_batch_y=None
        self.pre_loss_test =None


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

    def compute_MIR_loss(self, buffer, selectede_mem_indices):
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

    def compute_replay_times(self,buffer,selected_mem_indices):
        replay_times_list = buffer.buffer_replay_times[selected_mem_indices]
        ranking = torch.argsort(replay_times_list, descending=True)
        return ranking
    def compute_replay_timestamp(self,buffer,selected_mem_indices):
        replay_timestamp = buffer.buffer_replay_times[selected_mem_indices]
        ranking = torch.argsort(replay_timestamp,descending=True)
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

class RL_env_memControl_2dim(RL_env_memControl):
    def __init__(self, params, model):
        super().__init__(params, model)

    def from_action_to_indices(self, action, buffer, selected_mem_indices):
        coef1, coef2 = self.from_action_to_coef(action)
        MIR_loss_ranking = maybe_cuda(self.compute_MIR_loss(buffer, selected_mem_indices))
        replay_times_ranking = maybe_cuda(self.compute_replay_times(buffer, selected_mem_indices)) * coef1
        replay_timestamp_ranking = maybe_cuda(self.compute_replay_timestamp(buffer, selected_mem_indices)) * coef2

        idex = MIR_loss_ranking + coef1 * replay_times_ranking + coef2 * replay_timestamp_ranking
        idx = torch.sort(idex)[1][:self.num_retrieve]
        return idx

