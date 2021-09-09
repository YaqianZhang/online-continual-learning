import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from RL.pytorch_util import  build_mlp
import numpy as np
from utils.utils import cutmix_data
from models.modelfactory import ModelFactory
import models.learner as Learner
import math
from agents.scr import SupContrastReplay

class SupContrastReplay_meta(SupContrastReplay):
    def __init__(self, model, opt, params):
        super(SupContrastReplay_meta, self).__init__(model, opt, params)

        if(params.data in [ 'clrs25', 'core50']):
            softmax_inputdim = 2560
        elif(params.data in ['cifar100','cifar10']):
            softmax_inputdim = 160
        else:
            raise NotImplementedError("undefined dataset",params.data)
        print(softmax_inputdim)

        # if(params.softmax_type == "seperate"):
        #     self.seperate_softmax(softmax_inputdim)
        # elif(params.softmax_type == "meta"):
        self.meta_softmax(params,softmax_inputdim)



    def meta_softmax(self,params,softmax_inputdim):
        config = ModelFactory.get_model(model_type="linear_softmax",sizes = softmax_inputdim,
                                           dataset=params.data,hidden_size=params.softmax_nsize)
        self.softmax_head = Learner.Learner(config, params)
        self.softmax_head.cuda()
        # define the lr params
        self.softmax_head.define_task_lr_params(alpha_init=params.alpha_init)

        self.opt_wt = torch.optim.SGD(list(self.softmax_head.parameters()), lr=params.opt_wt)
        self.opt_lr = torch.optim.SGD(list(self.softmax_head.alpha_lr.parameters()), lr=params.opt_lr)



    def inner_update(self, x, fast_weights, y, ):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        ce = torch.nn.CrossEntropyLoss(reduction='mean')

        h_feature = self.model.features(x)
        logits = self.softmax_head.forward(h_feature,fast_weights)
        loss = ce(logits, y)



        if fast_weights is None:
            fast_weights = self.softmax_head.parameters()

        graph_required = self.params.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))

        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], min = -self.params.grad_clip_norm, max = self.params.grad_clip_norm)
        self.softmax_head.alpha_lr.cuda()
        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.softmax_head.alpha_lr))))
        # lr_value =self.softmax_head.alpha_lr[11][0]
        #
        # self.learning_rate_list.append(lr_value.item())
        return fast_weights

    def meta_loss(self, x, fast_weights, y, ):
        """
        differentiate the loss through the network updates wrt alpha
        """
        ce = torch.nn.CrossEntropyLoss(reduction='mean')

        h_feature = self.model.features(x)
        logits = self.softmax_head.forward(h_feature,fast_weights)
        loss = ce(logits, y)

        return loss, logits
    def zero_grads(self):
        if self.params.learn_lr:
            self.opt_lr.zero_grad()
        self.opt_wt.zero_grad()
        self.softmax_head.zero_grad()
        self.softmax_head.alpha_lr.zero_grad()


    def perform_softmax_training(self, x, y, mem_num):
        if (self.params.use_softmaxloss == False):
            return

        total_num = x.shape[0]
        inner_x = x[mem_num:]
        inner_y = y[mem_num:]
        outer_x = x
        outer_y = y

        self.softmax_head.train()

        # perm = torch.randperm(x.size(0))
        # x = x[perm]
        # y = y[perm]


        self.zero_grads()



        batch_sz = inner_x.shape[0]
        n_batches = 5 #self.args.cifar_batches
        rough_sz = math.ceil(batch_sz / n_batches)
        fast_weights = None
        meta_losses = [0 for _ in range(n_batches)]


        for i in range(n_batches):

            batch_x = inner_x[i * rough_sz: (i + 1) * rough_sz]
            batch_y = inner_y[i * rough_sz: (i + 1) * rough_sz]

            # assuming labels for inner update are from the same
            fast_weights = self.inner_update(batch_x, fast_weights, batch_y, )
            # only sample and push to replay buffer once for each task's stream
            # instead of pushing every epoch
            meta_loss, logits = self.meta_loss(outer_x, fast_weights, outer_y, )

            meta_losses[i] += meta_loss

        # Taking the meta gradient step (will update the learning rates)
        self.zero_grads()

        meta_loss = sum(meta_losses) / len(meta_losses)
        meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.softmax_head.alpha_lr.parameters(), self.params.grad_clip_norm)

        torch.nn.utils.clip_grad_norm_(self.softmax_head.parameters(), self.params.grad_clip_norm)
        if self.params.learn_lr:
            self.opt_lr.step()

        # if sync-update is being carried out (as in sync-maml) then update the weights using the optimiser
        # otherwise update the weights with sgd using updated LRs as step sizes
        if (self.params.sync_update):
            self.opt_wt.step()
        else:
            for i, p in enumerate(self.softmax_head.parameters()):
                # using relu on updated LRs to avoid negative values
                p.data = p.data - p.grad * nn.functional.relu(self.softmax_head.alpha_lr[i])
        self.softmax_head.zero_grad()
        self.softmax_head.alpha_lr.zero_grad()


        # index = "None"
        # if (self.params.softmax_membatch < mem_num):
        #     selected_idx = np.random.shuffle(np.range(0, mem_num))[:self.params.softmax_membatch]
        #
        #     # selected_idx = np.random.randint(0,mem_num,self.params.softmax_membatch)
        #     selected_idx = list(selected_idx) + list(np.arange(mem_num, total_num))
        #     # print(mem_num,selected_idx)
        #     # assert False
        #     x = x[selected_idx]
        #     y = y[selected_idx]
        #     mem_num = self.params.softmax_membatch

        # ## todo : cutmix
        # do_cutmix = self.params.do_cutmix and np.random.rand(1) < 0.5
        # if do_cutmix:
        #     # print(x.shape)
        #
        #     x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0, index=index)
        #     logits = self._compute_softmax_logits(x)
        #     # h_feature = self.model.features(x)
        #     # logits = self.softmax_head(h_feature)
        #     softmax_loss = lam * ce(logits, labels_a) + (1 - lam) * ce(
        #         logits, labels_b
        #     )
        # else:
        #     logits = self._compute_softmax_logits(x)
        #
        #     ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        #     softmax_loss_full = ce_all(logits, y)
        #
        #     mem_loss = torch.mean(softmax_loss_full[:mem_num])
        #     incoming_loss = torch.mean(softmax_loss_full[mem_num:])
        #     self.train_loss_mem.append(mem_loss.item())
        #     self.train_loss_incoming.append(incoming_loss.item())
        #     _, pred_label = torch.max(logits, 1)
        #     acc = (pred_label == y)
        #     acc_mem = acc[:mem_num].sum().item() / mem_num
        #     acc_incoming = acc[mem_num:].sum().item() / (total_num - mem_num)
        #     self.train_acc_mem.append(acc_mem)
        #     self.train_acc_incoming.append(acc_incoming)
        #     # softmax_loss = torch.mean(softmax_loss_full)
        #     softmax_loss = (self.params.mem_ratio * torch.sum(softmax_loss_full[:mem_num]) + \
        #                     self.params.incoming_ratio * torch.sum(softmax_loss_full[mem_num:])) / total_num
        #     if (self.params.use_test_buffer and self.memory_manager.buffer.current_index > 0):
        #         self.compute_testmem_loss()
        #
        # self.softmax_opt.zero_grad()
        # softmax_loss.backward()
        # self.softmax_opt.step()



