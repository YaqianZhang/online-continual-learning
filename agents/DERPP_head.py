import torch
from torch.utils import data
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.utils import maybe_cuda, AverageMeter
# from RL.RL_replay_base import RL_replay
# from RL.close_loop_cl import close_loop_cl
from torchvision.transforms import transforms

import numpy as np
import torch
import torch.nn as nn
from utils.setup_elements import transforms_match
from utils.utils import cutmix_data
from agents.exp_replay import ExperienceReplay
import time
from agents.DER import DER
from RL.pytorch_util import  build_mlp

from utils.setup_elements import n_classes

class DERPP_head(DER):
    def __init__(self, model, opt, params):
        super(DERPP_head, self).__init__(model, opt, params)

        self.buffer = self.memory_manager.buffer
        self.alpha = params.DER_alpha #0.3
        class_num = n_classes[params.data]
        self.predictor_head = maybe_cuda(build_mlp(input_size=class_num,
                                                 output_size=class_num,
                                                 n_layers=self.params.phead_layer,
                                                 size=self.params.phead_size, #1024
                                                   output_activation="relu",
                                                 use_dropout=self.params.softmax_dropout,))
        # self.predictor_head_opt =  torch.optim.SGD(self.predictor_head.parameters(),
        #                                     lr=self.params.softmaxhead_lr,
        #                                     )
        self.opt =torch.optim.SGD(list(model.parameters())+list(self.predictor_head.parameters()),
                                lr=self.params.learning_rate,
                                weight_decay=self.params.weight_decay)






    def train_learner(self, x_train, y_train): ## todo: move build of dataloader from agent to dataset script


        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=4,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()
        test_acc_list=[]
        acc_main=None
        acc_add = None

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.set_memIter()
                memiter=self.mem_iters
                start_time = time.time()

                for j in range(self.mem_iters):
                    self.opt.zero_grad()




                    aug_batch_x = self.aug_data(batch_x)
                    aug_inc_logits = self.model.forward(aug_batch_x)

                    # ce_all = torch.nn.CrossEntropyLoss(reduction='none')
                    # softmax_loss_full = ce_all(aug_inc_logits, batch_y)
                    # incoming_loss = torch.mean(softmax_loss_full)

                    ce_mean = torch.nn.CrossEntropyLoss(reduction='mean')
                    der_loss = ce_mean(aug_inc_logits,batch_y)
                    incoming_ce_loss = der_loss.item()

                    self.train_loss_incoming.append(der_loss.item())

                    mem_x, mem_y,mem_logits = self.memory_manager.retrieve_from_mem_logits(batch_x, batch_y,
                                                                         retrieve_num=10)
                    if mem_x.shape[0]>0:
                        aug_mem_x = self.aug_data(mem_x)
                        aug_mem_logits = self.model.forward(aug_mem_x)
                        aug_mem_logits_transform = self.predictor_head(aug_mem_logits)
                        mse_loss = torch.nn.MSELoss()
                        mem_reg_loss = self.alpha * mse_loss(aug_mem_logits_transform,mem_logits)
                        #self.train_loss_mem.append(mem_reg_loss.item())
                        der_loss += mem_reg_loss
                        #print(mem_reg_loss)
                        #assert False
                    if mem_x.shape[0]>0:
                        aug_mem_x = self.aug_data(mem_x)
                        aug_mem_logits = self.model.forward(aug_mem_x)
                        mem_ce_loss = ce_mean(aug_mem_logits,mem_y)
                        der_loss += mem_ce_loss

                    self.loss_batch.append(der_loss.item())
                    losses_batch.update(der_loss.item(), batch_y.size(0))

                    der_loss.backward()
                    self.opt.step()
                end_time = time.time()




                self.mem_iter_list.append(memiter)
                # if(self.params.use_test_buffer):
                #     self.close_loop_cl.compute_testmem_loss()
                self.memory_manager.update_memory_logits(batch_x, batch_y,aug_inc_logits.data)

                if i % 100 == 1 and self.verbose and ep % 10 ==0:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f} '
                        'ep:{:.0f}'
                            .format(i, losses_batch.avg(), acc_batch.avg(),ep)
                    )
                    print('time:{:.2f} '
                          'inc-ce-loss:{:.2f} '
                          "mem-logit-loss:{:.2f} "
                          'mem-ce-loss:{:.2f} '
                          .format(end_time-start_time,incoming_ce_loss,mem_reg_loss.item(),mem_ce_loss.item())
                          )

        self.after_train()
        return test_acc_list
