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


class ExperienceReplay_aug(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_aug, self).__init__(model, opt, params)
        self.N = None

    def auto_adjust(self,stats):
        if(stats['acc_mem']<0.8):
            N = 1
        elif(stats['acc_mem']<0.85):
            N = 2
        elif(stats['acc_mem']<0.9):
            N = 3
        else:
            N = 4

        self.N = N
        self.aug_agent.set_aug_para(N, N)
        return N


    def adjust_aug(self,):
        # if(stats == None):
        #     return
        if(self.task_seen <5):
            N=1
        elif(self.task_seen < 10):
            N = 2
        elif(self.task_seen<15):
            N = 3
        elif(self.task_seen<20):
            N = 4
        # if(stats['acc_mem']>self.params.train_acc_max_aug):
        #     N = self.aug_agent.current_N +1
        #     if(N>15):
        #         N = 15
        #     self.aug_agent.set_aug_para(N,self.aug_agent.current_M)
        # elif(stats['acc_mem']<self.params.train_acc_min_aug):
        #     N = int(self.aug_agent.current_N /2)
        #     if(N<1):
        #         N = 1
        #
        self.N = N
        self.aug_agent.set_aug_para(N, N)




    def train_learner(self, x_train, y_train):

        #self.adjust_aug()


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

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.set_memIter()
                memiter=self.mem_iters

                for j in range(self.mem_iters):

                    #self.set_aug_para(N, int(j*30/self.mem_iters), incoming_M=int(j*30/self.mem_iters))


                    concat_batch_x,concat_batch_y,mem_num = self.concat_memory_batch(batch_x,batch_y)




                    stats = self._batch_update(concat_batch_x,concat_batch_y, losses_batch, acc_batch, i,mem_num=mem_num)
                    STOP_FLAG = self.early_stop_check(stats)
                    if(STOP_FLAG ):
                        memiter=j+1
                        break
                    test_acc,test_loss = self.immediate_evaluate()
                    test_acc_list.append(test_acc)
                    self.test_acc_true.append(test_acc)
                    self.test_loss_true.append(test_loss)
                    if(stats != None):
                        N = self.auto_adjust(stats)
                        self.aug_N_list.append(N)
                        print(N)

                self.mem_iter_list.append(memiter)
                # if(self.params.use_test_buffer):
                #     self.close_loop_cl.compute_testmem_loss()
                self.memory_manager.update_memory(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    # print(
                    #     '==>>> it: {}, mem avg. loss: {:.6f}, '
                    #     'running mem acc: {:.3f}'
                    #         .format(i, losses_mem.avg(), acc_mem.avg())
                    # )
                    print("mem_iter", memiter,concat_batch_y.shape,"aug",self.N)
        self.after_train()
        return test_acc_list


