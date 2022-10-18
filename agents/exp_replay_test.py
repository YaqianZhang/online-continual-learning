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
import time
from agents.exp_replay import ExperienceReplay

class ER_test(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ER_test, self).__init__(model, opt, params)


    def before_train(self, train_loader, task_labels):
        new_labels = task_labels
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i
        # print("set new labels",self.lbl_inv_map)

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

    def train_learner(self, train_loader, task_labels): ## todo: move build of dataloader from agent to dataset script


        self.before_train(train_loader,task_labels)
        # set up loader
        # train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        # train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=4,
        #                                drop_last=True)
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
                start_time = time.time()

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

                    if(self.params.test_add_buffer == True):
                        acc_main, acc_add = self.test_add_buffer(batch_x,batch_y)


                self.mem_iter_list.append(memiter)
                # if(self.params.use_test_buffer):
                #     self.close_loop_cl.compute_testmem_loss()
                self.memory_manager.update_memory(batch_x, batch_y)

                end_time = time.time()

                if i % 100 == 1 and self.verbose and ep % 20 == 0:
                #if self.verbose and ep % 20 == 0:
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
                    print("mem_iter", memiter,"time",end_time-start_time,acc_main,acc_add)
        self.after_train()
        return test_acc_list


