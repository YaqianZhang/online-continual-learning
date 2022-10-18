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


class ExperienceReplay_ratio(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_ratio, self).__init__(model, opt, params)
        self.mem_ratio = None
        task_size_dict={
            "cifar100":2500,
            "mini_imagenet":5000,
            "clrs25":2500,
            "core50":10000
        }
        self.task_data_size = task_size_dict[self.params.data]

    def adjust_mem_ratio_rs(self,batch_num):
        if(self.task_seen ==0):
            self.mem_ratio = 1
            return
        if(self.params.save_prefix  == "debug"):
                self.mem_ratio = self.task_seen
        else:
            N_current = 10*batch_num
            N_past = self.task_seen * self.task_data_size

            beta_inv =1+2*N_current/N_past
            self.mem_ratio = beta_inv * self.task_seen * self.params.mem_size/self.task_data_size

    def adjust_mem_ratio(self):
        if(self.task_seen >0):
            if(self.params.save_prefix  == "debug"):
                self.mem_ratio = self.task_seen
            else:
                self.mem_ratio = self.task_seen * self.params.mem_size/self.task_data_size
        else:
            self.mem_ratio =1



    def _batch_update_ratio(self,batch_x,batch_y,losses_batch,acc_batch,i,replay_para=None,mem_num=0):
        self.model.train()
        #print("ratio train")
        STOP_FLAG = False
        if(replay_para == None):
            replay_para = self.replay_para


        # if(self.params.test_mem_recycle):
        #     recycle_test_x = recycle.store_tmp(img,cls_max)
        # ## todo : cutmix
        # do_cutmix = self.params.do_cutmix and np.random.rand(1) < self.params.cutmix_prob
        # if do_cutmix:
        #     # print(x.shape)
        #     ce = torch.nn.CrossEntropyLoss(reduction='mean')
        #
        #     x, labels_a, labels_b, lam = cutmix_data(x=batch_x, y=batch_y, alpha=1.0,index="None")
        #     #logits = self._compute_softmax_logits(x)
        #     logits = self.model.forward(x)
        #
        #     loss = lam * ce(logits, labels_a) + (1 - lam) * ce(
        #         logits, labels_b
        #     )
        #     avrg_acc = 0
        #     train_stats = None
        #
        #
        # else:





        logits = self.model.forward(batch_x)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == batch_y)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, batch_y)

        total_num = batch_x.shape[0]
        avrg_acc = acc.sum().item() / total_num
        #loss = torch.mean(softmax_loss_full)






        acc_incoming = acc[mem_num:].sum().item() / (total_num - mem_num)

        incoming_loss = torch.mean(softmax_loss_full[mem_num:])
        self.train_loss_incoming.append(incoming_loss.item())
        self.train_acc_incoming.append(acc_incoming)

        if(mem_num>0):

            acc_mem = acc[:mem_num].sum().item() / mem_num
            mem_loss = torch.mean(softmax_loss_full[:mem_num])
            self.train_acc_mem.append(acc_mem)
            self.train_loss_mem.append(mem_loss.item())
            # if(self.close_loop_cl != None):### used for state construction
            #
            #     self.close_loop_cl.last_train_loss = mem_loss.item()


            #loss = mem_loss+ incoming_loss
            loss = self.mem_ratio * mem_loss+ \
                   1 * incoming_loss
            #loss = torch.mean(softmax_loss_full)
            train_stats = {'acc_incoming': acc_incoming,
                           'acc_mem': acc_mem,
                           "loss_incoming": incoming_loss.item(),
                           "loss_mem": mem_loss.item(),
                           "batch_num": i,
                           }

        else:
            #loss = torch.mean(softmax_loss_full)
            loss = replay_para['incoming_ratio'] * incoming_loss
            #loss = incoming_loss
            acc_mem = None
            mem_loss = None
            train_stats=None


        acc_batch.update(avrg_acc, batch_y.size(0))
        losses_batch.update(loss.item(), batch_y.size(0))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_batch.append(loss.item())



        return  train_stats

    def train_learner(self, x_train, y_train):
        if(self.params.adaptive_ratio_type == "offline"):
            self.adjust_mem_ratio()



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
                if(self.params.adaptive_ratio_type == "online"):
                    self.adjust_mem_ratio_rs(i)
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.set_memIter()
                memiter=self.mem_iters
                #self.aug_N_list.append(self.N)

                for j in range(self.mem_iters):

                    #self.set_aug_para(N, int(j*30/self.mem_iters), incoming_M=int(j*30/self.mem_iters))


                    concat_batch_x,concat_batch_y,mem_num = self.concat_memory_batch(batch_x,batch_y)




                    stats = self._batch_update_ratio(concat_batch_x,concat_batch_y, losses_batch, acc_batch, i,mem_num=mem_num)
                    STOP_FLAG = self.early_stop_check(stats)
                    if(STOP_FLAG ):
                        memiter=j+1
                        break
                    test_acc,test_loss = self.immediate_evaluate()
                    test_acc_list.append(test_acc)
                    self.test_acc_true.append(test_acc)
                    self.test_loss_true.append(test_loss)


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
                    print("mem_iter", memiter,concat_batch_y.shape,"ratio",self.mem_ratio,self.task_seen)
        self.after_train()
        return test_acc_list


