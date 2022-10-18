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

class ExperienceReplay_batchsize(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_batchsize, self).__init__(model, opt, params)


    def _batch_update(self,batch_x,batch_y,losses_batch,acc_batch,i,replay_para=None,mem_num=0):
        self.model.train()
        STOP_FLAG = False
        if(replay_para == None):
            replay_para = self.replay_para




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
            #loss = replay_para['mem_ratio'] * mem_loss+ \
                   #replay_para['incoming_ratio'] * incoming_loss
            loss = torch.mean(softmax_loss_full)
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

    def adjust_aug(self,):
        if(self.params.adjust_aug_flag == False):
            return
        # if(stats == None):
        #     return

        if(self.task_seen <3):
            N=1
        elif (self.task_seen < 6):
            N = 2
        elif(self.task_seen < 9):
            N = 3
        elif(self.task_seen<12):
            N = 4
        elif(self.task_seen<15):
            N = 5
        elif(self.task_seen<18):
            N = 6
        else:
            N = 7

        self.N = N
        self.aug_agent.set_aug_para(N, N)
        return self.N
    def adaptive_batchsize(self):
        if(self.task_seen == 0):
            return 10
        else:
            return self.task_seen*10

    def adjust_iter(self, train_acc_list):
        if(self.params.adjust_iter_flag == False):
            return
        target_acc_start = self.params.train_acc_min  # 0.80
        target_acc_end = self.params.train_acc_max  # 0.9
        current_iter = len(train_acc_list)
        if (current_iter == 0 or current_iter == None):
            return self.params.mem_iters
        last_acc = train_acc_list[-1]
        max_acc = np.max(train_acc_list)
        mean_acc = np.mean(train_acc_list)
        acc = mean_acc
        # slope, intercept, r, p, se = linregress(np.arange(0,current_iter), train_acc_list)

        # print(r, p, train_acc_list)
        if (last_acc < target_acc_start):  ## under fitting condition
            ## increase
            mem_iter = current_iter + 2
            if (mem_iter > self.params.mem_iter_max):
                mem_iter = self.params.mem_iter_max
            return mem_iter
        elif (max_acc > target_acc_end):  ## overfitting condition
            # print(r,p,train_acc_list)
            return int(current_iter / 2)
        else:
            return current_iter
    def train_learner(self, x_train, y_train):


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
        train_acc = None
        train_acc_list=[]

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.set_memIter()
                memiter=self.mem_iters
                memiter = self.adjust_iter(train_acc_list)
                #print(memiter)
                train_acc_list=[]
                self.N = self.adjust_aug()
                adaptive_batch_num = self.adaptive_batchsize()
                self.mem_batchsize_list.append(adaptive_batch_num)

                self.aug_N_list.append(self.N)
                for j in range(memiter+1):

                    #self.set_aug_para(N, int(j*30/self.mem_iters), incoming_M=int(j*30/self.mem_iters))


                    concat_batch_x,concat_batch_y,mem_num = self.concat_memory_batch(batch_x,batch_y,retrieve_num =adaptive_batch_num )




                    train_stats = self._batch_update(concat_batch_x,concat_batch_y, losses_batch, acc_batch, i,mem_num=mem_num)
                    if(train_stats != None):
                        train_acc_list.append(train_stats['acc_mem'])
                        #train_loss_list.append(train_stats['loss_mem'])

                    STOP_FLAG = self.early_stop_check(train_stats)
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
                    print("mem_iter", memiter,"mem_batchsize",adaptive_batch_num,"aug",self.N)
        self.after_train()
        return test_acc_list