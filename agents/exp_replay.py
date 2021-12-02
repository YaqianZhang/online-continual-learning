import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from utils.utils import cutmix_data,cutmix_data_two_data
import numpy as np
from RL.RL_replay_base import RL_replay
from RL.close_loop_cl import close_loop_cl
from torchvision.transforms import transforms
from utils.augmentations import RandAugment

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from utils.setup_elements import transforms_match, input_size_match






class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        #self.buffer = Buffer(model, params)
        self.buffer = self.memory_manager.buffer
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.softmax_opt =  torch.optim.SGD(self.model.linear.parameters(),
                                lr=0.1,
                                )
        #if(self.params.online_hyper_RL):



        if(self.params.use_test_buffer):
            self.close_loop_cl = close_loop_cl(self,model,self.memory_manager)
            self.mix_label_pair = None
            self.low_acc_classes = None
            self.col=None
            self.row=None
            self.RL_replay = RL_replay(params, self.close_loop_cl)
        else:
            self.close_loop_cl = None
        self.replay_para={"mem_ratio":self.params.mem_ratio,
                          "incoming_ratio":self.params.incoming_ratio,
                          "mem_iter":self.params.mem_iters,
                          "randaug_M":self.params.randaug_M}


        # _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #
        # self.transform_train = transforms.Compose([
        #     # transforms.RandomCrop(32, padding=4),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     #transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # ])
        # # transform_test = transforms.Compose([
        # #     transforms.ToTensor(),
        # #     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # # ])
        #
        # self.transform_train.transforms.insert(0, RandAugment(self.params.randaug_N, self.params.randaug_M))
        #
        # self.scr_transform = nn.Sequential(
        #     RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
        #                       scale=(0.2, 1.)),
        #     RandomHorizontalFlip(),
        #     ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        #     RandomGrayscale(p=0.2)
        #
        # )
    # def set_aug_para(self,N,M):
    #     self.transform_train = transforms.Compose([
    #         # transforms.RandomCrop(32, padding=4),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    #     ])
    #     # transform_test = transforms.Compose([
    #     #     transforms.ToTensor(),
    #     #     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    #     # ])
    #
    #     self.transform_train.transforms.insert(0, RandAugment(N,M))

    def _compute_acc(self, batch_x, batch_y,  mem_num=0):


        logits = self.model.forward(batch_x)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == batch_y)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, batch_y)

        total_num = batch_x.shape[0]
        avrg_acc = acc.sum().item() / total_num
        # loss = torch.mean(softmax_loss_full)

        acc_incoming = acc[mem_num:].sum().item() / (total_num - mem_num)

        incoming_loss = torch.mean(softmax_loss_full[mem_num:])
        self.train_acc_incoming_org.append(acc_incoming)
        self.train_loss_incoming_org.append(incoming_loss.item())
        if(mem_num>0):

            acc_mem = acc[:mem_num].sum().item() / mem_num
            mem_loss = torch.mean(softmax_loss_full[:mem_num])
            self.train_acc_mem_org.append(acc_mem)
            self.train_loss_mem_org.append(mem_loss.item())
            #print(acc_incoming,acc_mem)


    def _batch_update(self,batch_x,batch_y,losses_batch,acc_batch,i,replay_para=None,mem_num=0):


        if(replay_para == None):
            replay_para = self.replay_para


        # if(self.params.test_mem_recycle):
        #     recycle_test_x = recycle.store_tmp(img,cls_max)





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
            if(self.close_loop_cl != None):

                self.close_loop_cl.last_train_loss = mem_loss.item()


            loss = replay_para['mem_ratio'] * mem_loss+ \
                   replay_para['incoming_ratio'] * incoming_loss
            train_stats = {'acc_incoming': acc_incoming,
                           'acc_mem': acc_mem,
                           "loss_incoming": incoming_loss.item(),
                           "loss_mem": mem_loss.item(),
                           "batch_num": i,
                           }
        else:
            loss = replay_para['incoming_ratio'] * incoming_loss
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

    # def tensor_to_image(self,tensor):
    #     tensor = tensor * 255
    #     tensor = np.array(tensor, dtype=np.uint8)
    #     if np.ndim(tensor) > 3:
    #         assert tensor.shape[0] == 1
    #         tensor = tensor[0]
    #     return PIL.Image.fromarray(tensor)

    # def aug_data(self,concat_batch_x):
    #     n, c, w, h = concat_batch_x.shape
    #
    #     images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(n)]
    #
    #     concat_batch_x = [self.transform_train(image).reshape([1, c, w, h]) for image in images]
    #
    #     concat_batch_x = maybe_cuda(torch.cat(concat_batch_x, dim=0))
    #     return concat_batch_x
    # def scr_aug_data(self,combined_batch):
    #     return self.scr_transform(combined_batch)


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

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.set_memIter()

                for j in range(self.mem_iters):
                    #self.set_aug_para(N, int(j*30/self.mem_iters), incoming_M=int(j*30/self.mem_iters))


                    concat_batch_x,concat_batch_y,mem_num = self.concat_memory_batch(batch_x,batch_y)




                    self._batch_update(concat_batch_x,concat_batch_y, losses_batch, acc_batch, i,mem_num=mem_num)

                if(self.params.use_test_buffer):
                    self.close_loop_cl.compute_testmem_loss()
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
                    print("mem_iter", self.mem_iters)
        self.after_train()


        ################# joint_replay_type

        # if (self.params.joint_replay_type == "together"):
        #     self._batch_update(batch_x, batch_y, losses_batch, acc_batch, i)
        # elif (self.params.joint_replay_type == "seperate"):
        #     self._batch_update_org(batch_x, batch_y, acc_batch, losses_batch, losses_mem, acc_mem)
        # else:
        #     raise NotImplementedError("undefined joint training implementation type",
        #                               self.params.joint_replay_type)

    # def _batch_update_org(self,batch_x,batch_y,acc_batch,losses_batch,losses_mem,acc_mem):
    #
    #
    #     for j in range(self.mem_iters):
    #         logits = self.model.forward(batch_x)
    #         loss = self.criterion(logits, batch_y)
    #         if self.params.trick['kd_trick']:
    #             loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
    #                        self.kd_manager.get_kd_loss(logits, batch_x)
    #         if self.params.trick['kd_trick_star']:
    #             loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
    #                    (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
    #         _, pred_label = torch.max(logits, 1)
    #         correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
    #         # update tracker
    #         acc_batch.update(correct_cnt, batch_y.size(0))
    #         losses_batch.update(loss, batch_y.size(0))
    #         # backward
    #         self.opt.zero_grad()
    #         loss.backward()
    #
    #
    #         # self.opt.step()
    #         # if (self.params.frozen_old_fc):
    #         #     self.model.linear.weight.data[self.old_labels, :] = weight_org[self.old_labels, :]
    #         #
    #
    #
    #         # mem update
    #         mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
    #         if mem_x.size(0) > 0:
    #             mem_x = maybe_cuda(mem_x, self.cuda)
    #             mem_y = maybe_cuda(mem_y, self.cuda)
    #             mem_logits = self.model.forward(mem_x)
    #
    #             loss_mem = self.criterion(mem_logits, mem_y)
    #             if self.params.trick['kd_trick']:
    #                 loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
    #                            self.kd_manager.get_kd_loss(mem_logits, mem_x)
    #             if self.params.trick['kd_trick_star']:
    #                 loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
    #                        (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
    #                                                                                              mem_x)
    #             # update tracker
    #             losses_mem.update(loss_mem, mem_y.size(0))
    #             _, pred_label = torch.max(mem_logits, 1)
    #             correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
    #             acc_mem.update(correct_cnt, mem_y.size(0))
    #
    #             #self.opt.zero_grad()
    #
    #             loss_mem.backward()
    #
    #         if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
    #             # opt update
    #             self.opt.zero_grad()
    #             combined_batch = torch.cat((mem_x, batch_x))
    #             combined_labels = torch.cat((mem_y, batch_y))
    #             combined_logits = self.model.forward(combined_batch)
    #             loss_combined = self.criterion(combined_logits, combined_labels)
    #             loss_combined.backward()
    #             self.opt.step()
    #         else:
    #
    #             self.opt.step()



