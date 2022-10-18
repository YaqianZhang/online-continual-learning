import time
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
from torchvision.transforms import transforms
from utils.augmentations import RandAugment
from agents.scr import SupContrastReplay

class SCR_spread_class(SupContrastReplay):
    def __init__(self, model, opt, params):
        super(SCR_spread_class, self).__init__(model, opt, params)
        self.self_sup_beta = params.self_sup_beta





    def perform_scr_update(self,combined_batch, combined_labels,mem_num):

        # print(self.model)
        # assert False
        ######## scr loss

        if (self.params.scrview=="None"):
            combined_batch_aug = combined_batch.clone()

        elif (self.params.scrview=="randaug"):
            print("randaug")


            combined_batch_aug = self.aug_agent.aug_data_old(combined_batch, mem_num )

        else:
            #print("SCR")

            combined_batch_aug = self.transform(combined_batch)

        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1),
                              self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
        loss, loss_full = self.criterion(features, combined_labels)

        unique_labels = torch.unique(combined_labels)
        self_sup_loss_sum_value = 0
        for label in unique_labels:
            idx = combined_labels==label

            same_class_num = torch.sum(idx)

            if same_class_num >1:


                fake_labels = torch.arange(0,same_class_num)
                self_sup_loss,_ =self.criterion(features[idx], fake_labels)


                loss += self.self_sup_beta * self_sup_loss
                self_sup_loss_sum_value = self_sup_loss.item()




        #print(loss,features.shape)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_batch.append(loss.item())




        return loss.item(),self_sup_loss_sum_value






    def train_learner(self, x_train, y_train):


        #self.memory_manager.reset_new_old()
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()


        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                start_time = time.time()
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)

                self.set_memIter()
                for j in range(self.mem_iters):
                    #s = time.time()
                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)
                   # e1 = time.time()
                    scr_loss,self_sup_loss = self.perform_scr_update(concat_batch_x, concat_batch_y,mem_num)
                    acc_mem,softmax_loss = self.perform_softmax_update(concat_batch_x, concat_batch_y,mem_num)
                    # e2 = time.time()
                    # print("aug",e1-s,"scr update",e2-e1)


                # update mem
                self.memory_manager.update_memory(batch_x, batch_y)
                time_per_batch = time.time() - start_time
                if i % 100 == 1 and self.verbose:
                    # print(
                    #     '==>>> it: {}, avg. loss: {:.6f},'
                    #         .format(i, scr_loss,)
                    # )
                    print("==>> it: {:},scr loss: {:.2f}, self_sup loss: {:.2f}, time:{:.2f}"
                          .format(i,scr_loss,self_sup_loss,acc_mem,time_per_batch))


        self.after_train()



