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

class SCR_der(SupContrastReplay):
    def __init__(self, model, opt, params):
        super(SCR_der, self).__init__(model, opt, params)
        self.self_sup_beta = params.self_sup_beta
        self.buffer = self.memory_manager.buffer
        self.alpha = params.DER_alpha #0.3
        print(self.alpha)
        buffer_size = self.params.mem_size
        self.buffer.buffer_logits=torch.FloatTensor(buffer_size, 128).fill_(0)


    def aug_data(self,batch_x,):
        if (self.task_seen >= self.params.aug_start):
            if (self.params.randaug):
                # print(concat_batch_x[0])

                # batch_x_aug1 = self.aug_agent.aug_data(batch_x,mem_x.size(0),)
                # end1=time.time()

                batch_x_aug2 = self.aug_agent.aug_data_old_batch(batch_x )
                batch_x = batch_x_aug2
            elif (self.params.deraug):
                self.aug_agent.set_deraug()
                batch_x_aug2 = self.aug_agent.aug_data_old_batch(batch_x)
                batch_x = batch_x_aug2
            # elif (self.params.scraug):
            #     batch_x = self.aug_agent.scr_aug_data(batch_x)
            else:
                pass

            if (self.params.aug_normal):
                # print(batch_x)
                # assert False

                batch_x = self.transform_normal(batch_x)
            return batch_x



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

        aug_features = self.model.forward(combined_batch)





        #print(loss,features.shape)

        self.loss_batch.append(loss.item())




        return loss





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
                    scr_loss = self.perform_scr_update(concat_batch_x, concat_batch_y,mem_num)
                    acc_mem,softmax_loss = self.perform_softmax_update(concat_batch_x, concat_batch_y,mem_num)
                    # e2 = time.time()
                    # print("aug",e1-s,"scr update",e2-e1)


                    self.opt.zero_grad()
                    scr_loss.backward()

                    mem_x, mem_y,mem_logits = self.memory_manager.retrieve_from_mem_logits(batch_x, batch_y,
                                                                                           retrieve_num=10)

                    aug_batch_x = self.aug_data(batch_x)
                    aug_inc_logits = self.model.forward(aug_batch_x)

                    if mem_x.shape[0]>0 and self.task_seen>0:

                        aug_mem_x = self.aug_data(mem_x)
                        aug_mem_logits = self.model.forward(aug_mem_x)
                        mse_loss = torch.nn.MSELoss()
                        mem_reg_loss = self.alpha * mse_loss(aug_mem_logits,mem_logits)
                        mem_reg_loss.backward()
                        #self.train_loss_mem.append(mem_reg_loss.item())
                    else:
                        mem_reg_loss=torch.zeros(1)
                    self.opt.step()

                # update mem
                self.memory_manager.update_memory_logits(batch_x, batch_y,aug_inc_logits.data)
                if i % 100 == 1 and self.verbose:
                    # print(
                    #     '==>>> it: {}, avg. loss: {:.6f},'
                    #         .format(i, scr_loss,)
                    # )
                    print("==>> it: {:},scr loss: {:.2f}, der loss: {:.2f}".format(i,scr_loss.item(),mem_reg_loss.item(),acc_mem))


        self.after_train()



