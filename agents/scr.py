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
from models.learner import Learner

class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )
        if(self.params.aug_type == "two"):
            self.transform = nn.Sequential(
                RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
                                  scale=(0.2, 1.)),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8)


            )
        if(params.data in [ 'clrs25', 'core50']):
            softmax_inputdim = 2560
        elif(params.data in ['cifar100','cifar10']):
            softmax_inputdim = 160
        else:
            raise NotImplementedError("undefined dataset",params.data)
        print(softmax_inputdim)

        if(params.softmax_type == "seperate"):
            self.seperate_softmax(softmax_inputdim)
        # elif(params.softmax_type == "meta"):
        #     self.meta_softmax(softmax_inputdim)
    def seperate_softmax(self,softmax_inputdim):
        self.softmax_head = maybe_cuda(build_mlp(input_size=softmax_inputdim,
            output_size=100,
            n_layers=self.params.softmax_nlayers,
            size=self.params.softmax_nsize,
            use_dropout=self.params.softmax_dropout,))
        self.softmax_opt =  torch.optim.SGD(self.softmax_head .parameters(),
                                lr=self.params.softmaxhead_lr,
                                )



    def compute_testmem_loss(self, ):

        if (self.memory_manager.test_buffer.current_index == 0):
            print("Test memory is empty")
            return None

        test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
        #loss,_ = self.perform_scr(test_batch_x, test_batch_y)
        logits = self._compute_softmax_logits(test_batch_x,need_grad = False)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, test_batch_y)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == test_batch_y).sum().item()/test_batch_x.shape[0]
        self.test_acc_mem.append(acc)
        self.test_loss_mem.append(torch.mean(softmax_loss_full).item())
        n = len(test_batch_y)

        idx_old = [test_batch_y[i] in self.old_labels for i in range(n)]
        idx_new = [test_batch_y[i] in self.new_labels for i in range(n)]
        correctness = pred_label == test_batch_y
        pred_old_new = [pred_label[i] in self.new_labels for i in range(n)]
        true_old_new = [test_batch_y[i] in self.new_labels for i in range(n)]
        old_acc = correctness[idx_old].sum().item()/np.sum(idx_old)
        new_acc = correctness[idx_new].sum().item()/np.sum(idx_new)

        correctness_old_new = np.array(pred_old_new)==np.array(true_old_new)
        new_old_classification = np.mean(correctness_old_new)
        print(old_acc,new_acc, new_old_classification)


        #print("test acc/loss",acc,self.test_loss_mem[-1])
    #
    # def examine_pure_train_loss(self,mem_x, mem_y,batch_x, batch_y):
    #     mem_loss,_ = self.perform_scr( mem_x, mem_y)
    #     incoming_loss,_ =self.perform_scr(batch_x, batch_y)
    #     self.train_loss_mem.append(mem_loss.item())
    #     self.train_loss_incoming.append(incoming_loss.item())
    #     return mem_loss.item(),incoming_loss.item()
    #
    #
    #
    # def perform_scr(self,x,y):
    #     x_aug = self.transform(x)
    #     features = torch.cat([self.model.forward(x).unsqueeze(1),
    #                           self.model.forward(x_aug).unsqueeze(1)], dim=1)
    #     loss, loss_full = self.criterion(features, y)
    #     return loss,loss_full



   ## override base function
    def compute_nmc_mean(self):
        exemplar_means = {}
        cls_exemplar = {cls: [] for cls in self.old_labels}
        # buffer_filled = self.buffer.current_index
        # for x, y in zip(self.buffer.buffer_img[:buffer_filled],
        #                 self.buffer.buffer_label[:buffer_filled]):
        buffer_filled = self.memory_manager.buffer.current_index
        for x, y in zip(self.memory_manager.buffer.buffer_img[:buffer_filled],
                        self.memory_manager.buffer.buffer_label[:buffer_filled]):
            x = maybe_cuda(x)
            y = maybe_cuda(y)
            cls_exemplar[y.item()].append(x)
        for cls, exemplar in cls_exemplar.items():
            features = []
            # Extract feature for each exemplar in p_y
            for ex in exemplar:
                feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()  # Normalize
                features.append(feature)
            if len(features) == 0:
                mu_y = maybe_cuda(
                    torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                mu_y = mu_y.squeeze()
            else:
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
            exemplar_means[cls] = mu_y
        return exemplar_means
    def _compute_softmax_logits(self,x,need_grad = True):
        if(need_grad == False):
            with torch.no_grad():
                h_feature = self.model.features(x)
                logits = self.softmax_head(h_feature)
        else:
            h_feature = self.model.features(x)
            logits = self.softmax_head(h_feature)
        return logits

    def perform_softmax_training(self,x,y,mem_num):
        if(self.params.softmax_type == "None"):

            return
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        total_num = x.shape[0]
        # incoming_num = x.shape[0]-mem_num
        #
        # idx_mem = np.random.randint(0,mem_num,incoming_num)
        # idx_incoming = np.random.randint(mem_num,total_num,mem_num)
        # index = torch.from_numpy(np.concatenate((idx_incoming , idx_mem)))
        # # print(idx_incoming,idx_mem,index)
        # # assert False
        #

        index="None"
        if(self.params.softmax_membatch < mem_num):
            selected_idx = np.random.shuffle(np.range(0,mem_num))[:self.params.softmax_membatch]

            #selected_idx = np.random.randint(0,mem_num,self.params.softmax_membatch)
            selected_idx = list(selected_idx)+list(np.arange(mem_num,total_num))
            # print(mem_num,selected_idx)
            # assert False
            x = x[selected_idx]
            y = y[selected_idx]
            mem_num =self.params.softmax_membatch



        ## todo : cutmix
        do_cutmix = self.params.do_cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            # print(x.shape)

            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0,index=index)
            logits = self._compute_softmax_logits(x)
            # h_feature = self.model.features(x)
            # logits = self.softmax_head(h_feature)
            softmax_loss = lam * ce(logits, labels_a) + (1 - lam) * ce(
                logits, labels_b
            )
        else:
            softmax_loss = 0
        logits = self._compute_softmax_logits(x)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, y)

        mem_loss = torch.mean(softmax_loss_full[:mem_num])
        incoming_loss = torch.mean(softmax_loss_full[mem_num:])
        self.train_loss_mem.append(mem_loss.item())
        self.train_loss_incoming.append(incoming_loss.item())
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == y)
        acc_mem =  acc[:mem_num].sum().item() /mem_num
        acc_incoming = acc[mem_num:].sum().item() / (total_num -mem_num)
        self.train_acc_mem.append(acc_mem)
        self.train_acc_incoming.append(acc_incoming)
        #softmax_loss = torch.mean(softmax_loss_full)
        # print(self.params.mem_ratio)
        # assert False
        softmax_loss += (self.params.mem_ratio*torch.sum(softmax_loss_full[:mem_num]) +\
                self.params.incoming_ratio * torch.sum(softmax_loss_full[mem_num:]))/total_num
        if (self.params.use_test_buffer and self.memory_manager.buffer.current_index > 0):
            self.compute_testmem_loss()

        self.softmax_opt.zero_grad()
        softmax_loss.backward()
        self.softmax_opt.step()



    def train_learner(self, x_train, y_train):
        self.memory_manager.reset_new_old()
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

        # if (self.params.use_test_buffer and self.memory_manager.test_buffer.current_index > 0):
        #     exemplar_means = self.compute_nmc_mean()
        # else:
        #     exemplar_means = None

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)

                for j in range(self.mem_iters):
                    #mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y, self.task_seen)

                    if mem_x.size(0) > 0:


                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))

                        ######## scr loss
                        if(self.params.no_aug):
                            combined_batch_aug = combined_batch.clone()
                        else:


                            combined_batch_aug = self.transform(combined_batch)

                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1),
                                              self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss, loss_full = self.criterion(features, combined_labels)
                        # do_cutmix = self.params.do_cutmix #and np.random.rand(1) < 0.5
                        # if do_cutmix and self.params.use_softmaxloss == False:
                        #     ## perform cutmix for SCR loss
                        #     cutmix_batch = combined_batch.clone() ### to prevent the in place change to x
                        #     combined_batch_cutmix, labels_a, labels_b, lam = cutmix_data(x=cutmix_batch, y=combined_labels, alpha=1.0)
                        #
                        #     features = torch.cat([self.model.forward(combined_batch).unsqueeze(1),
                        #                           self.model.forward(combined_batch_cutmix).unsqueeze(1)], dim=1)
                        #     loss_a,_ = self.criterion(features, labels_a)
                        #     loss_b,_ = self.criterion(features, labels_b)
                        #     loss_cutmix = lam *  loss_a + (1 - lam) * loss_b
                        #
                        #     loss = loss+loss_cutmix



                        # mem_loss = (loss_full[:10].sum() + loss_full[20:30].sum())/20
                        # incoming_loss = (loss_full[10:20].sum() + loss_full[30:40].sum()) / 20
                        # self.train_loss_mem.append(mem_loss.item())
                        # self.train_loss_incoming.append(incoming_loss.item())
                        #
                        # if(self.params.examine_train):
                        #     mem_loss,incoming_loss = self.examine_pure_train_loss(mem_x,mem_y,batch_x,batch_y)
                        #     #print("mem",mem_loss,"incoming",incoming_loss,"all",loss.item())

                        losses.update(loss, batch_y.size(0))
                        self.opt.zero_grad()
                        loss.backward()
                        #loss_cutmix.backward()
                        if (i % self.params.online_hyper_freq == 0 and self.params.online_hyper_tune):
                            final_lr = np.random.choice([0.001,0.01,0.1],1)[0]
                            print("!!! lr",final_lr)
                            for g in self.opt.param_groups:
                                g['lr'] = final_lr
                        self.opt.step()
                        self.loss_batch.append(loss.item())

                        #### softmax loss
                        if(self.params.save_prefix == "aug2"):

                            aug_labels = torch.cat([combined_labels,combined_labels])
                            aug_batch   = torch.cat([combined_batch,combined_batch_aug])
                            self.perform_softmax_training(aug_batch, aug_labels,mem_x.shape[0])
                        else:
                            self.perform_softmax_training(combined_batch, combined_labels, mem_x.shape[0])


                # update mem
                #self.buffer.update(batch_x, batch_y)
                self.memory_manager.update_memory(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                            .format(i, losses.avg(), acc_batch.avg())
                    )



        self.after_train()
        # if (self.params.use_test_buffer and self.memory_manager.buffer.current_index > 0):
        #     exemplar_means = self.compute_nmc_mean()
        # else:
        #     exemplar_means = None
        # if (exemplar_means != None):
        #     acc_mem, pred_label, loss_batch = self.nmc_predict(mem_x, mem_y,
        #                                                        exemplar_means)
        #     self.train_acc_mem.append(acc_mem)
        #     acc_incoming, pred_label, loss_batch = self.nmc_predict(batch_x, batch_y,
        #                                                             exemplar_means)
        #     self.train_acc_incoming.append(acc_incoming)
        #
        # if (self.params.use_test_buffer and exemplar_means != None):
        #     test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
        #
        #     acc_test, pred_label, loss_batch = self.nmc_predict(test_batch_x, test_batch_y, exemplar_means)
        #     print(acc_test)
        #     assert False
        #     self.test_acc_mem.append(acc_test)
        #     self.test_loss_mem.append(loss_batch)
        # if (exemplar_means != None):
        #     print("acc mem", acc_mem, "acc_incoming", acc_incoming,
        #           "acc_test", acc_test)
