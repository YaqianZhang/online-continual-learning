import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from utils.utils import cutmix_data
import numpy as np




class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.softmax_opt =  torch.optim.SGD(self.model.linear.parameters(),
                                lr=0.1,
                                )
    def _compute_softmax_logits(self,x,need_grad = True):
        if(need_grad == False):
            with torch.no_grad():
                logits = self.model.forward(x)

        else:
            logits = self.model.forward(x)
        return logits



    def compute_testmem_loss(self, ):

        if (self.memory_manager.test_buffer.current_index == 0):
            print("Test memory is empty")
            return None

        test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
        # loss,_ = self.perform_scr(test_batch_x, test_batch_y)
        logits = self._compute_softmax_logits(test_batch_x, need_grad=False)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, test_batch_y)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == test_batch_y).sum().item() / test_batch_x.shape[0]
        self.test_acc_mem.append(acc)
        self.test_loss_mem.append(torch.mean(softmax_loss_full).item())
        return acc
    def cutmix_softmax_training(self,x,y,mem_num):
        if(self.params.use_softmaxloss == False):

            return
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        total_num = x.shape[0]

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
            logits = self.model.forward(x)
            # h_feature = self.model.features(x)
            # logits = self.softmax_head(h_feature)
            softmax_loss = lam * ce(logits, labels_a) + (1 - lam) * ce(
                logits, labels_b
            )
        # else:
        #     logits = self.model.forward(x)
        #
        #     ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        #     softmax_loss_full = ce_all(logits, y)
        #
        #
        #     softmax_loss = (self.params.mem_ratio*torch.sum(softmax_loss_full[:mem_num]) +\
        #             self.params.incoming_ratio * torch.sum(softmax_loss_full[mem_num:]))/total_num
        #     # if (self.params.use_test_buffer and self.memory_manager.buffer.current_index > 0):
        #     #     self.compute_testmem_loss()

            self.softmax_opt.zero_grad()
            softmax_loss.backward()
            self.softmax_opt.step()

    def _batch_update(self,batch_x,batch_y,losses_batch,acc_batch,i):
        batch_x = maybe_cuda(batch_x, self.cuda)
        batch_y = maybe_cuda(batch_y, self.cuda)
        for j in range(self.mem_iters):
            mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y, self.task_seen)

            #mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            if mem_x.size(0) > 0:
                mem_x = maybe_cuda(mem_x, self.cuda)
                mem_y = maybe_cuda(mem_y, self.cuda)
                # print(batch_x.shape)
                # assert False
                batch_x = torch.cat([mem_x,batch_x,])
                batch_y = torch.cat([mem_y,batch_y,])
                ce = torch.nn.CrossEntropyLoss(reduction='mean')

                # do_cutmix = self.params.do_cutmix and np.random.rand(1) < 0.5
                # if do_cutmix:
                #     # print(x.shape)
                #
                #     x, labels_a, labels_b, lam = cutmix_data(x=batch_x, y=batch_y, alpha=1.0,)
                #
                #     logits = self.model.forward(x)
                #     loss = lam * ce(logits, labels_a) + (1 - lam) * ce(
                #         logits, labels_b
                #     )
                # else:
                logits = self.model.forward(batch_x)
                #correct_cnt, pred = self.compute_acc(logits, batch_y)
                _, pred_label = torch.max(logits, 1)
                acc = (pred_label == batch_y)

                mem_num = mem_x.shape[0]
                total_num = batch_x.shape[0]
                avrg_acc = acc.sum().item() / total_num

                acc_mem = acc[:mem_num].sum().item() / mem_num
                acc_incoming = acc[mem_num:].sum().item() / (total_num - mem_num)
                self.train_acc_mem.append(acc_mem)
                self.train_acc_incoming.append(acc_incoming)

                ce_all = torch.nn.CrossEntropyLoss(reduction='none')
                softmax_loss_full = ce_all(logits, batch_y)
                mem_loss = torch.mean(softmax_loss_full[:mem_num])
                incoming_loss = torch.mean(softmax_loss_full[mem_num:])

                self.train_loss_mem.append(mem_loss.item())
                self.train_loss_incoming.append(incoming_loss.item())
                loss = self.params.mem_ratio * mem_loss+ \
                       self.params.incoming_ratio * incoming_loss
                acc_batch.update(avrg_acc, batch_y.size(0))
                losses_batch.update(loss.item(), batch_y.size(0))

                self.opt.zero_grad()
                loss.backward()
                if(i%self.params.online_hyper_freq == 0 and self.params.online_hyper_tune):
                    final_lr = self.hyperparameter_tune()
                   # final_lr = 0.01
                    self.params.learning_rate = final_lr
                    #self.adaptive_learning_rate.append(final_lr)

                    for g in self.opt.param_groups:
                        g['lr'] = final_lr
                    #print("set final lr",final_lr)
                self.opt.step()
                #print(self.opt.param_groups[0]['lr'])
                self.loss_batch.append(loss.item())

                self.cutmix_softmax_training(batch_x,batch_y,mem_num)
                # if (self.params.use_test_buffer and self.memory_manager.buffer.current_index > 0):
                #     acc=self.compute_testmem_loss()
                #     print("True test mem acc",acc)


    def _batch_update_mmorg(self,batch_x,batch_y,):
        batch_x = maybe_cuda(batch_x, self.cuda)
        batch_y = maybe_cuda(batch_y, self.cuda)

        for j in range(self.mem_iters):
            logits = self.model.forward(batch_x)
            loss = self.criterion(logits, batch_y)
            self.opt.zero_grad()
            loss.backward()

            mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            if mem_x.size(0) > 0:
                mem_x = maybe_cuda(mem_x, self.cuda)
                mem_y = maybe_cuda(mem_y, self.cuda)
                # print(batch_x.shape)
                # assert False
                # batch_x = torch.cat([batch_x,mem_x])
                # batch_y = torch.cat([batch_y,mem_y])
                logits = self.model.forward(mem_x)
                loss_mem = self.criterion(logits, mem_y)
                #self.opt.zero_grad()
                loss_mem.backward()

        self.opt.step()

    def _batch_update_mmorg2(self,batch_x,batch_y,):
        batch_x = maybe_cuda(batch_x, self.cuda)
        batch_y = maybe_cuda(batch_y, self.cuda)

        for j in range(self.mem_iters):
            logits = self.model.forward(batch_x)
            loss = self.criterion(logits, batch_y)


            mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            if mem_x.size(0) > 0:
                mem_x = maybe_cuda(mem_x, self.cuda)
                mem_y = maybe_cuda(mem_y, self.cuda)
                # print(batch_x.shape)
                # assert False
                # batch_x = torch.cat([batch_x,mem_x])
                # batch_y = torch.cat([batch_y,mem_y])
                logits = self.model.forward(mem_x)
                loss_mem = self.criterion(logits, mem_y)
                loss_total = loss + loss_mem
                #self.opt.zero_grad()

                self.opt.zero_grad()
                loss_total.backward()
                self.opt.step()
                self.loss_batch.append(loss.item())
            else:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.loss_batch.append(loss.item())



    def _batch_update_org(self,batch_x,batch_y,acc_batch,losses_batch,losses_mem,acc_mem):

        batch_x = maybe_cuda(batch_x, self.cuda)
        batch_y = maybe_cuda(batch_y, self.cuda)
        for j in range(self.mem_iters):
            logits = self.model.forward(batch_x)
            loss = self.criterion(logits, batch_y)
            if self.params.trick['kd_trick']:
                loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                           self.kd_manager.get_kd_loss(logits, batch_x)
            if self.params.trick['kd_trick_star']:
                loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                       (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
            _, pred_label = torch.max(logits, 1)
            correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
            # update tracker
            acc_batch.update(correct_cnt, batch_y.size(0))
            losses_batch.update(loss, batch_y.size(0))
            # backward
            self.opt.zero_grad()
            loss.backward()


            # self.opt.step()
            # if (self.params.frozen_old_fc):
            #     self.model.linear.weight.data[self.old_labels, :] = weight_org[self.old_labels, :]
            #


            # mem update
            mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            if mem_x.size(0) > 0:
                mem_x = maybe_cuda(mem_x, self.cuda)
                mem_y = maybe_cuda(mem_y, self.cuda)
                mem_logits = self.model.forward(mem_x)

                loss_mem = self.criterion(mem_logits, mem_y)
                if self.params.trick['kd_trick']:
                    loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                               self.kd_manager.get_kd_loss(mem_logits, mem_x)
                if self.params.trick['kd_trick_star']:
                    loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                           (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
                                                                                                 mem_x)
                # update tracker
                losses_mem.update(loss_mem, mem_y.size(0))
                _, pred_label = torch.max(mem_logits, 1)
                correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                acc_mem.update(correct_cnt, mem_y.size(0))

                #self.opt.zero_grad()

                loss_mem.backward()

            if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
                # opt update
                self.opt.zero_grad()
                combined_batch = torch.cat((mem_x, batch_x))
                combined_labels = torch.cat((mem_y, batch_y))
                combined_logits = self.model.forward(combined_batch)
                loss_combined = self.criterion(combined_logits, combined_labels)
                loss_combined.backward()
                self.opt.step()
            else:
                #weight_org = self.model.linear.weight.clone() # 100*160
                #print(self.model.linear.weight)
                self.opt.step()

                #print(self.model.linear.weight)

                # fc_para = self.model.linear.parameters()
                # if(self.params.frozen_old_fc):
                #
                #     self.model.linear.weight.data[self.old_labels,:] = weight_org[self.old_labels,:]
                # #print(self.model.linear.weight)
                #assert False

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
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
                batch_x, batch_y = batch_data
                batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)

                if(self.params.save_prefix == "joint_training"):
                    self._batch_update(batch_x,batch_y,losses_batch,acc_batch,i)
                elif(self.params.save_prefix == "joint_training_mmorg"):
                    self._batch_update_mmorg(batch_x, batch_y)
                elif(self.params.save_prefix == "joint_training_mmorg2"):
                    self._batch_update_mmorg2(batch_x, batch_y)
                else:
                    self._batch_update_org(batch_x,batch_y,acc_batch,losses_batch,losses_mem,acc_mem)
                #self.buffer.update(batch_x, batch_y,acc_batch)
                self.memory_manager.update_memory(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
                    print("lr",self.params.learning_rate)
        self.after_train()

    # def virtual_train(self, model,x_train, y_train):
    #     #self.before_train(x_train, y_train)
    #     # set up loader
    #     train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
    #     train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
    #                                    drop_last=True)
    #     # set up model
    #     #self.model = self.model.train()
    #
    #     # setup tracker
    #     losses_batch = AverageMeter()
    #     #losses_mem = AverageMeter()
    #     acc_batch = AverageMeter()
    #     # acc_mem = AverageMeter()
    #
    #     for ep in range(self.epoch):
    #         for i, batch_data in enumerate(train_loader):
    #             # batch update
    #             batch_x, batch_y = batch_data
    #            # batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)
    #
    #             self._batch_update(batch_x,batch_y,losses_batch,acc_batch)
    #

        #self.after_train()

    # def hyperparameter_tune_task_level(self,x_train, y_train):
    #     lr_list = [0.01, 0.1, 0.2]
    #     if(self.params.save_prefix_tmp == "rnd"):
    #         acc_list=[1,1,1]
    #     else:
    #         ## try different lf
    #
    #
    #         for lr in lr_list:#[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
    #             acc, loss = self.virtual_train( model, x_train, y_train)
    #             if (acc > max_acc):
    #                 best_para = lr
    #                 max_acc = acc
    #
    #             acc_list.append(acc.item())


            #acc_list =[1,1,1]

        #
        #
        #
        # prob = np.exp(acc_list) / sum(np.exp(acc_list))
        #
        # selected_para = np.random.choice(lr_list, 1,p=prob)[0]
        #
        #
        # self.params.learning_rate = selected_para
        # #print("!! best lr",best_para,"selected_lr",selected_para,)#acc_list,)
        # final_lr = selected_para
        # self.adaptive_learning_rate.append(final_lr)
        # return final_lr