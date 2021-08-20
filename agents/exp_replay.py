import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter


class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
    def _batch_update(self,batch_x,batch_y,):
        batch_x = maybe_cuda(batch_x, self.cuda)
        batch_y = maybe_cuda(batch_y, self.cuda)
        for j in range(self.mem_iters):

            mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            if mem_x.size(0) > 0:
                mem_x = maybe_cuda(mem_x, self.cuda)
                mem_y = maybe_cuda(mem_y, self.cuda)
                # print(batch_x.shape)
                # assert False
                batch_x = torch.cat([batch_x,mem_x])
                batch_y = torch.cat([batch_y,mem_y])
            logits = self.model.forward(batch_x)
            loss = self.criterion(logits, batch_y,reduction_type="none")
            if(loss.shape[0] == 20):
                #print("True")
                loss = self.params.incoming_ratio * torch.mean(loss[:10])+ self.params.mem_ratio * torch.mean(loss[10:])
            else:
                #print(loss.shape[0])
                loss = torch.mean(loss)


            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

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
            else:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()



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
                if(self.params.save_prefix == "joint_training"):
                    self._batch_update(batch_x,batch_y)
                elif(self.params.save_prefix == "joint_training_mmorg"):
                    self._batch_update_mmorg(batch_x, batch_y)
                elif(self.params.save_prefix == "joint_training_mmorg2"):
                    self._batch_update_mmorg2(batch_x, batch_y)
                else:
                    self._batch_update_org(batch_x,batch_y,acc_batch,losses_batch,losses_mem,acc_mem)
                self.buffer.update(batch_x, batch_y,acc_batch)

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
        self.after_train()