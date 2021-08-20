import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from torch import nn
import math
import numpy as np


class LAMAML(ContinualLearner):
    def __init__(self, model, opt, params):
        super(LAMAML, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.define_task_lr_params(alpha_init=params.alpha_init)

    def define_task_lr_params(self, alpha_init=1e-3):
        # Setup learning parameters
        self.alpha_lr = nn.ParameterList([])

        self.lr_name = []
        for n, p in self.model.named_parameters():
            self.lr_name.append(n)

        for p in self.model.parameters():
            self.alpha_lr.append(nn.Parameter(alpha_init * torch.ones(p.shape, requires_grad=True)))
        self.alpha_lr = maybe_cuda(self.alpha_lr)

    def meta_loss(self, x, fast_weights, y, ):
        """
        differentiate the loss through the network updates wrt alpha
        """

        logits = self.model.forward_with_weights(x,fast_weights)


        loss_q = self.criterion(logits, y)



        return loss_q, logits

    def inner_update(self, x, fast_weights, y, ):
        """
        Update the fast weights using the current samples and return the updated fast
        """

        # offset1, offset2 = self.compute_offsets(t)
        #
        # logits = self.net.forward(x, fast_weights)[:, :offset2]
        # loss = self.take_loss(t, logits, y)

        if fast_weights is None:
            logits = self.model.forward(x)
        else:
        #
            logits = self.model.forward_with_weights(x,fast_weights)

        loss = self.criterion(logits, y)

        if fast_weights is None:
            fast_weights = nn.ParameterList()
            for p in self.model.parameters():#parameters()
                fast_weights.append(p)


        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = False #2.0# self.args.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required,
                                         allow_unused=True))

        # for i in range(len(grads)):
        #     grads[i] = torch.clamp(grads[i], min=-1.0, max=1.0)

        print(len(grads),grads[0])




        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.alpha_lr))))


        return fast_weights

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
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):

                    perm = torch.randperm(batch_x.size(0))
                    batch_x = batch_x[perm]
                    batch_y = batch_y[perm]

                    #self.epoch += 1
                    self.opt.zero_grad()

                    # if t != self.current_task:
                    #     self.M = self.M_new.copy()
                    #     self.current_task = t

                    batch_sz = batch_x.shape[0]
                    n_batches = 5 #self.args.cifar_batches
                    rough_sz = math.ceil(batch_sz / n_batches)
                    fast_weights = None
                    meta_losses = [0 for _ in range(n_batches)]

                    # get a batch by augmented incming data with old task data, used for
                    # computing meta-loss

                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    # bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)

                    for i in range(n_batches):

                        batch_batch_x = batch_x[i * rough_sz: (i + 1) * rough_sz]
                        batch_batch_y = batch_y[i * rough_sz: (i + 1) * rough_sz]

                        # assuming labels for inner update are from the same
                        fast_weights = self.inner_update(batch_batch_x, fast_weights, batch_batch_y,)
                        # only sample and push to replay buffer once for each task's stream
                        # instead of pushing every epoch
                        # if (self.real_epoch == 0):
                        #     if (self.args.use_test_mem and t > 0):
                        #         self.update_mem(batch_x, batch_y, torch.tensor(t))
                        #
                        #     else:
                        #
                        #         self.push_to_mem(batch_x, batch_y, torch.tensor(t))
                        #
                        #     # self.push_to_mem(batch_x, batch_y, torch.tensor(t))
                        if(mem_x.shape[0]>0):
                            meta_loss, logits = self.meta_loss(mem_x, fast_weights, mem_y, )

                            meta_losses[i] += meta_loss

                    # Taking the meta gradient step (will update the learning rates)
                    self.zero_grads()

                    meta_loss = sum(meta_losses) / len(meta_losses)
                    meta_loss.backward()

                    # torch.nn.utils.clip_grad_norm_(self.alpha_lr.parameters(), self.args.grad_clip_norm)
                    # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                    if self.args.learn_lr:
                        self.opt_lr.step()

                    # if sync-update is being carried out (as in sync-maml) then update the weights using the optimiser
                    # otherwise update the weights with sgd using updated LRs as step sizes
                    if (self.args.sync_update):
                        self.opt_wt.step()
                    else:
                        for i, p in enumerate(self.model.parameters()):
                            # using relu on updated LRs to avoid negative values
                            p.data = p.data - p.grad * nn.functional.relu(self.alpha_lr[i])
                    self.model.zero_grad()
                    self.alpha_lr.zero_grad()

                return meta_loss.item()

    # def train_learner(self, x_train, y_train):
    #     self.before_train(x_train, y_train)
    #     # set up loader
    #     train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
    #     train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
    #                                    drop_last=True)
    #     # set up model
    #     self.model = self.model.train()
    #
    #     # setup tracker
    #     losses_batch = AverageMeter()
    #     losses_mem = AverageMeter()
    #     acc_batch = AverageMeter()
    #     acc_mem = AverageMeter()
    #
    #     for ep in range(self.epoch):
    #         for i, batch_data in enumerate(train_loader):
    #             # batch update
    #             batch_x, batch_y = batch_data
    #             batch_x = maybe_cuda(batch_x, self.cuda)
    #             batch_y = maybe_cuda(batch_y, self.cuda)
    #             for j in range(self.mem_iters):
    #                 logits = self.model.forward(batch_x)
    #
    #                 loss = self.criterion(logits, batch_y)
    #                 if self.params.trick['kd_trick']:
    #                     loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
    #                                self.kd_manager.get_kd_loss(logits, batch_x)
    #                 if self.params.trick['kd_trick_star']:
    #                     loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
    #                            (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
    #                 _, pred_label = torch.max(logits, 1)
    #                 correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
    #                 # update tracker
    #                 acc_batch.update(correct_cnt, batch_y.size(0))
    #                 losses_batch.update(loss, batch_y.size(0))
    #                 # backward
    #                 self.opt.zero_grad()
    #                 loss.backward()
    #
    #
    #
    #
    #                 # mem update
    #                 mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
    #                 if mem_x.size(0) > 0:
    #                     mem_x = maybe_cuda(mem_x, self.cuda)
    #                     mem_y = maybe_cuda(mem_y, self.cuda)
    #                     mem_logits = self.model.forward(mem_x)
    #
    #                     loss_mem = self.criterion(mem_logits, mem_y)
    #                     if self.params.trick['kd_trick']:
    #                         loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
    #                                    self.kd_manager.get_kd_loss(mem_logits, mem_x)
    #                     if self.params.trick['kd_trick_star']:
    #                         loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
    #                                (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
    #                                                                                                      mem_x)
    #                     # update tracker
    #                     losses_mem.update(loss_mem, mem_y.size(0))
    #                     _, pred_label = torch.max(mem_logits, 1)
    #                     correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
    #                     acc_mem.update(correct_cnt, mem_y.size(0))
    #
    #                     self.opt.zero_grad()
    #
    #                     loss_mem.backward()
    #
    #                 if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
    #                     # opt update
    #                     self.opt.zero_grad()
    #                     combined_batch = torch.cat((mem_x, batch_x))
    #                     combined_labels = torch.cat((mem_y, batch_y))
    #                     combined_logits = self.model.forward(combined_batch)
    #                     loss_combined = self.criterion(combined_logits, combined_labels)
    #                     loss_combined.backward()
    #                     self.opt.step()
    #                 else:
    #                     weight_org = self.model.linear.weight.clone() # 100*160
    #                     #print(self.model.linear.weight)
    #                     self.opt.step()
    #                     #print(self.model.linear.weight)
    #
    #                     # fc_para = self.model.linear.parameters()
    #                     if(self.params.frozen_old_fc):
    #
    #                         self.model.linear.weight.data[self.old_labels,:] = weight_org[self.old_labels,:]
    #                     #print(self.model.linear.weight)
    #                     #assert False
    #
    #             # update mem
    #             self.buffer.update(batch_x, batch_y)
    #
    #             if i % 100 == 1 and self.verbose:
    #                 print(
    #                     '==>>> it: {}, avg. loss: {:.6f}, '
    #                     'running train acc: {:.3f}'
    #                         .format(i, losses_batch.avg(), acc_batch.avg())
    #                 )
    #                 print(
    #                     '==>>> it: {}, mem avg. loss: {:.6f}, '
    #                     'running mem acc: {:.3f}'
    #                         .format(i, losses_mem.avg(), acc_mem.avg())
    #                 )
    #     self.after_train()