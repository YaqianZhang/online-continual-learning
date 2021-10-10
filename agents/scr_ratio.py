import torch
from torch.utils import data

from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter

import numpy as np

from RL.RL_replay_base import RL_replay
from RL.RL_replay_MAB import RL_replay_MAB

from RL.close_loop_cl import close_loop_cl
from agents.scr import SupContrastReplay

class SCR_RL_ratio(SupContrastReplay):
    def __init__(self, model, opt, params):
        super(SCR_RL_ratio, self).__init__(model, opt, params)

        self.close_loop_cl = close_loop_cl(self,model, self.memory_manager)

        self.RL_replay = RL_replay(params,self.close_loop_cl)

    def _batch_update(self, concat_batch_x, concat_batch_y, losses_batch, acc_batch, i, replay_para=None, mem_num=0):

        scr_loss = self.perform_scr_update(concat_batch_x, concat_batch_y, )

        softmax_loss,acc_incoming,incoming_loss = self.perform_softmax_update(concat_batch_x, concat_batch_y, mem_num)


        return acc_incoming, incoming_loss

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
                batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)
                self.set_memIter()
                for j in range(self.mem_iters):
                    batch_x, batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)

                    self.close_loop_cl.set_weighted_test_stats(batch_y, mem_num, )

                    replay_para = self.RL_replay.make_replay_decision(i)  # and update RL

                    self._batch_update(batch_x, batch_y, losses_batch, acc_batch, i, replay_para, mem_num=mem_num)

                    ## compute reward
                    self.close_loop_cl.compute_testmem_loss()

                    self.RL_replay.set_reward()

                    # self.close_loop_cl.set_train_stats( i, acc_incoming=acc_incoming, incoming_loss=incoming_loss)

                self.memory_manager.update_memory(batch_x[mem_num:], batch_y[mem_num:])

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

                    print("replay_para", replay_para, "action:", self.RL_replay.RL_agent.greedy,
                          self.RL_replay.action, )
        self.after_train()

    # def train_learner(self, x_train, y_train):
    #     self.memory_manager.reset_new_old()
    #     self.before_train(x_train, y_train)
    #     # set up loader
    #     train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
    #     train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
    #                                    drop_last=True)
    #     # set up model
    #     self.model = self.model.train()
    #
    #     # setup tracker
    #     losses = AverageMeter()
    #     acc_batch = AverageMeter()
    #
    #     for ep in range(self.epoch):
    #         for i, batch_data in enumerate(train_loader):
    #             # batch update
    #             batch_x, batch_y = batch_data
    #             batch_x = maybe_cuda(batch_x, self.cuda)
    #             batch_y = maybe_cuda(batch_y, self.cuda)
    #
    #             replay_para = self.RL_replay.make_replay_decision(i)
    #             batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)
    #
    #             scr_loss,combined_batch,combined_labels,mem_num  = self.perform_scr_update(batch_x, batch_y, losses)
    #
    #             softmax_loss, acc_incoming, incoming_loss= self.perform_softmax_update(combined_batch,combined_labels,mem_num,replay_para)
    #                                                                                      ## compute reward
    #
    #             ## compute reward
    #             self.close_loop_cl.compute_testmem_loss()
    #
    #
    #             self.RL_replay.set_reward()
    #
    #             self.close_loop_cl.set_train_stats_scr(scr_loss, i, softmax_loss, acc_incoming, incoming_loss)
    #
    #
    #             self.memory_manager.update_memory(batch_x, batch_y)
    #             if i % 100 == 1 and self.verbose:
    #                 print(
    #                     '==>>> it: {}, avg. loss: {:.6f}, '
    #                         .format(i, losses.avg(), acc_batch.avg())
    #                 )
    #                 #print("lr", self.params.learning_rate)
    #                 print("replay_para", replay_para)
    #
    #
    #
    #     self.after_train()

