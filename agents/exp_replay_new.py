import torch
import time
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer
from utils.buffer.tmp_buffer import Tmp_Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from utils.buffer.test_buffer import Test_Buffer
from RL.RL_agent import RL_agent,RL_agent_2dim
from RL.env import RL_env,RL_env_2dim
from utils.buffer.buffer_utils import random_retrieve


class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)

        self.params = params
        if (params.retrieve == "RL" or params.use_test_buffer):
            if(params.RL_type =="2dim"):
                self.RL_agent = RL_agent_2dim(params)
                self.RL_env = RL_env_2dim(params, model)
            else:
                self.RL_agent = RL_agent(params)
                self.RL_env = RL_env(params, model)

            self.test_buffer = Test_Buffer(params,self.RL_agent,self.RL_env)
            self.buffer = Buffer(model, params,self.RL_agent,self.RL_env)
            #print("initial RL objects")
        else:
            self.buffer = Buffer(model, params)

        if(params.use_tmp_buffer):
            self.tmp_buffer=Tmp_Buffer(model,params,self.buffer)
        else:
            self.tmp_buffer = None

        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.mem_iter_list =[]

        ## save train acc
        self.train_acc_incoming = []
        self.train_acc_mem=[]
    def save_mem_iters(self,prefix):
        arr = np.array(self.mem_iter_list)
        np.save(prefix + "mem_iter_list", arr)

    def save_training_acc(self,prefix):
        arr = np.array(self.train_acc_incoming)
        np.save(prefix + "train_acc_incoming.npy", arr)

        arr = np.array(self.train_acc_mem)
        np.save(prefix + "train_acc_mem.npy", arr)

    def batch_loss(self,batch_x,batch_y,losses_log,acc_log,zero_grad = True):
        logits = self.model.forward(batch_x)
        loss = self.criterion(logits, batch_y)
        if self.params.trick['kd_trick']:
            loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                   self.kd_manager.get_kd_loss(logits, batch_x)
        if self.params.trick['kd_trick_star']:
            loss = 1 / ((self.task_seen + 1) ** 0.5) * loss + \
                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
        _, pred_label = torch.max(logits, 1)
        correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
        # update tracker
        acc_log.update(correct_cnt, batch_y.size(0))
        losses_log.update(loss, batch_y.size(0))
        # backward
        if(zero_grad):
            self.opt.zero_grad()
        loss.backward()
        return correct_cnt


    def perform_SGD(self,mem_x,mem_y,batch_x,batch_y):
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
            self.opt.step()

    def adjust_mem_iter(self,acc_mem,reward):
        if(reward == None): return
        if (self.params.dyna_mem_iter):
            if (self.test_buffer.current_index >= self.params.test_mem_batchSize):
                if (acc_mem.avg() > reward):
                    # self.mem_iters = 0
                    self.mem_iters -= 1
                    if (self.mem_iters < self.params.mem_iter_min): self.mem_iters = self.params.mem_iter_min
                    # print(self.mem_iters,"increase")
                else:
                    # self.mem_iters = 1
                    self.mem_iters += 1
                    if (self.mem_iters > self.params.mem_iter_max): self.mem_iters = self.params.mem_iter_max


    def update_memory(self,batch_x,batch_y):
        # save some parts of batch_x and batch_y into the memory
        if (self.params.retrieve == "RL" or self.params.use_test_buffer):
            test_size = int(batch_x.shape[0] * 0.5)
            # print("save batch to test buffer and buffer",test_size)
            self.test_buffer.update(batch_x[:test_size], batch_y[:test_size])
            self.buffer.update(batch_x[test_size:], batch_y[test_size:], self.tmp_buffer)

        else:
            self.buffer.update(batch_x, batch_y, self.tmp_buffer)

    def check_start_RL(self):
        return self.test_buffer.current_index >= self.params.test_mem_batchSize

    def train_learner(self, x_train, y_train):

        self.buffer.task_seen_so_far += 1
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

        if (self.params.dyna_mem_iter): self.mem_iters = 2

        start_RL_session = False

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                self.mem_iter_list.append(self.mem_iters)
                reward = None
                for j in range(self.mem_iters):

                    ## incoming update
                    correct_cnt = self.batch_loss(batch_x,batch_y,losses_batch,acc_batch,zero_grad = True)
                    self.train_acc_incoming.append(correct_cnt)

                    # mem update
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    mem_x = maybe_cuda(mem_x)
                    mem_y = maybe_cuda(mem_y)
                    if mem_x.size(0) > 0:
                        correct_cnt = self.batch_loss(mem_x,mem_y,losses_mem,acc_mem,zero_grad=False)
                        self.train_acc_mem.append(correct_cnt)

                    if (start_RL_session):# test memory pre sgd
                        self.RL_env.pre_sgd_loss(self.test_buffer, self.buffer,)

                    self.perform_SGD(mem_x, mem_y, batch_x, batch_y)


                    if (start_RL_session):
                        reward = self.RL_env.compute_reward(self.test_buffer) ## TODO: with other RL design, reward may be computed elsewhere

                        self.RL_agent.update_agent(reward)

                self.adjust_mem_iter(acc_mem,reward)

                self.update_memory(batch_x,batch_y)

                if (self.params.retrieve == "RL" or self.params.use_test_buffer):
                    start_RL_session = self.check_start_RL()

                if (i+1) % 100 == 0 and self.verbose:
                    if(self.params.retrieve == "RL" or self.params.use_test_buffer):
                        if (self.test_buffer.current_index > 50):
                            print("reward", reward,self.test_buffer.current_index,self.mem_iters)
                    print(
                        '==>>> it: {},  '
                        'running train acc: {:.3f}, '
                        'running mem acc: {:.3f}'
                            .format(i, acc_batch.avg(),acc_mem.avg())
                    )

            if(self.params.use_tmp_buffer):
                self.tmp_buffer.update_true_buffer()

        self.after_train()
