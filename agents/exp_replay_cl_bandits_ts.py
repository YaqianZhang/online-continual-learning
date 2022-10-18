import torch
from torch.utils import data
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.utils import maybe_cuda, AverageMeter
from RL.RL_replay_base_stop import RL_replay
from RL.close_loop_cl import close_loop_cl
from torchvision.transforms import transforms
from agents.exp_replay_cl import ExperienceReplay_cl

import numpy as np
import torch
import torch.nn as nn
from utils.setup_elements import transforms_match
from utils.utils import cutmix_data


class ExperienceReplay_cl_bandits_ts(ExperienceReplay_cl):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_cl_bandits_ts, self).__init__(model, opt, params)
        self.latest_rewards=[]
        self.action_num = self.params.mem_iters
        for i in range(self.action_num):
            self.latest_rewards.append([])
        self.mean=[1]*self.action_num
        self.std=[1]*self.action_num

    def store_reward(self,mem_iter, reward):
        self.latest_rewards[mem_iter].append(reward)
        recent=self.params.slide_window_size
        self.mean[mem_iter]=np.mean(self.latest_rewards[mem_iter][-recent:])
        self.std[mem_iter] = np.std(self.latest_rewards[mem_iter][-recent:])

    def sample_bandit(self):
        sampled_reward = np.random.normal(self.mean,self.std,(self.action_num))
        #print(sampled_reward)
        action = np.argmax(sampled_reward[1:])


        return action+1



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
        prev_test_acc = 0
        increase = 0

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.mem_iters = self.sample_bandit()
                memiter=self.mem_iters


                for j in range(memiter+1):

                    #self.set_aug_para(N, int(j*30/self.mem_iters), incoming_M=int(j*30/self.mem_iters))


                    concat_batch_x,concat_batch_y,mem_num = self.concat_memory_batch(batch_x,batch_y)




                    stats, STOP_FLAG = self._batch_update(concat_batch_x,concat_batch_y, losses_batch, acc_batch, i,mem_num=mem_num)
                    #if (self.params.immediate_evaluate == True):
                    if (self.params.use_test_buffer):
                        test_res = self.close_loop_cl.compute_testmem_loss()

                        if(test_res != None):
                            if (j == 0):
                                prev_test_acc = test_res[0]
                                increase = test_res[0]
                            if (j > 0):
                                increase = test_res[0] - prev_test_acc
                                reward = increase - j*self.params.iter_penalty
                                self.store_reward(j,reward)
                                # if(j==9):
                                #     print(j,increase)

                            #prev_test_acc = test_res[0]




                self.mem_iter_list.append(memiter+1)
                # if (self.params.immediate_evaluate == False):
                #     if(self.params.use_test_buffer):
                #         test_res = self.close_loop_cl.compute_testmem_loss()
                    #print("test acc",test_res)
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
                    print("mem_iter", memiter+1,self.mean)
        self.after_train()


