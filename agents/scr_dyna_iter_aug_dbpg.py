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
# from RL.pytorch_util import  build_mlp
import numpy as np
from utils.utils import cutmix_data
from torchvision.transforms import transforms
from utils.augmentations import RandAugment

from agents.scr import SupContrastReplay
def softmax(vec, j=0):
    # nn=vec-np.mean(vec)

    para = 0  # 1.0/ np.sqrt(j+1)

    noise = np.random.rand(1) * 2 - 1  ###[-1, 1]
    nn = (1 - para) * vec + (para) * noise

    nn = nn - np.max(nn)
    # print np.max(nn)

    nn1 = np.exp(nn)
    # print np.max(nn1)
    # nn1 = 1/(1+np.exp(-nn))
    vec_prob = nn1 * 1.0 / np.sum(nn1)
    return vec_prob

class SCR_dyna_aug_iter_dbpg(SupContrastReplay):
    def __init__(self, model, opt, params):
        super(SCR_dyna_aug_iter_dbpg, self).__init__(model, opt, params)



    def set_dyna_iter_aug(self,acc_mem):
        return self.params.randaug_N,self.params.randaug_M

    def drift_detection(self):
        reward_mean = []
        recent = 20
        for i in range(self.action_num):
            avrg = np.mean(self.reward_list[i][-recent:])
            reward_mean.append(avrg)
        current_opt = np.argmax(self.weights)
        reward_opt = np.max(reward_mean)
        current_max = reward_mean[current_opt]
        if (reward_opt - current_max > 0.3):
            return True
        else:
            return False

    def update_reward(self, reward, action):
        self.reward_list[action].append(reward)

    def update_weight_bpg(self, acc_mem, current_iter,STOP_FLAG):

        prob = self.action_prob
        last_acc = acc_mem

        target_acc_start = self.params.train_acc_min  # 0.80
        target_acc_end = self.params.train_acc_max  # 0.9
        alpha = self.params.bpg_lr
        action_id = current_iter - self.mem_iter_min

        if (last_acc > target_acc_end or STOP_FLAG and current_iter > self.mem_iter_min):  ## too large
            logs = [current_iter, "too large", last_acc, target_acc_end, STOP_FLAG]

            if np.sum(prob[action_id:]) > 0:
                self.weights[action_id:] -= alpha * prob[action_id] / np.sum(prob[action_id:])

            if np.sum(prob[:action_id + 1]) > 0:
                self.weights[:action_id] += alpha * prob[action_id] / np.sum(prob[:action_id + 1])
        elif (last_acc < target_acc_start and current_iter < self.mem_iter_max):  ## too small reduce the smaller
            logs = [current_iter, "too small", last_acc, target_acc_start]
            ## better
            if np.sum(prob[action_id:]) > 0:
                self.weights[action_id:] += alpha * prob[action_id] / np.sum(prob[action_id:])

            ## worse
            if np.sum(prob[:action_id + 1]) > 0:
                self.weights[:action_id] -= alpha * prob[action_id] / np.sum(prob[:action_id + 1])

        else:
            logs = [current_iter, self.action_prob]
            pass
        self.action_prob = softmax(self.weights)  # 200*1
        return logs

    def restart_bpg(self):

        self.weights = 100 * np.ones(self.action_num)
        self.action_prob = softmax(self.weights)

    def sample_action_bpg(self):

        ### sample from the action probabilty
        # self.action_prob = softmax(self.weights) # 200*1
        # print(self.action_num,self.action_prob.shape)
        # assert False

        action_idx = np.random.choice(range(0, self.action_num), 1, replace=False, p=self.action_prob)[0]

        return self.action_map[action_idx]  ## choice [action]

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
        acc_mem = None


        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                N,M,memiter = self.set_dyna_iter_aug(acc_mem)
                #self.mem_iter_list.append(memiter)
                #self.RL_replay.RL_agent.greedy_action.append(memiter)
                aug_strength=N*100+M
                self.aug_agent.set_aug_NM(N, M)
                self.aug_N_list.append(aug_strength)
                self.mem_iter_list.append(memiter)


                for j in range(memiter):
                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)
                    scr_loss = self.perform_scr_update(concat_batch_x, concat_batch_y)
                    acc_mem = self.perform_softmax_update(concat_batch_x, concat_batch_y,mem_num)

                STOP_FLAG = self.early_stop_check(train_stats)
                logs = self.update_weight_bpg(acc_mem,memiter)
                if(acc_mem == None):
                    reward = np.abs(acc_mem -0.95)
                    self.update_reward(reward,memiter-1)
                    if(self.params.drift_detection):
                        DETECT = self.drift_detection()
                        if(DETECT):
                            self.restart_bpg()

                # update mem
                self.memory_manager.update_memory(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                            .format(i, losses.avg(), acc_batch.avg())
                    )
                    print("Iter", memiter,aug_strength)


        self.after_train()



