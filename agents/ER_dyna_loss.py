from torch.utils import data
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from agents.exp_replay import  ExperienceReplay
import torch

from scipy.stats import linregress

# from RL.RL_replay_base import RL_replay
#
# from RL.close_loop_cl import close_loop_cl

import numpy as np


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


class ER_dyna_loss(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ER_dyna_loss, self).__init__(model, opt, params)
        self.mem_ratio = 1
        self.mem_ratio_max = 5
        self.mem_ratio_min = 0.1
        self.action_set = [0.1,0,-0.1]
        self.action_num = len(self.action_set)
        self.weights = 100*np.ones(self.action_num)
        self.action_prob = softmax(self.weights)
        self.state_list,self.reward_list, self.action_list = [], [], []



    def compute_state(self): ## todo design state
        pass

    def choose_action(self,state):

        ### sample from the action probabilty
        #self.action_prob = softmax(self.weights) # 200*1

        current_iter = np.random.choice(self.action_set, 1, replace=False, p=self.action_prob)

        return current_iter[0] ## choice [action]
    def from_action_to_replay_ratio(self,action):
        replay_para={}
        self.mem_ratio = self.mem_ratio + action
        if(self.mem_ratio > self.mem_ratio_max):
            self.mem_ratio = self.mem_ratio_max
        if(self.mem_ratio < self.mem_ratio_min):
            self.mem_ratio = self.mem_ratio_min
        replay_para={'mem_ratio':self.mem_ratio,
                     "incoming_ratio":1}
        return replay_para

    def compute_reward(self): ## todo design reward
        pass


    def store_experience(self,state,action,reward):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
    def _sample_experience(self):

         ## todo list type cannot be accessed by idx list

        idx = len(self.state_list)
        action_batch = self.action_list[idx]
        state_batch = self.state_list[idx]
        reward_batch = self.reward_list[idx]


        return action_batch, state_batch, reward_batch




    def update_RL_model(self,):

        action_batch, state_batch, reward_batch = self._sample_experience()
        logits = self.actor_NN(state_batch)
        loss = logits*reward_batch

        ## todo loss backward

        self.action_prob = softmax(self.weights) # 200*1
        #return logs



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
        replay_para = None
        STOP_FLAG = False

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                state = self.compute_state()
                action = self.choose_action(state)
                replay_para = self.from_action_to_replay_ratio(action)
                train_acc_list = []
                train_loss_list=[]
                DETECT = False


                for j in range(self.mem_iters):

                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)

                    train_stats = self._batch_update(concat_batch_x, concat_batch_y, losses_batch, acc_batch, i,replay_para,mem_num=mem_num)
                    if(train_stats != None):
                        train_acc_list.append(train_stats['acc_mem'])
                        train_loss_list.append(train_stats['loss_mem'])





                reward = self.compute_reward()
                self.store_experience(state,action,reward)
                self.update_RL_model()





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
                    print("memiter",memiter,logs,DETECT)#np.max(train_acc_list),train_acc_list[-1],np.mean(train_acc_list))

        self.after_train()

