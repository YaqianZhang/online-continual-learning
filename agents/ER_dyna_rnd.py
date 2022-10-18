from torch.utils import data
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from agents.exp_replay import  ExperienceReplay

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


class ER_dyna_rnd(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ER_dyna_rnd, self).__init__(model, opt, params)
        #if(self.params.online_hyper_RL or self.params.scr_memIter ):

        # self.close_loop_cl = close_loop_cl(self,model, self.memory_manager)
        #
        # self.RL_replay = RL_replay(params,self.close_loop_cl)
        self.action_num = self.params.mem_iter_max - self.params.mem_iter_min +1
        self.weights = 100*np.ones(self.action_num)
        self.action_prob = softmax(self.weights)
        self.mem_iter_min = self.params.mem_iter_min
        self.mem_iter_max = self.params.mem_iter_max
        self.reward_list=[]
        for i in range(self.action_num):
            self.reward_list.append([0.5])
        self.N=None
    def adjust_aug(self,stats):
        # if(self.params.dyna_type == "bpg"):
        #     self.bpg_aug_adjust(stats)
        # else:
        return self.rule_aug_adjust(stats)


    def rule_aug_adjust(self,stats):
        action_map=[(1,5),(1,14),(2,14),(3,14),(4,14)]
        action_id = np.random.randint(0,5,1)[0]
        N=action_map[action_id][0]
        M=action_map[action_id][1]




        return N*100+M

    def train_learner(self, x_train, y_train):
        # if(self.params.bpg_restart):
        #     self.restart_bpg()
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
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                memiter = np.random.randint(1,21,1)[0]
                # print(memiter)
                # assert False
                #self.mem_iter_list.append(memiter)
                #self.RL_replay.RL_agent.greedy_action.append(memiter)
                train_acc_list = []
                train_loss_list=[]
                DETECT = False


                for j in range(memiter):


                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)

                    train_stats = self._batch_update(concat_batch_x, concat_batch_y, losses_batch, acc_batch, i,replay_para,mem_num=mem_num)
                    if(train_stats != None):
                        train_acc_list.append(train_stats['acc_mem'])
                        train_loss_list.append(train_stats['loss_mem'])
                    # if(STOP_FLAG ):
                    #     memiter=j+1
                    #     break

                self.mem_iter_list.append(memiter)

                if(train_stats != None):
                    N=self.adjust_aug(train_stats)
                    self.aug_N_list.append(N)
                else:
                    N=0

                self.memory_manager.current_performance=train_acc_list


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
                    print("memiter",memiter,"aug",N)#np.max(train_acc_list),train_acc_list[-1],np.mean(train_acc_list))

                    #print("replay_para", replay_para,"action:",self.RL_replay.RL_agent.greedy,self.RL_replay.action,)
        self.after_train()

