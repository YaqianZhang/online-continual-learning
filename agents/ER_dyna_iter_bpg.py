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

class ER_dyna_iter(ExperienceReplay):
    def __init__(self, model, opt, params):
        super(ER_dyna_iter, self).__init__(model, opt, params)
        #if(self.params.online_hyper_RL or self.params.scr_memIter ):

        # self.close_loop_cl = close_loop_cl(self,model, self.memory_manager)
        #
        # self.RL_replay = RL_replay(params,self.close_loop_cl)

    def set_dyna_iter(self,train_acc_list):
        if(self.params.dyna_type == "random"):
            return np.random.randint(low=self.params.mem_iter_min,high=self.params.mem_iter_max)
        else:
            return  self.dyna_train_acc(train_acc_list)
    def dyna_train_acc(self,train_acc_list):
        target_acc_start = self.params.train_acc_min #0.80
        target_acc_end=self.params.train_acc_max #0.9
        current_iter = len(train_acc_list)
        if(current_iter ==0 or current_iter == None):
            return self.params.mem_iters
        last_acc = train_acc_list[-1]
        max_acc = np.max(train_acc_list)
        mean_acc=np.mean(train_acc_list)
        acc=mean_acc
        #slope, intercept, r, p, se = linregress(np.arange(0,current_iter), train_acc_list)

        #print(r, p, train_acc_list)
        if(last_acc <target_acc_start): ## under fitting condition
            ## increase
            mem_iter =current_iter + 2
            if(mem_iter>self.params.mem_iter_max):
                mem_iter = self.params.mem_iter_max
            return mem_iter
        elif( max_acc >target_acc_end): ## overfitting condition
            #print(r,p,train_acc_list)
            return int(current_iter /2 )
        else:
            return current_iter


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

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                memiter = self.set_dyna_iter(self.memory_manager.current_performance)
                #self.mem_iter_list.append(memiter)
                #self.RL_replay.RL_agent.greedy_action.append(memiter)
                train_acc_list = []
                train_loss_list=[]


                for j in range(memiter):


                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)

                    train_stats = self._batch_update(concat_batch_x, concat_batch_y, losses_batch, acc_batch, i,replay_para,mem_num=mem_num)
                    if(train_stats != None):
                        train_acc_list.append(train_stats['acc_mem'])
                        train_loss_list.append(train_stats['loss_mem'])
                    STOP_FLAG = self.early_stop_check(train_stats)
                    if(STOP_FLAG ):
                        memiter=j
                        break

                self.mem_iter_list.append(memiter)

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
                    print("memiter",memiter,)#np.max(train_acc_list),train_acc_list[-1],np.mean(train_acc_list))

                    #print("replay_para", replay_para,"action:",self.RL_replay.RL_agent.greedy,self.RL_replay.action,)
        self.after_train()

