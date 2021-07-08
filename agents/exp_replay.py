import torch
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer
from utils.buffer.tmp_buffer import Tmp_Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from utils.buffer.test_buffer import Test_Buffer

from RL.agent.RL_agent_MDP import RL_memIter_agent
from RL.env.RL_env_MDP import RL_env_MDP
from utils.buffer.buffer_utils import random_retrieve


class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)

        self.mem_iter_list =[]
        self.incoming_ratio_list=[]
        self.setup_RL_agent(model,params)

        ## save train acc
        self.train_acc_incoming = []
        self.train_acc_mem=[]
        self.test_acc_mem=[]

        self.train_loss_incoming = []
        self.train_loss_mem=[]
        self.test_loss_mem=[]

        self.start_RL = False
        self.evaluator = None
        self.buffer = Buffer(model, params)

    def setup_RL_agent(self,model,params):
        if (params.RL_type != "NoRL" ):
            self.test_buffer = Test_Buffer(params, )
            self.params.use_test_buffer = True
            if (params.RL_type in [ "RL_ratio","RL_memIter","RL_ratioMemIter","DormantRL",]):
                self.RL_agent = RL_memIter_agent(params)
                self.RL_env = RL_env_MDP(params, model,self.test_buffer,self.RL_agent,self)
            else:
                raise NotImplementedError("undefined RL_type")
        else:
            self.params.use_test_buffer = False

            # if(params.RL_type =="2dim"):
            #     self.RL_agent = RL_agent_2dim(params)
            #     self.RL_env = RL_env_2dim(params, model)
            #     self.params.retrieve = "RL"
            # elif(params.RL_type =="1dim"):
            #     self.RL_agent = RL_agent(params)
            #     self.RL_env = RL_env(params, model)
            #     self.params.retrieve = "RL"


        if(params.use_tmp_buffer):
            self.tmp_buffer=Tmp_Buffer(model,params,self.buffer)
        else:
            self.tmp_buffer = None



    def reset_buffer(self,params,model):
        self.buffer.reset()
        self.test_buffer.reset()


    def save_mem_iters(self,prefix):
        arr = np.array(self.mem_iter_list)
        np.save(prefix + "mem_iter_list", arr)
        arr = np.array(self.incoming_ratio_list)
        np.save(prefix + "incoming_ratio_list", arr)

    def save_training_acc(self,prefix):
        arr = np.array(self.train_acc_incoming)
        np.save(prefix + "train_acc_incoming.npy", arr)

        arr = np.array(self.train_acc_mem)
        np.save(prefix + "train_acc_mem.npy", arr)

        arr = np.array(self.test_acc_mem)
        np.save(prefix + "test_acc_mem.npy", arr)

        arr = np.array(self.train_loss_incoming)
        np.save(prefix + "train_loss_incoming.npy", arr)

        arr = np.array(self.train_loss_mem)
        np.save(prefix + "train_loss_mem.npy", arr)

        arr = np.array(self.test_loss_mem)
        np.save(prefix + "test_loss_mem.npy", arr)

    def batch_loss(self,batch_x,batch_y,losses_log,acc_log,zero_grad = True,labels=None,ratio = 1.0):
        logits = self.model.forward(batch_x)

        loss = ratio*self.criterion(logits, batch_y)
        if self.params.trick['kd_trick']:
            loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                   self.kd_manager.get_kd_loss(logits, batch_x)
        if self.params.trick['kd_trick_star']:
            loss = 1 / ((self.task_seen + 1) ** 0.5) * loss + \
                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
        _, pred_label = torch.max(logits, 1)
        # if(labels != None):
        #     for x in pred_label:
        #
        #         if(x in labels):
        #             pass
        #         else:
        #             print("predict unseen labels",x,labels)

        correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
        # update tracker
        acc_log.update(correct_cnt, batch_y.size(0))
        losses_log.update(loss, batch_y.size(0))
        # backward

        return correct_cnt,loss




    def save_stats_list(self,stats):
        if(stats == None): return
        self.train_acc_mem.append(stats['correct_cnt_mem'])
        self.train_acc_incoming.append(stats['correct_cnt_incoming'])
        self.train_loss_mem.append(stats['loss_mem_value'])
        self.train_loss_incoming.append(stats['loss_incoming_value'])


    def save_test_stats_list(self,stats,):
        if (stats == None): return


        self.test_acc_mem.append(stats['correct_cnt_test_mem'])
        self.test_loss_mem.append(stats['loss_test_value'])

    def joint_training(self,batch_x, batch_y, losses_batch, acc_batch,losses_mem, acc_mem,iters=1,ratio=1):
      ## mem_iter>1
        for j in range(iters):
            ## incoming update
            correct_cnt_incoming,loss_incoming = self.batch_loss(batch_x, batch_y, losses_batch, acc_batch,ratio = 1 )

            self.opt.zero_grad()
            loss_incoming.backward()
            loss_incoming_value = loss_incoming.item()
            if(self.params.switch_buffer_type == "one_buffer"):
                mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            elif(self.params.switch_buffer_type == "two_buffer"):
                if (self.task_seen % 2 == 1):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                else:
                    mem_x, mem_y = random_retrieve(self.test_buffer, 10)
            else:
                raise NotImplementedError("Undefined buffer switch strategy",self.params.switch_buffer_type)


            # mem update
            #mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            mem_x = maybe_cuda(mem_x)
            mem_y = maybe_cuda(mem_y)
            if mem_x.size(0) > 0: ## the first batch training does not have memory batch
                correct_cnt_mem,loss_mem = self.batch_loss(mem_x, mem_y, losses_mem, acc_mem, ratio = ratio )
                #self.train_acc_mem.append(correct_cnt_mem)
                loss_mem.backward()
                loss_mem_value = loss_mem.item()
                stats_dict = {'correct_cnt_incoming': correct_cnt_incoming,
                              'correct_cnt_mem': correct_cnt_mem,
                              "loss_incoming_value": loss_incoming_value,
                              "loss_mem_value": loss_mem_value}
            else:
                correct_cnt_mem = None
                loss_mem_value = None
                stats_dict = None

            self.opt.step()


        return stats_dict

    def compute_test_accuracy(self,stats_dict):
        if(self.test_buffer.current_index==0):
            print("Test memory is empty")
            return None

        with torch.no_grad():
            batch_x,batch_y = random_retrieve(self.test_buffer,self.params.test_mem_batchSize) ## TODO: can change to class-balanced retrieve
            logits = self.model.forward(batch_x)
            _, pred_label = torch.max(logits, 1)
            test_memory_loss = self.criterion(logits,batch_y)
            correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
        stats_dict['correct_cnt_test_mem']=correct_cnt
        stats_dict['loss_test_value']=test_memory_loss.item()

        return stats_dict

    def update_memory(self,batch_x,batch_y):
        # save some parts of batch_x and batch_y into the memory
        if (self.params.use_test_buffer and self.params.test_mem_type =="after"):
            test_size = int(batch_x.shape[0] * 0.5)
            # print("save batch to test buffer and buffer",test_size)
            self.test_buffer.update(batch_x[:test_size], batch_y[:test_size])
            self.buffer.update(batch_x[test_size:], batch_y[test_size:], self.tmp_buffer)

        else:
            self.buffer.update(batch_x, batch_y,self.tmp_buffer)






    def train_learner(self, x_train, y_train,labels):
        if(self.params.episode_type == "batch"):
            self.RL_agent.initialize_q()
        ## reset q function

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
        self.batch_num = len(train_loader)
        state = None
        done = None
        action = None
        reward = None

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)


                ## TODO SAVEINFO,  multiple rl episode, multiple buffers #NoRL ,DormentRL,

                if(self.params.RL_type == "NoRL"):

                    stats_dict = self.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem, iters=self.params.mem_iters,
                                            ratio=self.params.incoming_ratio)
                    self.save_stats_list(stats_dict)
                    self.buffer.update(batch_x, batch_y, self.tmp_buffer)
                else:
                    stats_dict = self.RL_env.RL_joint_training(i,batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,
                                                  self.task_seen,er_agent=self)
                    self.save_stats_list(stats_dict)
                    self.save_test_stats_list(stats_dict)
                    self.mem_iter_list.append(self.RL_env.mem_iters)
                    self.incoming_ratio_list.append(self.RL_env.incoming_ratio)
                    self.update_memory(batch_x, batch_y)

                if (i) % 50 == 0 and self.verbose:
                    if(self.params.RL_type != "NoRL"):
                        if (self.test_buffer.current_index > 50):
                            print("reward", self.RL_env.reward," MemIter:",self.RL_env.mem_iters," ratio:",self.RL_env.incoming_ratio)
                    if(stats_dict!= None):
                        print(
                            '==>>> it: {},  '
                            'running train acc: {:.3f}, '
                            'running mem acc: {:.3f}'
                                .format(i,stats_dict["correct_cnt_incoming"],stats_dict["correct_cnt_mem"])
                        )
                    # print(
                    #     '==>>> it: {},  '
                    #     'running train acc: {:.3f}, '
                    #     'running mem acc: {:.3f}'
                    #     .format(i, acc_batch.avg(),acc_mem.avg())
                    # )

            if(self.params.use_tmp_buffer):
                self.tmp_buffer.update_true_buffer()

        self.after_train()



