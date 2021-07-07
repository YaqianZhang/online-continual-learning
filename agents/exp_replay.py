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
from RL.RL_MDP_agent import RL_memIter_agent
from utils.buffer.buffer_utils import random_retrieve


class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)

        self.params = params
        if (params.RL_type != "NoRL" ):

            if(params.RL_type =="2dim"):
                self.RL_agent = RL_agent_2dim(params)
                self.RL_env = RL_env_2dim(params, model)
                self.params.retrieve = "RL"
            elif(params.RL_type =="1dim"):
                self.RL_agent = RL_agent(params)
                self.RL_env = RL_env(params, model)
                self.params.retrieve = "RL"
            elif(params.RL_type == "RL_memIter" or params.RL_type == "RL_ratio" or params.RL_type  == "RL_ratioMemIter"):
                print("RL_memIter")
                self.RL_agent = RL_memIter_agent(params)
                self.RL_env = RL_env(params, model)

            elif(params.RL_type == "DormantRL"): ## use test_buffer
                self.RL_agent = RL_agent(params)
                self.RL_env = RL_env(params, model)
            else:
                raise NotImplementedError("Not implemented RL type")
            self.params.use_test_buffer = True
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
        self.incoming_ratio = params.ratio
        self.mem_iter_list =[]

        ## save train acc
        self.train_acc_incoming = []
        self.train_acc_mem=[]
        self.test_acc_mem=[]
        self.start_RL = False
        self.evaluator = None


    def reset_buffer(self,params,model):
        self.buffer.reset()
        self.test_buffer.reset()




    def save_mem_iters(self,prefix):
        arr = np.array(self.mem_iter_list)
        np.save(prefix + "mem_iter_list", arr)

    def save_training_acc(self,prefix):
        arr = np.array(self.train_acc_incoming)
        np.save(prefix + "train_acc_incoming.npy", arr)

        arr = np.array(self.train_acc_mem)
        np.save(prefix + "train_acc_mem.npy", arr)

        arr = np.array(self.test_acc_mem)
        np.save(prefix + "test_acc_mem.npy", arr)

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


    # def perform_SGD(self,mem_x,mem_y,batch_x,batch_y):
    #     if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
    #         # opt update
    #         self.opt.zero_grad()
    #         combined_batch = torch.cat((mem_x, batch_x))
    #         combined_labels = torch.cat((mem_y, batch_y))
    #         combined_logits = self.model.forward(combined_batch)
    #         loss_combined = self.criterion(combined_logits, combined_labels)
    #         loss_combined.backward()
    #         self.opt.step()
    #     else:
    #         self.opt.step()

    def er_acc_test_buffer(self,test_memory,):

        if(test_memory.current_index==0):
            print("Test memory is empty")
            return None

        with torch.no_grad():
            batch_x,batch_y = random_retrieve(test_memory,self.params.test_mem_batchSize)
            logits = self.model.forward(batch_x)
            _, pred_label = torch.max(logits, 1)
            test_memory_loss = self.criterion(logits,batch_y)
            #print(pred_label) ## TODO: never predict classes not seen
            correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

        return correct_cnt, test_memory_loss.item()

    def adjust_mem_iter(self,acc_mem,reward):
        if(reward == None): return

        if (self.test_buffer.current_index >= self.params.test_mem_batchSize):
            if (acc_mem > reward):
                # self.mem_iters = 0
                self.mem_iters -= 1
                if (self.mem_iters < self.params.mem_iter_min): self.mem_iters = self.params.mem_iter_min
                # print(self.mem_iters,"increase")
            else:
                # self.mem_iters = 1
                self.mem_iters += 1
                if (self.mem_iters > self.params.mem_iter_max): self.mem_iters = self.params.mem_iter_max


    def update_memory_before(self,batch_x,batch_y):
        test_size = int(batch_x.shape[0] * 0.2)
        # print("save batch to test buffer and buffer",test_size)
        self.test_buffer.update(batch_x[:test_size], batch_y[:test_size])
        return batch_x[test_size:], batch_y[test_size:] #todo save the sample that not used in test memory into training memory


    def update_memory(self,batch_x,batch_y):
        # save some parts of batch_x and batch_y into the memory
        if (self.params.use_test_buffer and self.params.test_mem_type =="after"):
            test_size = int(batch_x.shape[0] * 0.5)
            # print("save batch to test buffer and buffer",test_size)
            self.test_buffer.update(batch_x[:test_size], batch_y[:test_size])
            self.buffer.update(batch_x[test_size:], batch_y[test_size:], self.tmp_buffer)

        else:
            self.buffer.update(batch_x, batch_y,self.tmp_buffer)

    def check_start_RL(self):
        if(self.task_seen==0):
            return False
        if(self.params.RL_type != "NoRL"):
            self.start_RL = self.test_buffer.current_index >= self.params.test_mem_batchSize
        else:
            self.start_RL = False
    def save_stats_list(self,stats):
        [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem,
         loss_incoming_value, loss_mem_value, loss_test_value, ] = stats

        self.test_acc_mem.append(correct_cnt_test_mem)
        self.train_acc_mem.append(correct_cnt_mem)
        self.train_acc_incoming.append(correct_cnt_incoming)



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
                self.mem_iter_list.append(self.mem_iters)


                if(self.params.test_mem_type == "before"):
                    batch_x,batch_y = self.update_memory_before(batch_x,batch_y)
                if(self.params.state_type[:10] == "same_batch" ):
                    stats = self.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem,iters=1 ,ratio=1)
                    if(self.task_seen>0 and  self.start_RL):

                        [correct_cnt_incoming, correct_cnt_mem,correct_cnt_test_mem,
                                                                    loss_incoming_value, loss_mem_value, loss_test_value,]=stats

                        if(self.params.state_type == "same_batch" ):
                            state_type = "6_dim"
                        else:
                            state_type = "7_dim"

                        next_state = self.RL_agent.convert_to_state(correct_cnt_incoming, correct_cnt_mem,
                                                                    correct_cnt_test_mem,
                                                                    loss_incoming_value, loss_mem_value, loss_test_value, i,state_type=state_type)
                        if(state != None and reward != None and action != None):
                            done =( i %50 ==0)
                            self.RL_agent.update_agent(reward, state,
                                                       action, next_state, done) ## todo next state and done
                        state = next_state

                        self.RL_agent.current_state = state
                        action,action_mem_iter,action_ratio = self.set_replay_para(state)
                        if(action != None):

                            if(action_mem_iter>0):
                                stats = self.joint_training(batch_x, batch_y, losses_batch, acc_batch, losses_mem, acc_mem, iters=action_mem_iter,ratio=action_ratio)




                            reward= self.RL_env.get_reward(stats[:3],self.evaluator,self.model,self.task_seen,correct_cnt_test_mem)
                                                                                #-correct_cnt_test_mem*100



                            self.RL_agent.current_reward =  reward
                            self.mem_iters=action_mem_iter
                            self.incoming_ratio = action_ratio


                            # self.RL_agent.update_agent(reward, state,
                            #                            action, state, action) ## todo next state and done
                        self.save_stats_list(stats)

                else:
                    self.set_replay_para(self.RL_agent.current_state)

                    stats = self.joint_training(  batch_x, batch_y, losses_batch, acc_batch, losses_mem,acc_mem,
                                                  iters=self.mem_iters,
                                                  ratio=self.incoming_ratio)

                    self.adjust_replay_dynamics(stats,i)

                self.update_memory(batch_x,batch_y)

                self.check_start_RL()

                if (i) % 50 == 0 and self.verbose:
                    if(self.params.RL_type != "NoRL"):
                        if (self.test_buffer.current_index > 50):
                            print("reward", self.RL_agent.current_reward,stats[:3]," MemIter:",self.mem_iters," ratio:",self.incoming_ratio)
                    print(
                        '==>>> it: {},  '
                        'running train acc: {:.3f}, '
                        'running mem acc: {:.3f}'
                            .format(i, acc_batch.avg(),acc_mem.avg())
                    )

            if(self.params.use_tmp_buffer):
                self.tmp_buffer.update_true_buffer()

        self.after_train()




    def joint_training(self,batch_x, batch_y, losses_batch, acc_batch,losses_mem, acc_mem,iters=1,ratio=1):
      ## mem_iter>1
        for j in range(iters):
            ## incoming update
            correct_cnt_incoming,loss_incoming = self.batch_loss(batch_x, batch_y, losses_batch, acc_batch,ratio = 1 )
            #self.train_acc_incoming.append(correct_cnt_incoming)
            self.opt.zero_grad()
            loss_incoming.backward()
            loss_incoming_value = loss_incoming.item()
            if(self.params.test_retrieval_step >-1):
                if(self.task_seen % 2==1):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                else:
                    mem_x, mem_y = random_retrieve(self.test_buffer, 10)
            else:
                mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)


            # if((self.task_seen>=5  and self.task_seen<=10 )or self.task_seen>=15):
            #     mem_x, mem_y = random_retrieve(self.test_buffer,10)
            #     #mem_x, mem_y = self.test_buffer.retrieve(x=batch_x, y=batch_y)
            # else:
            #     mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

            # mem update
            #mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            mem_x = maybe_cuda(mem_x)
            mem_y = maybe_cuda(mem_y)
            if mem_x.size(0) > 0: ## the first batch training does not have memory batch
                correct_cnt_mem,loss_mem = self.batch_loss(mem_x, mem_y, losses_mem, acc_mem, ratio = ratio )
                #self.train_acc_mem.append(correct_cnt_mem)
                loss_mem.backward()
                loss_mem_value = loss_mem.item()
            else:
                correct_cnt_mem = None
                loss_mem_value = None

            self.opt.step()

            if (self.start_RL):
                acc_testMem,loss_test_value = self.er_acc_test_buffer(self.test_buffer)
                #self.test_acc_mem.append(acc_testMem)
                #acc_testMem = self.RL_env.compute_acc_test_buffer(self.test_buffer)  ## TODO: with other RL design, reward may be computed elsewhere
            else:
                acc_testMem = None
                loss_test_value = None

        return correct_cnt_incoming, correct_cnt_mem, acc_testMem,loss_incoming_value,loss_mem_value,loss_test_value



    def set_replay_para(self,state):
        if(self.start_RL ==  False):
            return None, None,None
        if(self.RL_agent.current_state == None):
            return None, None,None
        if(self.params.RL_type == "RL_memIter"):
            self.RL_agent.current_action = self.RL_agent.sample_action(state)
            self.mem_iters  = self.RL_agent.from_action_to_memIter(self.RL_agent.current_action)
            return self.RL_agent.current_action,self.mem_iters,self.incoming_ratio
        elif(self.params.RL_type == "RL_ratio"):
            self.RL_agent.current_action = self.RL_agent.sample_action(state)
            self.incoming_ratio = self.RL_agent.from_action_to_ratio(self.RL_agent.current_action)
            return self.RL_agent.current_action,self.mem_iters,self.incoming_ratio
        elif(self.params.RL_type == "RL_ratioMemIter"):
            self.RL_agent.current_action = self.RL_agent.sample_action(state)
            self.mem_iters,self.incoming_ratio = self.RL_agent.from_action_to_ratio_memIter(self.RL_agent.current_action)
            return self.RL_agent.current_action, self.mem_iters, self.incoming_ratio
        else:
            return None, None,None



    def adjust_replay_dynamics(self,stats,i):
        if(self.task_seen==0): return # todo: all methods pass for the first task
        [correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem,loss_incoming_value,loss_mem_value,loss_test_value]=stats
        if(correct_cnt_test_mem != None and correct_cnt_mem != None  and correct_cnt_test_mem != None):
            self.test_acc_mem.append(correct_cnt_test_mem)
            self.train_acc_mem.append(correct_cnt_mem)
            self.train_acc_incoming.append(correct_cnt_incoming)
        if (self.params.RL_type == "RL_memIter" or self.params.RL_type  == "RL_ratio"):
            if(self.start_RL == False):
                return
            next_state = self.RL_agent.convert_to_state(correct_cnt_incoming, correct_cnt_mem, correct_cnt_test_mem,
                                                        loss_incoming_value,loss_mem_value,loss_test_value,i,state_type=self.params.state_type)
            # reward = correct_cnt_test_mem - prev_acc

            if(self.params.reward_type == "real_reward"):
                self.RL_agent.current_reward = self.evaluator.evaluate_model(self.model,self.task_seen)
            else:

                self.RL_agent.current_reward = self.RL_env.get_reward(stats[:3])

            #prev_acc = correct_cnt_test_mem
            done = (i == (self.batch_num - 1))
            if(self.RL_agent.current_state != None and  self.RL_agent.current_action != None):
                self.RL_agent.update_agent(self.RL_agent.current_reward , self.RL_agent.current_state, self.RL_agent.current_action, next_state, done)
            self.RL_agent.current_state = next_state
        else:
            if(self.params.dyna_mem_iter == "random"):
                self.mem_iters = np.random.randint(self.params.mem_iter_min,self.params.mem_iter_max+1)

            elif(self.params.dyna_mem_iter == "dyna"):
                self.adjust_mem_iter(correct_cnt_mem, correct_cnt_test_mem)
            elif(self.params.dyna_mem_iter == "None"):
                pass
            else:
                raise NotImplementedError("Not defined dyna_mem_iter strategy")


            if(self.params.dyna_ratio == "random"):
                self.incoming_ratio = np.random.uniform(0.1,2)

            if(self.params.RL_type != "NoRL"):

                self.RL_agent.current_reward = correct_cnt_test_mem



