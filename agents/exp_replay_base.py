import torch
import numpy as np

from agents.base import ContinualLearner

from utils.utils import maybe_cuda, AverageMeter

from utils.buffer.memory_manager import memory_manager_class


class ExperienceReplay_base(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_base, self).__init__(model, opt, params)

        self.memory_manager = memory_manager_class(model, params)
        self.stats = None

        self.episode_start_test_acc = None
        self.adaptive_ratio = False

        self.evaluator = None
        self.task_seen_so_far = 0
        self.incoming_batch = None
        self.replay_para = None

        ## save train acc
        self.mem_iter_list =[]
        self.incoming_ratio_list=[]
        self.mem_ratio_list=[]

        self.train_acc_incoming = []
        self.train_acc_mem=[]
        self.test_acc_mem=[]

        self.train_loss_incoming = []
        self.train_loss_mem=[]
        self.test_loss_mem=[]
        self.test_loss_mem_new=[]
        self.test_loss_mem_old=[]
        self.train_loss_old=[]



    def _init_all(self,model, init_funcs):

        for p in model.parameters():
            init_func = init_funcs.get(len(p.shape), init_funcs["default"])
            init_func(p)
            #print("model parameters",p.shape)

    def initialize_agent(self,params):
        ## empty buffer
        ## initialize model
        self.task_seen_so_far = 0

        self.memory_manager.reset_buffer(params)
        if(params.test == "not_reset"):
            pass
        else:
            model = self.model

            init_funcs = {
                1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.),  # can be bias
                2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.),  # can be weight
                3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.),  # can be conv1D filter
                4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.),  # can be conv2D filter
                "default": lambda x: torch.nn.init.constant(x, 1.),  # everything else
            }

            self._init_all(model, init_funcs)




    def batch_loss(self,batch_x,batch_y,losses_log=None,acc_log=None,need_grad = False,labels=None,ratio = 1.0):
        mem_x = maybe_cuda(batch_x)
        mem_y = maybe_cuda(batch_y)
        if(need_grad):
            logits = self.model.forward(batch_x)
        else:
            with torch.no_grad():
                logits = self.model.forward(batch_x)


        if(self.adaptive_ratio == False):

            loss = ratio*self.criterion(logits, batch_y)
        else:
            loss = ratio*self.adaptive_criterion(logits, batch_y,ratio)

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
        if(acc_log != None):
            acc_log.update(correct_cnt, batch_y.size(0))
        if(losses_log != None):
            losses_log.update(loss, batch_y.size(0))
        # backward

        return correct_cnt,loss



    def joint_training(self,replay_para,TEST=False):


        iters = replay_para['mem_iter']
        mem_ratio = replay_para['mem_ratio']
        incoming_ratio = replay_para['incoming_ratio']

        batch_x = self.incoming_batch['batch_x']
        batch_y = self.incoming_batch['batch_y']
        i = self.incoming_batch['batch_num']

        stats_dict = None


        for j in range(iters):
            ## incoming update
            correct_cnt_incoming,loss_incoming = self.batch_loss(batch_x, batch_y, self.losses_batch, self.acc_batch,
                                                                 ratio = incoming_ratio,need_grad=True)
            self.opt.zero_grad()
            loss_incoming.backward()
            loss_incoming_value = loss_incoming.item()

            mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y,self.task_seen)


            if mem_x.size(0) > 0: ## the first batch training does not have memory batch

                correct_cnt_mem,loss_mem = self.batch_loss(mem_x, mem_y, self.losses_mem, self.acc_mem, ratio = mem_ratio,need_grad=True)
                #self.train_acc_mem.append(correct_cnt_mem)
                loss_mem.backward()
                loss_mem_value = loss_mem.item()
                stats_dict = {'correct_cnt_incoming': correct_cnt_incoming,
                              'correct_cnt_mem': correct_cnt_mem,
                              "loss_incoming_value": loss_incoming_value,
                              "loss_mem_value": loss_mem_value,
                              "batch_num":i,
                              }
                #stats_dict = self.add_old_new_task_feature(mem_x, mem_y, mem_ratio, stats_dict)

            else:
                correct_cnt_mem = None
                loss_mem_value = None
                stats_dict = None

            self.opt.step()
        if(TEST):
            stats_dict = self.compute_test_accuracy(stats_dict)


        return stats_dict



    def save_mem_iters(self,prefix):
        arr = np.array(self.mem_iter_list)
        np.save(prefix + "mem_iter_list", arr)
        arr = np.array(self.incoming_ratio_list)
        np.save(prefix + "incoming_ratio_list", arr)
        arr = np.array(self.mem_ratio_list)
        np.save(prefix + "mem_ratio_list", arr)

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
        arr = np.array(self.test_loss_mem_new)
        np.save(prefix + "test_loss_mem_new.npy", arr)
        arr = np.array(self.test_loss_mem_old)
        np.save(prefix + "test_loss_mem_old.npy", arr)

        arr = np.array(self.train_loss_old)
        np.save(prefix + "train_loss_old.npy", arr)
        if(self.params.RL_type != "NoRL"):

            arr = np.array(self.RL_trainer.return_list)
            np.save(prefix + "return_list.npy", arr)


    def log_stats_list(self,stats):
        if(stats == None): return
        if("correct_cnt_mem" in stats.keys()):
            self.train_acc_mem.append(stats['correct_cnt_mem'])
            self.train_acc_incoming.append(stats['correct_cnt_incoming'])
            self.train_loss_mem.append(stats['loss_mem_value'])
            self.train_loss_incoming.append(stats['loss_incoming_value'])


    def log_test_stats_list(self,stats,):
        if (stats == None): return
        if("correct_cnt_test_mem" in stats.keys()):
            self.test_acc_mem.append(stats['correct_cnt_test_mem'])
            self.test_loss_mem.append(stats['loss_test_value'])

        if("loss_mem_old" in stats.keys()):

            self.test_loss_mem_old.append(stats['loss_mem_old'])
            self.test_loss_mem_new.append(stats['loss_mem_new'])
        if("train_loss_old" in stats.keys()):

            self.train_loss_old.append(stats['train_loss_old'])




