import torch
import numpy as np
from torch.utils import data

from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter

from RL.RL_trainer import RL_trainer
from RL.agent.RL_agent_MDP import RL_memIter_agent
from RL.env.RL_env_MDP import RL_env_MDP
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


    def _add_old_new_task_feature(self, mem_x, mem_y, mem_ratio, stats_dict):

        n = len(mem_y)

        idx_old = [mem_y[i] in self.old_labels for i in range(n)]
        idx_new = [mem_y[i] in self.new_labels for i in range(n)]

        (x_old, y_old, x_new, y_new) = (mem_x[idx_old], mem_y[idx_old], mem_x[idx_new], mem_y[idx_new])
        if (len(y_old) != 0 ):
            correct_cnt_mem_old, loss_mem_old = self.batch_loss(x_old, y_old, need_grad=False, ratio=mem_ratio)

            stats_dict.update({'correct_cnt_mem_old': correct_cnt_mem_old,
                               "loss_mem_old": loss_mem_old.detach().cpu(),
                               "old_task_num": len(y_old), })

        else:
            raise NotImplementedError(" no old classes in test mem!")

        if (len(y_new != 0)):
            correct_cnt_mem_new, loss_mem_new = self.batch_loss(x_new, y_new, need_grad=False, ratio=mem_ratio)
            stats_dict.update({ 'correct_cnt_mem_new': correct_cnt_mem_new,
                               "loss_mem_new": loss_mem_new.detach().cpu(),
                               "new_task_num": len(y_new)})
        else:

            #correct_cnt_mem_new, loss_mem_new = self.batch_loss(x_new, y_new, zero_grad=True, ratio=mem_ratio)
            stats_dict.update({ "new_task_num": len(y_new)})

            ## when there is no new classes in the test memory (e.g. after i=0 training), use incoming data stats
            if('correct_cnt_incoming' in stats_dict.keys()):
                stats_dict.update({ 'correct_cnt_mem_new': stats_dict['correct_cnt_incoming'],
                                   "loss_mem_new": stats_dict['loss_incoming_value'],
                                   })

        if(len(y_new != 0)):

            test_acc,test_loss = self._compute_avg(correct_cnt_mem_old, loss_mem_old,len(y_old),
                                               correct_cnt_mem_new, loss_mem_new ,len(y_new))
        else:
            [test_acc, test_loss] = [correct_cnt_mem_old, loss_mem_old,]


        #test_acc0,test_loss0,  = self.batch_loss(mem_x, mem_y, need_grad=False,)
        #print(test_acc0,test_acc,test_loss0,test_loss)
        #
        stats_dict['correct_cnt_test_mem'] = test_acc
        stats_dict['loss_test_value'] = test_loss.detach().cpu()  ## used for reward

        return stats_dict

    def _compute_avg(self,correct_cnt_mem_old, loss_mem_old,n_old,
                                               correct_cnt_mem_new, loss_mem_new ,n_new):
        test_acc = (correct_cnt_mem_old*n_old +correct_cnt_mem_new*n_new)/(n_old+n_new)

        test_loss = (loss_mem_old * n_old + loss_mem_new * n_new) / (n_old + n_new)

        return test_acc, test_loss



    def compute_test_accuracy(self, stats_dict):
        if(stats_dict == None):
            stats_dict={}


        if (self.memory_manager.test_buffer.current_index == 0):
            print("Test memory is empty")
            return None

        batch_x, batch_y = self.memory_manager.test_buffer.retrieve_all()
        # test_acc,test_loss,  = self.batch_loss(batch_x, batch_y, need_grad=False,)
        #
        # stats_dict['correct_cnt_test_mem'] = test_acc
        # stats_dict['loss_test_value'] = test_loss.detach().cpu()  ## used for reward
        stats_dict = self._add_old_new_task_feature(batch_x, batch_y, 1.0, stats_dict) ## used for next state

        #mem_x, mem_y = self.memory_manager.buffer.retrieve(retrieve_num=300)

        #train_loss,train_acc = self.evaluate_model(mem_x, mem_y)
        if(self.params.state_feature_type == "new_old6mn"):
           # print("!! add more features")

            mem_x, mem_y = self.memory_manager.buffer.retrieve(x=batch_x, y=batch_y,retrieve_num=50)
            stats_dict_train={}
            stats_dict_train = self._add_old_new_task_feature(batch_x, batch_y, 1.0, stats_dict_train)  ## used for next state
           # stats_dict['train_loss_new']=stats_dict_train['loss_mem_new']
            stats_dict['train_loss_old'] = stats_dict_train['loss_mem_old']

        return stats_dict

    # def compute_init_stats(self,batch_num):
    #     print("compute initial state")
    #     if(batch_num != 0):
    #         raise NotImplementedError("init_states wrong",batch_num)
    #     stats_dict={'batch_num':batch_num}
    #     x_old, y_old = self.memory_manager.test_buffer.retrieve_all()
    #
    #     correct_cnt_mem_old, loss_mem_old = self.batch_loss(x_old, y_old, need_grad=False,)
    #
    #     stats_dict.update({'correct_cnt_mem_old': correct_cnt_mem_old,
    #                        "loss_mem_old": loss_mem_old.detach().cpu(),
    #                        "old_task_num": len(y_old), })
    #
    #     x_new = self.incoming_batch['batch_x']
    #     y_new = self.incoming_batch['batch_y']
    #     correct_cnt_mem_new, loss_mem_new = self.batch_loss(x_new, y_new, need_grad=False, )
    #     stats_dict.update({'correct_cnt_mem_new': correct_cnt_mem_new,
    #                        "loss_mem_new": loss_mem_new.detach().cpu(),
    #                        "new_task_num": len(y_new)})
    #
    #     return stats_dict




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




