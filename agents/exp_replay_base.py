import torch
import numpy as np

from agents.base import ContinualLearner

from utils.utils import maybe_cuda, AverageMeter


from utils.utils import cutmix_data
from utils.buffer.buffer_utils import get_grad_vector
import copy
from utils.setup_elements import setup_opt



class ExperienceReplay_base(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_base, self).__init__(model, opt, params)

        self.stats = None

        self.episode_start_test_acc = None
        self.adaptive_ratio = False

        self.evaluator = None
        self.task_seen_so_far = 0
        self.incoming_batch = None
        self.replay_para = None
        self.replay_feature = None





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
    def get_logits(self,batch_x,need_grad):
        if(need_grad):
            logits = self.model.forward(batch_x)
        else:
            with torch.no_grad():
                logits = self.model.forward(batch_x)
        return logits

    def compute_logits_loss(self,x,y,ratio=1,need_grad=True,MEM_FLAG=False):
        do_cutmix = self.params.do_cutmix and np.random.rand(1) < 0.5
        if do_cutmix :
            #print(x.shape)
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.get_logits(x,need_grad)
            loss = lam * self.criterion(logit, labels_a) + (1 - lam) *self.criterion(
                logit, labels_b
            )
            #print(x.shape)

        else:
            logit = self.get_logits(x,need_grad)
            loss = self.criterion(logit, y)
        loss = loss*ratio
        return loss,logit




    def batch_loss(self,batch_x,batch_y,losses_log=None,acc_log=None,need_grad = False,labels=None,ratio = 1.0,loss_reduction_type ="mean",MEM_FLAG=False):
        batch_x = maybe_cuda(batch_x)
        batch_y = maybe_cuda(batch_y)

        loss, logits = self.compute_logits_loss(batch_x, batch_y, ratio, need_grad,MEM_FLAG)


        # if(need_grad):
        #     logits = self.model.forward(batch_x)
        # else:
        #     with torch.no_grad():
        #         logits = self.model.forward(batch_x)
        #
        #
        # if(self.adaptive_ratio == False):
        #
        #
        #     loss = ratio*self.criterion(logits, batch_y, reduction_type= loss_reduction_type)
        # else:
        #     loss = ratio*self.adaptive_criterion(logits, batch_y,ratio)

        if self.params.trick['kd_trick']:
            loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                   self.kd_manager.get_kd_loss(logits, batch_x)
        if self.params.trick['kd_trick_star']:
            loss = 1 / ((self.task_seen + 1) ** 0.5) * loss + \
                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
        correct_cnt,pred_label = self.compute_acc(logits, batch_y)
        # _, pred_label = torch.max(logits, 1)
        labels = self.old_labels+self.new_labels
        for x in pred_label:

            if(x in labels):
                pass
            else:
                print("predict unseen labels",x,labels)
        #
        # correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
        # update tracker
        if(acc_log != None):
            acc_log.update(correct_cnt, batch_y.size(0))
        if(losses_log != None):
            losses_log.update(loss, batch_y.size(0))
        # backward

        return correct_cnt,loss


    ##### cutmix jointtrain####

    # def joint_training(self,replay_para,TEST=False):
    #     iters = replay_para['mem_iter']
    #     mem_ratio = replay_para['mem_ratio']
    #     incoming_ratio = replay_para['incoming_ratio']
    #
    #     batch_x = self.incoming_batch['batch_x']
    #     batch_y = self.incoming_batch['batch_y']
    #     i = self.incoming_batch['batch_num']
    #
    #     stats_dict = None
    #
    #
    #     for j in range(iters):
    #
    #
    #         mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y, self.task_seen)
    #         #
    #         if mem_x.size(0) > 0:
    #             mem_x = maybe_cuda(mem_x, self.cuda)
    #             mem_y = maybe_cuda(mem_y, self.cuda)
    #             # print(batch_x.shape)
    #             # assert False
    #             batch_x = torch.cat([batch_x,mem_x])
    #             batch_y = torch.cat([batch_y,mem_y])
    #         loss, logits = self.compute_logits_loss(batch_x, batch_y,)
    #
    #         # logits = self.model.forward(batch_x)
    #         # loss = self.criterion(logits, batch_y)
    #         self.opt.zero_grad()
    #         loss.backward()
    #         self.opt.step()
    #     return stats_dict


    #### jointrain with one concat
    def set_mem_data(self):
        batch_x = self.incoming_batch['batch_x']
        batch_y = self.incoming_batch['batch_y']
        self.mem_x, self.mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y, self.task_seen)


    def virtual_joint_training(self,replay_para,TEST=False,use_set_mem=False):


        model_temp = copy.deepcopy(self.model)
        opt_temp = setup_opt(self.params.optimizer, model_temp,
                        self.params.learning_rate, self.params.weight_decay)


        return self.joint_training( replay_para, TEST=TEST, model=model_temp,opt =opt_temp,use_set_mem=use_set_mem )

    def joint_training(self,replay_para,TEST=False,model=None,opt=None,use_set_mem=False):
        if(model == None):
            model = self.model
        if(opt ==  None):
            opt = self.opt
        iters = replay_para['mem_iter']
        mem_ratio = replay_para['mem_ratio']
        incoming_ratio = replay_para['incoming_ratio']

        batch_x = self.incoming_batch['batch_x']
        batch_y = self.incoming_batch['batch_y']
        i = self.incoming_batch['batch_num']

        stats_dict = None
        batch_x = maybe_cuda(batch_x, self.cuda)
        batch_y = maybe_cuda(batch_y, self.cuda)
        for j in range(iters):

            #mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
            if(use_set_mem):
                mem_x = self.mem_x
                mem_y = self.mem_y

            else:
                mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y, self.task_seen)

            if mem_x.size(0) > 0:
                mem_x = maybe_cuda(mem_x, self.cuda)
                mem_y = maybe_cuda(mem_y, self.cuda)


                batch_x = torch.cat([batch_x,mem_x])
                batch_y = torch.cat([batch_y,mem_y])
            logits, feature = model.forward(batch_x,feature_flag=True)
            self.replay_feature  = torch.mean(feature,dim = 0).detach().cpu().numpy()
            #torch.max(feature,dim = 0)[0]
            # print(self.replay_feature.shape)
            # assert False


            loss = self.criterion(logits, batch_y,reduction_type="none")

            if(loss.shape[0] == 20):
                loss_incoming=torch.mean(loss[:10]).item()
                loss_mem = torch.mean(loss[10:]).item()

                loss = incoming_ratio * torch.mean(loss[:10])+ mem_ratio * torch.mean(loss[10:])
                _, pred_label = torch.max(logits, 1)
                acc_all = pred_label == batch_y
                acc_incoming=acc_all[:10].sum().item()/10
                acc_mem = acc_all[10:].sum().item() / 10

            else:

                loss = torch.mean(loss)
                _, pred_label = torch.max(logits, 1)
                acc_incoming = (pred_label == batch_y).sum().item() / batch_y.size(0)
                loss_incoming = loss.item()
                acc_mem = None
                loss_mem = None
            opt.zero_grad()
            loss.backward()
            opt.step()

            if(loss_mem != None):



                stats_dict = {'correct_cnt_incoming': acc_incoming,
                              'correct_cnt_mem': acc_mem,
                              "loss_incoming_value": loss_incoming,
                              "loss_mem_value": loss_mem,
                              "batch_num": i,
                              }
            else:
                stats_dict = None

        if(TEST):
            stats_dict = self.compute_test_accuracy(stats_dict,model)


        return stats_dict




    # def joint_training(self,replay_para,TEST=False):
    #
    #
    #
    #     iters = replay_para['mem_iter']
    #     mem_ratio = replay_para['mem_ratio']
    #     incoming_ratio = replay_para['incoming_ratio']
    #
    #     batch_x = self.incoming_batch['batch_x']
    #     batch_y = self.incoming_batch['batch_y']
    #     i = self.incoming_batch['batch_num']
    #
    #     stats_dict = None
    #
    #
    #     for j in range(iters):
    #         ## incoming update
    #         correct_cnt_incoming,loss_incoming = self.batch_loss(batch_x, batch_y, self.losses_batch, self.acc_batch,
    #                                                              ratio = incoming_ratio,need_grad=True)
    #         self.opt.zero_grad()
    #         loss_incoming.backward()
    #         loss_incoming_value = loss_incoming.item()
    #
    #         mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y,self.task_seen)
    #         if(self.task_seen>1 and self.params.replay_old_only):
    #             for y in list(mem_y):
    #                 if y in self.new_labels:
    #                     raise NotImplementedError("saw new class in mem batch")
    #         if mem_x.size(0) > 0: ## the first batch training does not have memory batch
    #             if(self.params.split_new_old):
    #                 n = len(mem_y)
    #                 idx_old = [mem_y[i] in self.old_labels for i in range(n)]
    #                 idx_new = [mem_y[i] in self.new_labels for i in range(n)]
    #                 (x_old, y_old, x_new, y_new) = (mem_x[idx_old], mem_y[idx_old], mem_x[idx_new], mem_y[idx_new])
    #                 if (len(y_new != 0)):
    #                     #print("new", len(y_new))
    #                     correct_cnt_mem_new, loss_mem_new = self.batch_loss(x_new, y_new, need_grad=True,ratio=incoming_ratio)
    #                     loss_mem_new.backward()
    #                     [correct_cnt_mem, loss_mem_value] = [correct_cnt_mem_new, loss_mem_new]
    #                 if (len(y_old) != 0):
    #
    #                     #print("old",len(y_old))
    #                     correct_cnt_mem_old, loss_mem_old = self.batch_loss(x_old, y_old, need_grad=True, ratio=mem_ratio)
    #                     [correct_cnt_mem, loss_mem_value] = [correct_cnt_mem_old, loss_mem_old]
    #                     loss_mem_old.backward()
    #
    #                 stats_dict = {'correct_cnt_incoming': correct_cnt_incoming,
    #                               'correct_cnt_mem': correct_cnt_mem,
    #                               "loss_incoming_value": loss_incoming_value,
    #                               "loss_mem_value": loss_mem_value.detach().cpu(),
    #                               "batch_num": i,
    #                               }
    #
    #             else:
    #
    #                 correct_cnt_mem,loss_mem = self.batch_loss(mem_x, mem_y, self.losses_mem, self.acc_mem, ratio = mem_ratio,need_grad=True,MEM_FLAG=True)
    #                 #self.train_acc_mem.append(correct_cnt_mem)
    #                 loss_mem.backward()
    #                 loss_mem_value = loss_mem.item()
    #                 stats_dict = {'correct_cnt_incoming': correct_cnt_incoming,
    #                               'correct_cnt_mem': correct_cnt_mem,
    #                               "loss_incoming_value": loss_incoming_value,
    #                               "loss_mem_value": loss_mem_value,
    #                               "batch_num":i,
    #                               }
    #                 #stats_dict = self.add_old_new_task_feature(mem_x, mem_y, mem_ratio, stats_dict)
    #
    #         else:
    #             correct_cnt_mem = None
    #             loss_mem_value = None
    #             stats_dict = None
    #
    #         self.opt.step()
    #     if(TEST):
    #         stats_dict = self.compute_test_accuracy(stats_dict)
    #
    #
    #     return stats_dict
    #





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



