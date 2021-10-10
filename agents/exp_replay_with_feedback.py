
from agents.exp_replay_base import ExperienceReplay_base
import numpy as np
import torch
from RL.temperature_scaling import ModelWithTemperature
from utils.buffer.buffer_utils import get_grad_vector
import copy


class ExperienceReplay_eval(ExperienceReplay_base):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_eval, self).__init__(model, opt, params)
        if (self.params.temperature_scaling):
            self.scaled_model = ModelWithTemperature(model)


    # def hyperparameter_tune(self):
    #     grad_dims = []
    #     for param in self.model.parameters():
    #         grad_dims.append(param.data.numel())
    #     grad_vector = get_grad_vector(self.model.parameters, grad_dims)
    #     max_acc = -1
    #     for lr in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
    #         model_temp = self.get_future_step_parameters(self.model, grad_vector, grad_dims,lr)
    #         test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
    #         acc,loss = self.compute_acc_data_model(model_temp,test_batch_x,test_batch_y)
    #         if(acc>max_acc):
    #             best_para =  lr
    #             max_acc = acc
    #     self.params.learning_rate = best_para


    def compute_acc_data_model(self,model,x,y):
        logits = model.forward(x)
        loss = self.criterion(logits, y, reduction_type="mean")
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == y)/len(y)
        return acc,loss



    ##### concat implementation
    def _add_old_new_task_feature_jt(self,model, batch_x, batch_y, mem_ratio, stats_dict, flag=None):
        if (self.params.temperature_scaling):

            logits = self.scaled_model.forward(batch_x)
        else:
            logits = model.forward(batch_x)
        loss = self.criterion(logits, batch_y, reduction_type="none")
        _, pred_label = torch.max(logits, 1)
        acc_all = pred_label == batch_y

        n = len(batch_y)

        idx_old = [batch_y[i] in self.old_labels for i in range(n)]
        idx_new = [batch_y[i] in self.new_labels for i in range(n)]
        if(len(acc_all[idx_new]) == 0):
            return None

        loss_test_old = torch.mean(loss[idx_old]).item()
        loss_test_new = torch.mean(loss[idx_new]).item()
        acc_test_old = acc_all[idx_old].sum().item()/len(acc_all[idx_old])
        acc_test_new = acc_all[idx_new].sum().item()/len(acc_all[idx_new])
        test_loss = torch.mean(loss).item()
        acc_overall = acc_all.sum().item()

        stats_dict.update({'correct_cnt_mem_new': acc_test_new,
                           "loss_mem_new": loss_test_new,
                           "new_task_num": len(batch_y[idx_new]),
                        'correct_cnt_mem_old': acc_test_old,
                        "loss_mem_old": loss_test_old,
                        "old_task_num": len(batch_y[idx_old])}
                          )
        stats_dict['correct_cnt_test_mem'] = acc_overall/len(acc_all)
        stats_dict['loss_test_value'] = test_loss  ## used for reward
        stats_dict['acc_test']=acc_overall
        return stats_dict

    def _add_old_new_task_feature(self, mem_x, mem_y, mem_ratio, stats_dict,flag=None):

        n = len(mem_y)

        idx_old = [mem_y[i] in self.old_labels for i in range(n)]
        idx_new = [mem_y[i] in self.new_labels for i in range(n)]

        (x_old, y_old, x_new, y_new) = (mem_x[idx_old], mem_y[idx_old], mem_x[idx_new], mem_y[idx_new])
        if (len(y_old) != 0 ):
            if(self.params.reward_type == "test_loss_median"):
                correct_cnt_mem_old, loss_mem_old_list = self.batch_loss(x_old, y_old, need_grad=False, ratio=mem_ratio,
                                                                         loss_reduction_type="none")
                loss_mem_old_list = loss_mem_old_list.detach().cpu().numpy()
                loss_mem_old = np.mean(loss_mem_old_list)

                stats_dict.update({'correct_cnt_mem_old': correct_cnt_mem_old,
                                   "loss_mem_old": loss_mem_old,
                                   "old_task_num": len(y_old), })
            else:
                correct_cnt_mem_old, loss_mem_old = self.batch_loss(x_old, y_old, need_grad=False, ratio=mem_ratio,
                                                                     )

                stats_dict.update({'correct_cnt_mem_old': correct_cnt_mem_old,
                                   "loss_mem_old": loss_mem_old.detach().cpu(),
                                   "old_task_num": len(y_old), })

        else:
            raise NotImplementedError(" no old classes in test mem!")

        if (len(y_new != 0)):
            # if(flag == "Train" ):
            #     #print()
            #     print("found new class in train mem!", mem_y[idx_new])

                #raise NotImplementedError("found new class in train mem!",mem_y[idx_new])
            if(self.params.reward_type == "test_loss_median"):
                correct_cnt_mem_new, loss_mem_new_list = self.batch_loss(x_new, y_new, need_grad=False, ratio=mem_ratio,
                                                                loss_reduction_type="none")
                loss_mem_new_list = loss_mem_new_list.detach().cpu().numpy()
                loss_mem_new = np.mean(loss_mem_new_list)
                stats_dict.update({ 'correct_cnt_mem_new': correct_cnt_mem_new,
                                   "loss_mem_new": loss_mem_new.detach().cpu(),
                                   "new_task_num": len(y_new)})
            else:
                correct_cnt_mem_new, loss_mem_new = self.batch_loss(x_new, y_new, need_grad=False, ratio=mem_ratio,
                                                           )


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

            if (self.params.reward_type == "test_loss_median"):
                loss_median_old = np.median(loss_mem_old_list)
                loss_median_new = np.median(loss_mem_new_list)
                _,loss_median = self._compute_avg(correct_cnt_mem_old, loss_median_old,len(y_old),
                                                   correct_cnt_mem_new, loss_median_new ,len(y_new))

                stats_dict['loss_median']=loss_median
        else:
            [test_acc, test_loss] = [correct_cnt_mem_old, loss_mem_old,]
            if (self.params.reward_type == "test_loss_median"):
                loss_median = np.median(np.array(loss_mem_old_list))
                stats_dict['loss_median']=loss_median
            #print(rwd,test_loss)#.detach().cpu())





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



    def compute_test_accuracy(self, stats_dict,model):
        if(stats_dict == None):
            stats_dict={}


        if (self.memory_manager.test_buffer.current_index == 0):
            print("Test memory is empty")
            return None
        if(self.params.strict_balance):


            test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_strict_blc()#retrieve_all()
        else:

            test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
        ####### temperature scaling
        if(self.params.temperature_scaling):
            #print(test_batch_x.shape,test_batch_y.shape)
            N,C,W,H = test_batch_x.shape
            test_batch_x_reshape = test_batch_x.reshape([N,1,C,W,H])
            test_batch_y_reshape = test_batch_y.reshape([N, 1])

            self.scaled_model.set_temperature(zip(test_batch_x_reshape,test_batch_y_reshape))

        #### add train stats balanced todo: state 7dim train memory
        if(self.params.state_feature_type == "train_test4"):
            train_batch_x,train_batch_y = self.memory_manager.buffer.retrieve_class_balance_sample(num_retrieve=100)
            train_acc,train_loss,  = self.batch_loss(train_batch_x, train_batch_y, need_grad=False,)
            #print(train_acc,train_loss)
            stats_dict.update({"train_acc":train_acc,
                               "train_loss":train_loss.item()})
            self.train_acc_blc.append(train_acc)
            self.train_loss_blc.append(train_loss.item())

        if self.params.joint_replay_type  == "together":
            stats_dict = self._add_old_new_task_feature_jt(model,test_batch_x, test_batch_y, 1.0, stats_dict) ## used for next state

        else:
            stats_dict = self._add_old_new_task_feature(test_batch_x, test_batch_y, 1.0,
                                                        stats_dict)  ## used for next state

        #mem_x, mem_y = self.memory_manager.buffer.retrieve(retrieve_num=300)

        #train_loss,train_acc = self.evaluate_model(mem_x, mem_y)
        if(self.params.state_feature_type[:10] == "new_old6mn" ):
           # print("!! add more features")

            if (self.params.state_feature_type == "new_old6mn_org"):
                [train_mem_x, train_mem_y]= [test_batch_x, test_batch_y ]
                stats_dict['train_loss_old'] = stats_dict['loss_mem_old']
            else:
                train_mem_x, train_mem_y = self.memory_manager.buffer.random_retrieve(retrieve_num=50)
                stats_dict_train={}
                stats_dict_train = self._add_old_new_task_feature(train_mem_x, train_mem_y, 1.0, stats_dict_train,flag="Train")  ## used for next state
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









