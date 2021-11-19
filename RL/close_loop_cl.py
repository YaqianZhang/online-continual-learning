import torch

from sklearn.metrics import confusion_matrix
import numpy as np
class close_loop_cl(object):
    def __init__(self,CL_agent,model,memory_manager):
        self.CL_agent = CL_agent
        self.model=model
        self.memory_manager = memory_manager
        self.test_mem_loss_list= []
        self.test_mem_acc_list = []

        self.task_tloss=[]
        self.task_tloss_train = []
        self.task_tacc=[]
        self.class_acc_stats = None
        self.class_loss_stats = None
        self.train_stats = None
        self.test_stats_prev = None
        self.test_stats= None
    def update_task_reward(self):
        if (len(self.test_mem_loss_list) == 0): return
        self.task_tloss.append(self.test_mem_loss_list[-1])
        #self.task_tloss_train.append(self.last_train_loss)
        if (len(self.test_mem_acc_list) == 0): return
        self.task_tacc.append(self.test_mem_acc_list[-1])
        print("tacc update",self.task_tacc)

    def save_task_reward(self, prefix):

        arr = np.array(self.task_tacc)
        np.save(prefix + "tacc_list.npy", arr)
        print("save tacc!!!!!!!!",arr)

        arr = np.array(self.task_tloss)
        np.save(prefix + "tloss_list.npy", arr)

        arr = np.array(self.task_tloss_train)
        np.save(prefix + "tloss_list_train.npy", arr)

        # arr = np.array(self.influence_score_list)
        # np.save(prefix + "influence_score_list).npy", arr)





    def _most_confused_sample(self,confusion_matrix):
        n = confusion_matrix.shape[0]
        off_diag = confusion_matrix.copy()
        pos_select = []
        for i in range(n):
            off_diag[i,i]=0
            pos_select.append(confusion_matrix[i,i]+1)

        second_best_id = np.argmax(off_diag, axis=0)



        second_best = np.max(off_diag,axis=0)

        pos_select=np.array(pos_select)
        ratio = (pos_select-second_best)/pos_select

        col = np.argmin(ratio)
        row = np.argmax(off_diag[:,col])
        print(col)

        return second_best_id,col,row

    def _low_acc_class(self, confusion_matrix):
        n = confusion_matrix.shape[0]

        acc_class = []
        for i in range(n):
            acc_class.append(confusion_matrix[i, i]/ (np.sum(confusion_matrix[:,i])+1))
        return acc_class

    # def compute_test_accuracy(self, stats_dict, model):
    #     if (stats_dict == None):
    #         stats_dict = {}
    #
    #     if (self.memory_manager.test_buffer.current_index == 0):
    #         print("Test memory is empty")
    #         return None
    #     if (self.params.strict_balance):
    #
    #         test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_strict_blc()  # retrieve_all()
    #     else:
    #
    #         test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
    #     ####### temperature scaling
    #     if (self.params.temperature_scaling):
    #         # print(test_batch_x.shape,test_batch_y.shape)
    #         N, C, W, H = test_batch_x.shape
    #         test_batch_x_reshape = test_batch_x.reshape([N, 1, C, W, H])
    #         test_batch_y_reshape = test_batch_y.reshape([N, 1])
    #
    #         self.scaled_model.set_temperature(zip(test_batch_x_reshape, test_batch_y_reshape))
    #
    #     #### add train stats balanced todo: state 7dim train memory
    #     if (self.params.state_feature_type == "train_test4"):
    #         train_batch_x, train_batch_y = self.memory_manager.buffer.retrieve_class_balance_sample(num_retrieve=100)
    #         train_acc, train_loss, = self.batch_loss(train_batch_x, train_batch_y, need_grad=False, )
    #         # print(train_acc,train_loss)
    #         stats_dict.update({"train_acc": train_acc,
    #                            "train_loss": train_loss.item()})
    #         self.train_acc_blc.append(train_acc)
    #         self.train_loss_blc.append(train_loss.item())
    #
    #     if self.params.joint_replay_type == "together":
    #         stats_dict = self._add_old_new_task_feature_jt(model, test_batch_x, test_batch_y, 1.0,
    #                                                        stats_dict)  ## used for next state
    #
    #     else:
    #         stats_dict = self._add_old_new_task_feature(test_batch_x, test_batch_y, 1.0,
    #                                                     stats_dict)  ## used for next state
    #
    #     # mem_x, mem_y = self.memory_manager.buffer.retrieve(retrieve_num=300)
    #
    #     # train_loss,train_acc = self.evaluate_model(mem_x, mem_y)
    #     if (self.params.state_feature_type[:10] == "new_old6mn"):
    #         # print("!! add more features")
    #
    #         if (self.params.state_feature_type == "new_old6mn_org"):
    #             [train_mem_x, train_mem_y] = [test_batch_x, test_batch_y]
    #             stats_dict['train_loss_old'] = stats_dict['loss_mem_old']
    #         else:
    #             train_mem_x, train_mem_y = self.memory_manager.buffer.random_retrieve(retrieve_num=50)
    #             stats_dict_train = {}
    #             stats_dict_train = self._add_old_new_task_feature(train_mem_x, train_mem_y, 1.0, stats_dict_train,
    #                                                               flag="Train")  ## used for next state
    #             stats_dict['train_loss_old'] = stats_dict_train['loss_mem_old']
    #
    #     return stats_dict

    def set_train_stats(self, acc_incoming=None, acc_mem=None, incoming_loss=None, mem_loss=None, i=None):

        self.train_stats = {'acc_incoming': acc_incoming,
                            'acc_mem': acc_mem,
                            "loss_incoming": incoming_loss,
                            "loss_mem": mem_loss,
                            "batch_num": i,
                            }


    def set_train_stats_scr(self, loss_mem, i, softmax_loss=None, acc_incoming=None, incoming_loss=None):
        self.train_stats = {
            "loss_mem": loss_mem,
            "batch_num": i,
            "softmax_loss": softmax_loss,
            "acc_incoming": acc_incoming,
            "loss_incoming": incoming_loss,
        }

    def set_test_stats(self, acc, loss):
        self.test_stats = {'test_acc': acc,
                           'test_loss': loss,
                           }

    def set_weighted_test_stats(self,batch_y,mem_num,):
        if(self.class_acc_stats == None):
            return
        weighted_mem_acc =0
        weighted_incoming_acc=0
        weighted_mem_loss =0
        weighted_incoming_loss=0

        num_mem = 0
        mem_num_unseen = 0

        num_incoming = 0
        incoming_num_unseen = 0


        y_mem = batch_y[:mem_num]
        y_incoming = batch_y[mem_num:]

        labels = set(y_mem)
        for cls in labels:
            if(cls in self.class_acc_stats.keys()):
                cls_acc = self.class_acc_stats[cls]
                cls_loss = self.class_loss_stats[cls]
                cls_num = ( y_mem== cls ).sum().item()
                weighted_mem_acc += cls_acc * cls_num
                weighted_mem_loss += cls_loss * cls_num
                num_mem += cls_num
            else:
                cls_num = (y_mem == cls).sum().item()
                mem_num_unseen += cls_num
        if(num_mem>0):
            weighted_mem_acc = weighted_mem_acc/num_mem
            weighted_mem_loss = weighted_mem_loss / num_mem
        else:
            weighted_mem_acc=0
            weighted_mem_loss=0


        labels = set(y_incoming)
        for cls in labels:
            if(cls in self.class_acc_stats.keys()):
                cls_acc = self.class_acc_stats[cls]
                cls_loss = self.class_loss_stats[cls]
                cls_num = ( y_incoming== cls ).sum().item()
                weighted_incoming_acc += cls_acc * cls_num
                weighted_incoming_loss += cls_loss * cls_num
                num_incoming += cls_num
            else:
                cls_num = (y_incoming == cls).sum().item()
                incoming_num_unseen += cls_num
        if (num_incoming > 0):
            weighted_incoming_acc = weighted_incoming_acc/num_incoming
            weighted_incoming_loss = weighted_incoming_loss / num_incoming
        else:
            weighted_incoming_acc=0
            weighted_incoming_loss=0


        self.weighted_test_stats={
            "weighted_mem_acc": weighted_mem_acc,
            "weighted_incoming_acc":weighted_incoming_acc,
            "weighted_mem_loss": weighted_mem_loss,
            "weighted_incoming_loss": weighted_incoming_loss,
            "mem_num_unseen":mem_num_unseen,
            "incoming_num_unseen":incoming_num_unseen,

        }

    def set_class_stats(self,pred_label,test_batch_y,loss_full):

        labels = set(test_batch_y)
        self.class_acc_stats = {}
        self.class_loss_stats = {}
        for l in labels:
            idx = test_batch_y == l
            acc = pred_label[idx] == test_batch_y[idx].sum().item() /test_batch_y[idx].shape[0]
            loss = torch.mean(loss_full[idx]).item()
            self.class_acc_stats[l] = acc
            self.class_loss_stats[l] = loss

    def compute_testmem_loss(self, ):


        if (self.memory_manager.test_buffer.current_index == 0):
            print("Test memory is empty")
            return None

        test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
        logits = self.CL_agent._compute_softmax_logits(test_batch_x, need_grad=False)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, test_batch_y)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == test_batch_y).sum().item() / test_batch_x.shape[0]
        loss = torch.mean(softmax_loss_full).item()

        self.set_class_stats( pred_label, test_batch_y, softmax_loss_full)

        n = len(test_batch_y)
        idx_old = [test_batch_y[i] in self.CL_agent.old_labels for i in range(n)]
        idx_new = [test_batch_y[i] in self.CL_agent.new_labels for i in range(n)]


        old_num = len(test_batch_y[idx_old])
        new_num = len(test_batch_y[idx_new])
        if(old_num >0):

            test_loss_old = torch.mean(softmax_loss_full[idx_old]).item()
            test_acc_old = (pred_label[idx_old] == test_batch_y[idx_old]).sum().item() / len(test_batch_y[idx_old])
        else:
            test_loss_old=0
            test_acc_old = 0

        if(new_num >0):
            test_acc_new = (pred_label[idx_new] == test_batch_y[idx_new]).sum().item() / len(test_batch_y[idx_new])
            test_loss_new = torch.mean(softmax_loss_full[idx_new]).item()
        else:
            test_loss_new=0
            test_acc_new = 0

        if self.CL_agent.params.close_loop_mem_type == "low_acc":
            label_set = set(test_batch_y.cpu().detach().numpy()) | set(pred_label.cpu().detach().numpy())
            label = sorted(label_set, reverse=False)
            c_matrix = confusion_matrix(test_batch_y.detach().cpu(),
                                        pred_label.detach().cpu()).T
            acc_class =  self._low_acc_class( c_matrix)
            selected_num = 5 #min(10,int(c_matrix.shape[0]/5))
            selected_idx = np.argsort(acc_class)[:selected_num+1]

            self.CL_agent.low_acc_classes = [label[idx] for idx in selected_idx ]




        self.test_mem_loss_list.append(loss)
        self.test_mem_acc_list.append(acc)


        self.test_stats = {'test_acc': acc,
                           'test_loss': loss,
                           'test_acc_new':test_acc_new,
                           'test_loss_new':test_loss_new,
                           'test_acc_old': test_acc_old,
                           'test_loss_old': test_loss_old,
                           }

        self.CL_agent.test_acc_mem.append(acc)
        self.CL_agent.test_loss_mem.append(loss)



        return acc,loss



    def compute_real_test_loss(self,model):
        acc, loss =  self.CL_agent.evaluator.evaluate_model(model,self.CL_agent.task_seen,return_loss=True)
        return acc, loss




    # def compute_test_with_virtual_update(self):
    #     grad_dims = []
    #     for param in self.model.parameters():
    #         grad_dims.append(param.data.numel())
    #     grad_vector = get_grad_vector(self.model.parameters, grad_dims)
    #     model_temp = self.get_future_step_parameters(self.model, grad_vector, grad_dims, lr)
    #     if (self.params.online_hyper_valid_type == "real_data"):
    #         acc, loss = self.evaluator.evaluate_model(model_temp, self.task_seen, return_loss=True)
    #
    #     else:
    #         acc, loss = self.compute_acc_data_model(model_temp, test_batch_x, test_batch_y)

