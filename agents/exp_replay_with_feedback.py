
from agents.exp_replay_base import ExperienceReplay_base


class ExperienceReplay_eval(ExperienceReplay_base):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_eval, self).__init__(model, opt, params)



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
        if(self.params.state_feature_type == "new_old6mn" or self.params.state_feature_type == "new_old6mnt" ):
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









