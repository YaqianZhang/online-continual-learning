import torch
from torch.utils import data

from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter

import numpy as np

from RL.RL_replay_base import RL_replay
from RL.RL_replay_MAB import RL_replay_MAB

from RL.close_loop_cl import close_loop_cl
from agents.scr import SupContrastReplay

class SCR_RL_iter(SupContrastReplay):
    def __init__(self, model, opt, params):
        super(SCR_RL_iter, self).__init__(model, opt, params)

        if(self.params.online_hyper_RL or self.params.scr_memIter ):
            if(self.params.scr_memIter_type == "MAB"):
                self.RL_replay = RL_replay_MAB(params, )
            else:
                self.RL_replay = RL_replay(params,)
            self.close_loop_cl = close_loop_cl(self,model,self.memory_manager)




    def train_learner(self, x_train, y_train):
        self.memory_manager.reset_new_old()
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x, batch_y = self.memory_manager.update_before_training(batch_x, batch_y)

                scr_loss,combined_batch,combined_labels,mem_num  = self.perform_scr_update(batch_x, batch_y, losses)

                softmax_loss, acc_incoming, incoming_loss= self.perform_softmax_update(combined_batch,
                                                                                          combined_labels,
                                                                                          mem_num)

                more_iters = -2
                if(loss != None):
                    ### compute state
                    self.RL_replay.set_train_stats_scr(loss, i,softmax_loss,acc_incoming,incoming_loss )
                    test_acc, test_loss = self.close_loop_cl.compute_testmem_loss()
                    self.RL_replay.set_test_stats(test_acc,test_loss)

                    ### get action
                    #for j in range(self.mem_iters):
                    more_iters = self.RL_replay.make_replay_decision()
                    for add_iter in range(more_iters):
                        #loss = self._scr_train(batch_x, batch_y, losses)
                        scr_loss,combined_batch, combined_labels, mem_num = self.perform_scr_update(batch_x, batch_y, losses)

                        loss, softmax_loss, acc_incoming, incoming_loss = self.perform_softmax_update(combined_batch,
                                                                                                        combined_labels,
                                                                                                        mem_num)
                    ## compute reward
                    test_acc, test_loss = self.close_loop_cl.compute_testmem_loss()
                    self.test_acc_mem.append(test_acc)
                    self.test_loss_mem.append(test_loss)
                    #self.RL_replay.set_test_stats(test_acc, test_loss)
                    self.RL_replay.set_reward()
                else:
                    more_iters = -1


                self.memory_manager.update_memory(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                            .format(i, losses.avg(), acc_batch.avg())
                    )
                    #print("lr", self.params.learning_rate)
                    print("add_memIter", more_iters)



        self.after_train()

        # def compute_testmem_loss(self, ):
        #
        #     if (self.memory_manager.test_buffer.current_index == 0):
        #         print("Test memory is empty")
        #         return None
        #
        #     test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()
        #     #loss,_ = self.perform_scr(test_batch_x, test_batch_y)
        #     logits = self._compute_softmax_logits(test_batch_x,need_grad = False)
        #
        #     ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        #     softmax_loss_full = ce_all(logits, test_batch_y)
        #     _, pred_label = torch.max(logits, 1)
        #     acc = (pred_label == test_batch_y).sum().item()/test_batch_x.shape[0]
        #     self.test_acc_mem.append(acc)
        #     self.test_loss_mem.append(torch.mean(softmax_loss_full).item())
        #     n = len(test_batch_y)
        #
        #     idx_old = [test_batch_y[i] in self.old_labels for i in range(n)]
        #     idx_new = [test_batch_y[i] in self.new_labels for i in range(n)]
        #     correctness = pred_label == test_batch_y
        #     pred_old_new = [pred_label[i] in self.new_labels for i in range(n)]
        #     true_old_new = [test_batch_y[i] in self.new_labels for i in range(n)]
        #     old_acc = correctness[idx_old].sum().item()/np.sum(idx_old)
        #     new_acc = correctness[idx_new].sum().item()/np.sum(idx_new)
        #
        #     correctness_old_new = np.array(pred_old_new)==np.array(true_old_new)
        #     new_old_classification = np.mean(correctness_old_new)
        #     print(old_acc,new_acc, new_old_classification)
