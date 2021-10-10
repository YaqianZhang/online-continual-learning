
import torch

class hyper_tune(object):
    def __init__(self):
        pass

    def hyperparameter_tune(self,lr_list_type="basic"):

        lr_list_dict ={
            "basic": [0.01, 0.1, 0.2],
        "4lr":[0.001,0.01, 0.1, 0.2],
        "5lr":[0.001,0.01, 0.1, 0.2,0.5],
            "scr":[0.001,0.01, 0.1, 0.5],
        }
        lr_list=lr_list_dict[lr_list_type]
        if (self.params.save_prefix_tmp == "debug"):
            return 0.01
        if (self.params.save_prefix_tmp == "rnd"):
            # acc_list=[1,1,1]
            # prob = np.exp(acc_list) / sum(np.exp(acc_list))

            id = np.random.choice(len(lr_list), 1, )[0]
            selected_para = lr_list[id]
        else:

            grad_dims = []
            for param in self.model.parameters():
                grad_dims.append(param.data.numel())
            grad_vector = get_grad_vector(self.model.parameters, grad_dims)
            max_acc = -1
            acc_list = []
            loss_list = []
            if (self.params.online_hyper_valid_type != "real_data"):
                test_batch_x, test_batch_y = self.memory_manager.test_buffer.retrieve_all()  # retrieve_all()
                # if (self.params.test_mem_type == "before"):
                #     print(test_batch_x.shape, test_batch_y.shape)
            for lr in lr_list:  # [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]:
                model_temp = self.get_future_step_parameters(self.model, grad_vector, grad_dims, lr)
                if(self.params.online_hyper_valid_type == "real_data"):
                    acc,loss = self.evaluator.evaluate_model(model_temp,self.task_seen,return_loss=True)

                else:
                    acc, loss = self.compute_acc_data_model(model_temp, test_batch_x, test_batch_y)
                if (acc > max_acc):
                    best_para = lr
                    max_acc = acc
                if (self.params.save_prefix_tmp == "reverse"):
                    acc_list.append(-acc.item())
                else:
                    acc_list.append(acc.item())

                loss_list.append(loss.item())

            # prob = np.exp(acc_list) / sum(np.exp(acc_list))
            #
            # id= np.random.choice(3, 1,p=prob)[0]
            # selected_para = lr_list[id]
            selected_para = best_para

        self.params.learning_rate = selected_para
        # print("!! best lr",best_para,"selected_lr",selected_para,)#acc_list,)
        final_lr = selected_para
        self.adaptive_learning_rate.append(final_lr)

        return final_lr

        # self.opt = torch.optim.SGD(self.model.parameters(),
        #                         lr=best_para,
        #                         weight_decay=self.params.weight_decay)
        #     #setup_opt(self.params.optimizer, model, params.learning_rate, params.weight_decay)

    ####SCR overide Todo: scr hyper tune; compute logits
    def compute_acc_data_model(self, model, x, y):
        with torch.no_grad():
            h_feature = model.features(x)
            logits = self.softmax_head(h_feature)
            ce = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = ce(logits, y)
            _, pred_label = torch.max(logits, 1)
            acc = (pred_label == y).sum() / len(y)
        return acc, loss

    def compute_acc_data_model(self, model, x, y):
        with torch.no_grad():
            logits = model.forward(x)
            loss = self.criterion(logits, y, reduction_type="mean")
            _, pred_label = torch.max(logits, 1)
            acc = (pred_label == y).sum() / len(y)
        return acc, loss

    def get_future_step_parameters(self, model, grad_vector, grad_dims, lr):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - lr * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1


    def _scr_train_with_hp(self,batch_x, batch_y, losses):
        mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
        mem_x, mem_y = self.memory_manager.retrieve_from_mem(batch_x, batch_y, self.task_seen)

        if mem_x.size(0) > 0:


            mem_x = maybe_cuda(mem_x, self.cuda)
            mem_y = maybe_cuda(mem_y, self.cuda)
            combined_batch = torch.cat((mem_x, batch_x))
            combined_labels = torch.cat((mem_y, batch_y))

            ######## scr loss
            if(self.params.no_aug):
                combined_batch_aug = combined_batch.clone()
            else:


                combined_batch_aug = self.transform(combined_batch)

            features = torch.cat([self.model.forward(combined_batch).unsqueeze(1),
                                  self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
            loss, loss_full = self.criterion(features, combined_labels)

            losses.update(loss, batch_y.size(0))
            if (self.params.online_hyper_RL):
                self.RL_replay.set_train_stats_scr(loss, i)

            self.opt.zero_grad()
            loss.backward()
            #loss_cutmix.backward()
            if (self.params.online_hyper_RL):
                final_lr = self.RL_replay.make_replay_decision_scr()
                self.params.learning_rate = final_lr
                for g in self.opt.param_groups:
                    g['lr'] = final_lr
                self.adaptive_learning_rate.append(final_lr)
            if (i % self.params.online_hyper_freq == 0 and self.params.online_hyper_tune):
                #final_lr = self.hyperparameter_tune()
                final_lr = self.hyperparameter_tune(self.params.online_hyper_lr_list_type)

                # final_lr = 0.01
                self.params.learning_rate = final_lr
                # self.adaptive_learning_rate.append(final_lr)

                for g in self.opt.param_groups:
                    g['lr'] = final_lr

            self.opt.step()
            self.loss_batch.append(loss.item())


            #### softmax loss

            self.perform_softmax_training(combined_batch, combined_labels, mem_x.shape[0])


            if (self.params.online_hyper_RL):
                acc, loss = self.close_loop_cl.compute_testmem_loss()
                self.test_acc_mem.append(acc)
                self.test_loss_mem.append(loss)
                self.RL_replay.set_test_stats(acc, loss)
                self.RL_replay.set_reward()




                ############## from ER batch update

                # self.cutmix_softmax_training(batch_x,batch_y,mem_num)

                # if(self.params.use_test_buffer):
                #     acc, loss = self.close_loop_cl.compute_testmem_loss()
                # if(self.params.online_hyper_RL):
                #
                #
                #     acc, loss = self.close_loop_cl.compute_testmem_loss()
                #     self.test_acc_mem.append(acc)
                #     self.test_loss_mem.append(loss)
                #     self.RL_replay.set_test_stats(acc,loss)
                #     self.RL_replay.set_reward()

                # if(self.params.online_hyper_RL):
                #     final_lr = self.RL_replay.make_replay_decision()
                #     self.params.learning_rate = final_lr
                #     for g in self.opt.param_groups:
                #         g['lr'] = final_lr
                #     self.adaptive_learning_rate.append(final_lr)
                # if(i%self.params.online_hyper_freq == 0 and self.params.online_hyper_tune):
                #     final_lr = self.hyperparameter_tune(self.params.online_hyper_lr_list_type)
                #
                #     self.params.learning_rate = final_lr
                #     #self.adaptive_learning_rate.append(final_lr)
                #
                #     for g in self.opt.param_groups:
                #         g['lr'] = final_lr