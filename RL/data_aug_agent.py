
import torch
import numpy as np
from utils.utils import cutmix_data,cutmix_data_two_data
from utils.utils import maybe_cuda

class data_aug_agent(object):

    def __init__(self):
        pass

    def perform_cutmix(self, x, y):
        ce = torch.nn.CrossEntropyLoss(reduction='mean')

        index="None"



        ## todo : cutmix
        do_cutmix = self.params.do_cutmix and np.random.rand(1) < self.params.cutmix_prob
        if do_cutmix:
            # print(x.shape)

            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0,index=index)
            logits = self._compute_softmax_logits(x)
            # h_feature = self.model.features(x)
            # logits = self.softmax_head(h_feature)
            softmax_loss = lam * ce(logits, labels_a) + (1 - lam) * ce(
                logits, labels_b
            )
            self.softmax_opt.zero_grad()
            softmax_loss.backward()
            self.softmax_opt.step()

    #### from ER
    def cutmix_softmax_training(self,x,y,mem_num):
        # if(self.params.use_softmaxloss == False):
        #
        #     return

        do_cutmix = self.params.do_cutmix and np.random.rand(1) < self.params.cutmix_prob
        if(do_cutmix == False):
            return


        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        total_num = x.shape[0]

        index="None"
        #print("du_cutmix", self.params.cutmix_prob, self.params.softmax_membatch , mem_num)
        #if(self.params.softmax_membatch <= mem_num):
            # selected_idx = np.random.shuffle(np.arange(0,mem_num))[:self.params.softmax_membatch]
            #
            # #selected_idx = np.random.randint(0,mem_num,self.params.softmax_membatch)
            # selected_idx = list(selected_idx)+list(np.arange(mem_num,total_num))
            # # print(mem_num,selected_idx)
            # # assert False
            # x = x[selected_idx]
            # y = y[selected_idx]
        if(self.params.cutmix_type == "random"):
            selected_idx = np.random.randint(0,len(y),self.params.cutmix_batch)
            x = x[selected_idx]
            y = y[selected_idx]
        if (self.params.cutmix_type == "cross_task"):
            incoming_num = len(y)-mem_num
            index_incoming = np.random.randint(0,mem_num,incoming_num)
            index_mem = np.random.randint(mem_num, len(y), mem_num)
            index = torch.from_numpy(np.concatenate((index_mem,index_incoming)))
            # print(index)
            # assert False
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0,index=index)
        elif(self.params.cutmix_type == "train_mem"):
            train_batch_x, train_batch_y = self.memory_manager.buffer.retrieve_class_balance_sample(num_retrieve=20)
            train_batch_x = maybe_cuda(train_batch_x)
            train_batch_y = maybe_cuda(train_batch_y)
            x = torch.cat((train_batch_x,x[-10:]))
            y = torch.cat((train_batch_y,y[-10:]))
            # print(x.shape,y)
            # assert False
            #x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0, index=index)
        elif (self.params.cutmix_type == "most_confused" and self.col != None):
            num_retrieve = 10
            x,y = self.memory_manager.buffer.retrieve_class_num([self.col], num_retrieve)
            x_b,y_b = self.memory_manager.buffer.retrieve_class_num([self.row], num_retrieve)
            x, labels_a, labels_b, lam = cutmix_data_two_data(x,y,x_b,y_b, alpha=1.0, )

        elif(self.params.cutmix_type == "mixed"  and self.mix_label_pair != None):


            index =[]

            y_numpy = y.detach().cpu().numpy()
            for l in y_numpy:


                if(l in self.mix_label_pair.keys()):
                    mixed_label = self.mix_label_pair[l]
                    idx = np.arange(len(y))[y_numpy==mixed_label]
                    if(len(idx)>0):
                        index.append(np.random.choice(idx,1))

                    else:
                        index.append(np.random.randint(0, len(y), 1))
                        #print("random")
                else:
                    index.append(np.random.randint(0,len(y), 1))
                    #print("random")

            index = torch.from_numpy(np.array(index).reshape([-1]))
           #s print(index.shape)

        else:
            print("wrong")
            # x, labels_a, labels_b, lam = cutmix_data_two_data(x,y,)
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0, index=index)


        logits = self.model.forward(x)
        # h_feature = self.model.features(x)
        # logits = self.softmax_head(h_feature)
        softmax_loss = lam * ce(logits, labels_a) + (1 - lam) * ce(
            logits, labels_b)


        self.opt.zero_grad()
        softmax_loss.backward()
        self.opt.step()

    def most_confused_test_mem(self,test_batch_y,pred_label):
        if(self.CL_agent.params.cutmix_type == "mixed"):
            label_set = set(test_batch_y.cpu().detach().numpy()) | set(pred_label.cpu().detach().numpy())


            c_matrix = confusion_matrix(test_batch_y.detach().cpu(), pred_label.detach().cpu())

            second,col,row = self._most_confused_sample(c_matrix.T )

            label = sorted(label_set,reverse = False)
            #print(len(label), c_matrix.shape, )
            # mixed = [label[i] for i in second]
            # #print(label,mixed)
            # self.mix_label_pair = [label,mixed]
            self.CL_agent.mix_label_pair={}
            for j,l in enumerate(label):
                self.CL_agent.mix_label_pair[l]=label[second[j]]
        elif (self.CL_agent.params.cutmix_type == "most_confused"):
            label_set = set(test_batch_y.cpu().detach().numpy()) | set(pred_label.cpu().detach().numpy())
            label = sorted(label_set, reverse=False)

            c_matrix = confusion_matrix(test_batch_y.detach().cpu(), pred_label.detach().cpu())

            second,col,row = self._most_confused_sample(c_matrix.T )

            label = sorted(label_set,reverse = False)
            self.CL_agent.col = label[col]
            self.CL_agent.row = label[row]
