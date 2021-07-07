import torch
import numpy as np
from utils.utils import maybe_cuda

class evaluator(object):
    def __init__(self,test_loaders):
        self.test_loaders=test_loaders

    def evaluate_model(self,model,task_seen):
        acc_list=[]
        num_batch = 1
        for task, test_loader in enumerate(self.test_loaders):

            if(task> task_seen):break ## task so far

            random_i = np.random.randint(0,len(test_loader))
            for i, (batch_x, batch_y) in enumerate(test_loader):
                if(i>=random_i): break


            batch_x = maybe_cuda(batch_x, )
            batch_y = maybe_cuda(batch_y, )
            with torch.no_grad():

                logits = model.forward(batch_x)
                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                acc_list.append(correct_cnt)

            # for i, (batch_x, batch_y) in enumerate(test_loader):
            #     if(i>=num_batch): break
            #     #print("real reward",batch_x.shape)
            #     batch_x = maybe_cuda(batch_x, )
            #     batch_y = maybe_cuda(batch_y, )
            #     with torch.no_grad():
            #
            #         logits = model.forward(batch_x)
            #         _, pred_label = torch.max(logits, 1)
            #         correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)
            #         acc_list.append(correct_cnt)
        return np.mean(np.array(acc_list))




