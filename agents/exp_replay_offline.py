import torch
from torch.utils import data
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.utils import maybe_cuda, AverageMeter
import torch

from utils.setup_elements import transforms_match, input_size_match






class ExperienceReplay_offline(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay_offline, self).__init__(model, opt, params)
        #self.buffer = Buffer(model, params)





    def _batch_update(self,batch_x,batch_y,losses_batch,acc_batch,i,replay_para=None,mem_num=0):


        logits = self.model.forward(batch_x)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == batch_y)

        ce_all = torch.nn.CrossEntropyLoss(reduction='none')
        softmax_loss_full = ce_all(logits, batch_y)

        total_num = batch_x.shape[0]
        avrg_acc = acc.sum().item() / total_num
        loss = torch.mean(softmax_loss_full)


        acc_batch.update(avrg_acc, batch_y.size(0))
        losses_batch.update(loss.item(), batch_y.size(0))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss_batch.append(loss.item())






    def train_learner(self, x_train, y_train):


        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)



                self._batch_update(batch_x,batch_y, losses_batch, acc_batch, i)


                if i % 10 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )

        self.after_train()





