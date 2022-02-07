from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from torch.utils import data
from utils.utils import maybe_cuda, AverageMeter
import torch

class RL_EWC(ContinualLearner):
    def __init__(self, model, opt, params):
        super(RL_EWC, self).__init__(model, opt, params)


    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        # set up model
        self.model.train()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                #self.lambda_

                # update the running fisher
                if (ep * len(train_loader) + i + 1) % self.fisher_update_after == 0:
                    self.update_running_fisher()

                out = self.forward(batch_x)
                loss = self.total_loss(out, batch_y)
                if self.params.trick['kd_trick']:
                    loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(out, batch_x)
                if self.params.trick['kd_trick_star']:
                    loss = 1 / ((self.task_seen + 1) ** 0.5) * loss + \
                           (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(out, batch_x)
                # update tracker
                losses_batch.update(loss.item(), batch_y.size(0))
                _, pred_label = torch.max(out, 1)
                acc = (pred_label == batch_y).sum().item() / batch_y.size(0)
                acc_batch.update(acc, batch_y.size(0))
                # backward
                self.opt.zero_grad()
                loss.backward()

                # accumulate the fisher of current batch
                self.accum_fisher()
                self.opt.step()

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )

        # save params for current task
        for n, p in self.weights.items():
            self.prev_params[n] = p.clone().detach()

        # update normalized fisher of current task
        max_fisher = max([torch.max(m) for m in self.running_fisher.values()])
        min_fisher = min([torch.min(m) for m in self.running_fisher.values()])
        for n, p in self.running_fisher.items():
            self.normalized_fisher[n] = (p - min_fisher) / (max_fisher - min_fisher + 1e-32)
        self.after_train()



