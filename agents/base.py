from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy


class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.initialize()
    def initialize(self):
        print("base agent initialize")
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def after_train(self):
        #self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            self.model.train()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # mem_x = maybe_cuda(mem_x)
            # mem_y = maybe_cuda(mem_y)
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters() if p.requires_grad]
                        grad = [p.grad.clone()/10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        else:
            return ce(logits, labels)
    def adaptive_criterion(self,logits,labels,ratio):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='none')
        print(ce,ratio)
        assert False
        return torch.mean(ce(logits,labels)*ratio)

    def forward(self, x):
        return self.model.forward(x)

    def evaluate(self, test_loaders):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        loss_array = np.zeros(len(test_loaders))
        if self.params.trick['nmc_trick'] or self.params.agent == 'ICARL':
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                x = maybe_cuda(x)
                y = maybe_cuda(y)
                cls_exemplar[y.item()].append(x)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y
        with torch.no_grad():
            if self.params.error_analysis:
                error = 0
                no = 0
                nn = 0
                oo = 0
                on = 0
                new_class_score = AverageMeter()
                old_class_score = AverageMeter()
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                loss = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    if self.params.trick['nmc_trick'] or self.params.agent == 'ICARL':
                        feature = self.model.features(batch_x)  # (batch_size, feature_size)
                        for j in range(feature.size(0)):  # Normalize
                            feature.data[j] = feature.data[j] / feature.data[j].norm()
                        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                        means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)
                        means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                        means = means.transpose(1, 2)
                        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                        _, preds = dists.min(1)
                        correct_cnt = (np.array(self.old_labels)[
                                           preds.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)

                        ## todo:zyq how to compute loss for icarl
                        logits = self.model.forward(batch_x)
                        _, pred_label = torch.max(logits, 1)
                        loss_batch = self.criterion(logits, batch_y)


                    else:
                        logits = self.model.forward(batch_x)
                        _, pred_label = torch.max(logits, 1)
                        correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)
                        loss_batch = self.criterion(logits, batch_y)

                        if self.params.error_analysis:
                            if task < self.task_seen-1:
                                # old test
                                total = (pred_label != batch_y).sum().item()
                                wrong = pred_label[pred_label != batch_y]
                                error += total
                                on_tmp = sum([(wrong == i).sum().item() for i in self.new_labels_zombie])
                                oo += total - on_tmp
                                on += on_tmp
                                old_class_score.update(logits[:, list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item(), batch_y.size(0))
                            elif task == self.task_seen -1:
                                # new test
                                total = (pred_label != batch_y).sum().item()
                                error += total
                                wrong = pred_label[pred_label != batch_y]
                                no_tmp = sum([(wrong == i).sum().item() for i in list(set(self.old_labels) - set(self.new_labels_zombie))])
                                no += no_tmp
                                nn += total - no_tmp
                                new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                            else:
                                pass
                    acc.update(correct_cnt, batch_y.size(0))
                    loss.update(loss_batch,batch_y.size(0))
                acc_array[task] = acc.avg()
                loss_array[task] = loss.avg()
        print(acc_array)

        print(loss_array)
        print(task, self.task_seen, "acc", np.mean(acc_array[:self.task_seen]), np.mean(loss_array[:self.task_seen]))
        if self.params.error_analysis:
            self.error_list.append((no, nn, oo, on))
            self.new_class_score.append(new_class_score.avg())
            self.old_class_score.append(old_class_score.avg())
            print("no ratio: {}\non ratio: {}".format(no/(no+nn+0.1), on/(oo+on+0.1)))
            print(self.error_list)
            print(self.new_class_score)
            print(self.old_class_score)
            self.fc_norm_new.append(self.model.linear.weight[self.new_labels_zombie].mean().item())
            self.fc_norm_old.append(self.model.linear.weight[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            self.bias_norm_new.append(self.model.linear.bias[self.new_labels_zombie].mean().item())
            self.bias_norm_old.append(self.model.linear.bias[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            print(self.fc_norm_old)
            print(self.fc_norm_new)
            print(self.bias_norm_old)
            print(self.bias_norm_new)
        return acc_array,loss_array