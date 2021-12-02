from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from utils.setup_elements import input_size_match
from torchvision.transforms import transforms
from utils.augmentations import RandAugment
import torch
from utils.utils import maybe_cuda

class aug_agent(object):
    def __init__(self,params):
        self.params = params
        _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # ])

        self.transform_train.transforms.insert(0, RandAugment(self.params.randaug_N, self.params.randaug_M))

        self.scr_transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )

    def set_aug_para(self, N, M,incoming_N=1,incoming_M=14):
        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # ])

        self.transform_train.transforms.insert(0, RandAugment(N, M))


        # self.transform_train_incoming = transforms.Compose([
        #     # transforms.RandomCrop(32, padding=4),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # ])
        # # transform_test = transforms.Compose([
        # #     transforms.ToTensor(),
        # #     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # # ])
        #
        # self.transform_train_incoming.transforms.insert(0, RandAugment(incoming_N, incoming_M))

    def aug_data(self,concat_batch_x,mem_num):
        n, c, w, h = concat_batch_x.shape

        #mem_images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(mem_num)]
        #incoming_images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(mem_num,n)]

        #aug_concat_batch_x = [self.transform_train(image).reshape([1, c, w, h]) for image in mem_images]
        #aug_concat_batch_x += [self.transform_train_incoming(image).reshape([1, c, w, h]) for image in incoming_images]

        images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(n)]
        aug_concat_batch_x = [self.transform_train(image).reshape([1, c, w, h]) for image in images]
        aug_concat_batch_x = maybe_cuda(torch.cat(aug_concat_batch_x, dim=0))
        return aug_concat_batch_x
    def scr_aug_data(self,combined_batch):
        return self.scr_transform(combined_batch)



