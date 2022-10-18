import os
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.data_utils import create_task_composition, load_task_with_labels,create_task_composition_order,load_task_with_labels_correct
import pickle as pkl
import logging
from hashlib import md5
import numpy as np
from PIL import Image
from continuum.data_utils import shuffle_data, load_task_with_labels
import time
import pandas as pd
import random

import torch
from torchvision import datasets,transforms
import torch
from torchvision import datasets,transforms
from torch.utils.data import Subset
from torch.utils import data
import matplotlib.pyplot as plt
imagenet_ntask = {

    'nc': 10,


}
class IMAGENET1000(DatasetBase):
    def __init__(self, scenario, params):

        dataset = 'imagenet1000'
        self.task_nums = imagenet_ntask[scenario]
        if(params.offline):
            self.task_nums = 1
        self.scenario =scenario


        super(IMAGENET1000, self).__init__(dataset, scenario, self.task_nums, params.num_runs, params)
        self.train_batchsize=params.batch
        self.test_batchsize=params.batch


    def download_load(self):

        self.root_folder = self.root
        ## setup the dataloader for  imagenet
        ## setup test dataloader
        traindir = "/Scratch/repository/ml/datasets/ImageNet/" + "train"
        testdir = "/Scratch/repository/ml/datasets/ImageNet/" + "val"
        self.trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ])
        )
        self.testset = datasets.ImageFolder(
            testdir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ])
        )




    def setup(self, cur_run):
        # generate random task order
        ## setup test dataloader list for different tasks

        if (self.params.dataset_random_type == "task_random"):

            self.task_labels = create_task_composition(class_nums=1000, num_tasks=self.task_nums,
                                                       fixed_order=self.params.fix_order)
            print(self.task_labels)
        elif (self.params.dataset_random_type == "order_random"):
            self.task_labels = create_task_composition_order(class_nums=1000, num_tasks=self.task_nums, )
        else:
            raise NotImplementedError("undefined dataset_random_type", self.params.dataset_random_type)

        self.test_dataloaders = []

        for task in range(self.task_nums):
            task_label = self.task_labels[task]
            idx = [i for i in range(len(self.testset)) if self.testset.imgs[i][1] in task_label]
            # build the appropriate subset
            subset = Subset(self.trainset, idx)
            test_loader = data.DataLoader(subset, batch_size=self.test_batchsize, shuffle=True, num_workers=1,
                                           drop_last=True)
            self.test_dataloaders.append(test_loader)
        self.test_set = self.test_dataloaders ## used in base class
        return self.test_dataloaders



    def new_task(self, cur_task, **kwargs):
        ## setup the subset dataloader for each task of imagenet
        ## return the dataloader of each task
        task_label = self.task_labels[cur_task]
        idx = [i for i in range(len(self.trainset)) if self.trainset.imgs[i][1] in task_label]
        # build the appropriate subset
        subset = Subset(self.trainset, idx)
        train_loader = data.DataLoader(subset, batch_size=self.train_batchsize, shuffle=True, num_workers=0,
                                       drop_last=True)
        return train_loader,task_label,task_label



    def new_run(self, **kwargs):
        cur_run = kwargs['cur_run']
        self.setup(cur_run)


