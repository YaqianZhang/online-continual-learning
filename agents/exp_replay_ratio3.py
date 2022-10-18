import torch
from torch.utils import data
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.utils import maybe_cuda, AverageMeter
# from RL.RL_replay_base import RL_replay
# from RL.close_loop_cl import close_loop_cl
from torchvision.transforms import transforms

import numpy as np
import torch
import torch.nn as nn
from utils.setup_elements import transforms_match
from utils.utils import cutmix_data
import time
from agents.exp_replay import ExperienceReplay


class ER_ratio3(ExperienceReplay):

    def __init__(self, model, opt, params):
        super(ER_ratio3, self).__init__(model, opt, params)

        self.replay_para={"mem_ratio":self.params.mem_ratio,
                          "incoming_ratio":self.params.incoming_ratio,
                          "mem_iter":self.params.mem_iters,
                          "randaug_M":self.params.randaug_M}



        adjust_step_size = (self.params.mem_ratio +1)/2 #["mem_ratio"]
        self.replay_para["mem_ratio"] = self.params.mem_ratio / adjust_step_size
        self.replay_para["incoming_ratio"] =1/ adjust_step_size





