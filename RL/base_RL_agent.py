

import torch
import numpy as np
from RL.RL_agent import RL_agent
from utils.utils import maybe_cuda
from RL.pytorch_util import  build_mlp
from RL.RL_buffer import ExperienceReplay

class RL_base_agent(object):
    def __init__(self, params):
        pass


    def sample_action(self, state):
        pass


    def initialize_q(self):
        pass




    def update_agent(self,reward,state,action,next_state,done,):
        pass



    def save_q(self,prefix):
        pass




