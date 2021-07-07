from utils.setup_elements import input_size_match
from utils import name_match  # import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
import numpy as np



class Test_Buffer(torch.nn.Module):
    def __init__(self,  params,RL_agent, RL_env):
        super().__init__()
        self.params = params
        #self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.task_seen_so_far = 0

        # define buffer
        buffer_size = int( params.mem_size)#int( params.test_mem_size)
        print('test buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        # self.buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        # self.buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        self.buffer_label = torch.LongTensor(buffer_size).fill_(0)



        # registering as buffer allows us to save the object using `torch.save`
        #self.register_buffer('buffer_img', buffer_img)
        #self.register_buffer('buffer_label', buffer_label)

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params,)
        #self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

        if(params.retrieve == "RL"):
            self.retrieve_method = name_match.retrieve_methods[params.retrieve](params,RL_agent, RL_env)
        else:
            self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)


    def update(self, x, y,tmp_buffer=None):
        return self.update_method.update(buffer=self, x=x, y=y,tmp_buffer=tmp_buffer)

    def overwrite(self,idx_map,x,y):

        self.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        self.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())]

    def reset(self):
        buffer_size = self.params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[self.params.data]
        self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        self.buffer_label = torch.LongTensor(buffer_size).fill_(0)
        self.buffer_replay_times =maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_last_replay = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))


