from utils.setup_elements import input_size_match
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
import numpy as np



class Buffer(torch.nn.Module):
    def __init__(self, model, params,RL_agent=None, RL_env=None):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.task_seen_so_far = 0
        self.training_steps = 0


        # define buffer
        buffer_size = params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        self.buffer_label = torch.LongTensor(buffer_size).fill_(0)
        #self.buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        #self.buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_replay_times =maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_last_replay = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.unique_replay_list=[]
        self.replay_sample_label=[]

        # registering as buffer allows us to save the object using `torch.save`
        #self.register_buffer('buffer_img', buffer_img)
        #self.register_buffer('buffer_label', buffer_label)

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params)
        if(params.retrieve == "RL"):
            self.retrieve_method = name_match.retrieve_methods[params.retrieve](params,RL_agent, RL_env)
        else:
            self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

    def update_replay_times(self, indices):
        self.buffer_replay_times[indices]+=1
        self.buffer_last_replay +=1
        self.buffer_last_replay[indices] =0
    def reset(self):
        buffer_size = self.params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[self.params.data]
        self.buffer_img = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        self.buffer_label = torch.LongTensor(buffer_size).fill_(0)
        self.buffer_replay_times =maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.buffer_last_replay = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        self.current_index = 0
        self.n_seen_so_far = 0
        self.task_seen_so_far = 0
        self.training_steps = 0


    def update(self, x, y,tmp_buffer=None):
        self.training_steps += 1
        return self.update_method.update(buffer=self, x=x, y=y,tmp_buffer=tmp_buffer)

    def retrieve(self, **kwargs):
        if(self.retrieve_method.num_retrieve==-1):
            print("dynamic mem batch size")

            self.retrieve_method.num_retrieve = self.task_seen_so_far * 10 # to-do: change 10 to the batch size of new data
        return self.retrieve_method.retrieve(buffer=self, **kwargs)
    def save_buffer_info(self,prefix=""):
        removed_sample = np.array(self.unique_replay_list)
        arr = self.buffer_replay_times.detach().cpu().numpy()
        np.save(prefix+"_removed_sample.npy",removed_sample)
        np.save(prefix+"_remain_sample.npy",arr)
        np.save(prefix+"_sample_label.npy", np.array(self.replay_sample_label))
        np.save(prefix+"_sample_label_remain.npy", self.buffer_label.detach().cpu().numpy())

    def overwrite(self,idx_map,x,y):
        ## zyq: save replay_times
        for i in list(idx_map.keys()):
            replay_times = self.buffer_replay_times[i].detach().cpu().numpy()
            self.unique_replay_list.append(int(replay_times))
            self.buffer_replay_times[i]=0
            self.buffer_last_replay[i]=0
            sample_label = int(self.buffer_label[i].detach().cpu().numpy())
            self.replay_sample_label.append(sample_label)

        self.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        self.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())]


