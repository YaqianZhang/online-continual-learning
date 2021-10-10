import numpy as np
import torch

class RL_ExperienceReplay(object):

    def __init__(self,params):
        super().__init__()
        self.params = params
        self.size = 10000
        self.state_list=[]
        self.action_list=[]
        self.reward_list=[]
        self.next_state_list=[] ## to-do: more efficient storage with one state_list
        self.done_list=[]

        self.recent_steps=params.critic_recent_steps
    def reset_RL_buffer(self):
        self.state_list=[]
        self.action_list=[]
        self.reward_list=[]
        self.next_state_list=[] ## to-do: more efficient storage with one state_list
        self.done_list=[]


    def can_sample(self):
        if (self.params.critic_ER_type == "recent3" or self.params.critic_ER_type == "recent4"):
            return len(self.action_list)>3
        return len(self.action_list)>self.params.ER_batch_size
    def sample(self,num_retrieve):
        valid_indices = len(self.action_list)
        if(self.params.state_feature_type == "new_old5_4time"):
            valid_indices -= 4

        if(self.params.critic_ER_type == "recent2" or self.params.critic_ER_type == "recent3" or self.params.critic_ER_type == "recent4"):
            if( (valid_indices-self.recent_steps)<0):
                start = 0
            else:
                start = valid_indices-self.recent_steps
            valid_range = np.arange(start ,valid_indices)
            if(valid_range.shape[0] == 0):
                raise NotImplementedError("not enough samples in RL buffer")
            if (self.params.critic_ER_type == "recent4"):
                if(num_retrieve > valid_range.shape[0]):
                    num_retrieve = valid_range.shape[0]
                   # print("retrieve",num_retrieve)

            idx = np.random.choice(valid_range, num_retrieve, )
            #print("idx!!!",idx.shape,len(self.action_list),idx)
        elif(self.params.critic_ER_type == "recent"):
            valid_indices = len(self.action_list)
            idx = np.random.choice(valid_indices, num_retrieve, )

        elif(self.params.critic_ER_type == "random"):

            idx = np.random.choice(valid_indices, num_retrieve, )
        else:
            raise NotImplementedError("Not defined critic ER type")

        if(self.params.state_feature_type =="new_old5_4time"):
            return self.from_idx_to_data_time(idx)
        else:
            return self.from_idx_to_data(idx)

    def append_state(self,state):
        # print(state)
        # print(self.state_list[-3:])
        new_state = self.state_list[-3:]+[state]
       # print(new_state)
        #state = np.array(new_state, dtype=np.float32).reshape([1, -1])
        state = torch.cat(new_state).reshape([1,-1])
        #state = torch.from_numpy(state)

        return  state


    def from_idx_to_data_time(self, idx): ## todo id need to be less than n-4


        time_idx = [np.arange(id,id+4) for id in idx]

        state_batch = np.concatenate(self.state_list)[time_idx]
       # print(state_batch)

        state_batch = state_batch.reshape((-1,20))
        # print(state_batch)
        # assert False


        action_batch = np.array(self.action_list)[idx]
        reward_batch = np.array(self.reward_list)[idx]
        next_state_batch = np.concatenate(self.next_state_list)[idx]
        done_batch = np.array(self.done_list)[idx]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch





    def from_idx_to_data(self,idx):
        state_batch = np.concatenate(self.state_list)[idx]
        action_batch = np.array(self.action_list)[idx]
        reward_batch = np.array(self.reward_list)[idx]
        #if(self.params.episode_type == "multi-step"):
        next_state_batch =np.concatenate(self.next_state_list)[idx]
        done_batch = np.array(self.done_list)[idx]
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch
        # else:
        #     return state_batch, action_batch, reward_batch, None, None



    def store(self,state,action,reward,next_state,done):
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.state_list.append(state)
        self.next_state_list.append(next_state)
        self.done_list.append(done)
