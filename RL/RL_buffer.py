import numpy as np

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
        self.done_list=[]
        self.recent_steps=params.critic_recent_steps

    def can_sample(self):
        return len(self.action_list)>self.params.ER_batch_size
    def sample(self,num_retrieve):
        valid_indices = len(self.action_list)

        if(self.params.critic_ER_type == "recent2"):
            if( (valid_indices-self.recent_steps)<0):
                start = 0
            else:
                start = valid_indices-self.recent_steps
            valid_range = np.arange(start ,valid_indices)
            idx = np.random.choice(valid_range, num_retrieve, )
        elif(self.params.critic_ER_type == "recent"):
            valid_indices = len(self.action_list)
            idx = np.random.choice(valid_indices, num_retrieve, )

        elif(self.params.critic_ER_type == "random"):

            idx = np.random.choice(valid_indices, num_retrieve, )
        else:
            raise NotImplementedError("Not defined critic ER type")




        state_batch = np.concatenate(self.state_list)[idx]
        action_batch = np.array(self.action_list)[idx]
        reward_batch = np.array(self.reward_list)[idx]
        next_state_batch =np.concatenate(self.next_state_list)[idx]
        done_batch = np.array(self.done_list)[idx]


        return state_batch,action_batch,reward_batch,next_state_batch,done_batch



    def store(self,state,action,reward,next_state,done):
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.state_list.append(state)
        self.next_state_list.append(next_state)
        self.done_list.append(done)
