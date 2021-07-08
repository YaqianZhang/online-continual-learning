class RL_base_agent(object):
    def __init__(self, params):
        self.params = params

        self.epsilon = 0.1
        self.alpha = 0.9 # Q update parameters
        self.selected_num = 1


    def sample_action(self, state):
        pass


    def initialize_q(self):
        pass




    def update_agent(self,reward,state,action,next_state,done,):
        pass



    def save_q(self,prefix):
        pass




