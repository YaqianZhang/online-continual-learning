#
#
class RL_trainer(object):

    def __init__(self,params,RL_env,RL_agent,):
        self.params = params
        self.state = None
        self.action = None
        self.reward = None
        self.total_reward = 0
        self.return_list = []
        self.RL_env = RL_env
        self.RL_agent = RL_agent
        #self.memoryManager = memoryManager






    def update_return(self,done,reward):
        if(done == 1):
            #print("___________________________________")
            self.return_list.append(self.total_reward)
            self.total_reward = 0
        else:
            self.total_reward += reward






    def RL_training_step(self,stats_dict, task_seen=None,):

        ## step_env based on stats_dict,i
        ## previous history is in self.state, self.action, self.reward

        ## update RL agent
        ## step

        [state, reward, action,]=[self.state,  self.reward,self.action,]


        next_state = self.RL_env.get_state(stats_dict, task_seen=task_seen)
        done = self.RL_env.check_episode_done(stats_dict, )
        if (state != None ):
            self.update_return( done, reward)
            self.RL_agent.real_reward_list.append(reward)
            self.RL_agent.RL_running_steps += 1
            self.RL_agent.ExperienceReplayObj.store(state, action, reward, next_state, done)
            self.RL_agent.update_agent(reward, state,
                                       action, next_state, done)  ## update RL agent


        state = next_state

        self.RL_env.CL_agent.set_mem_data()

        #### virtual explore
        for i in range(self.params.virtual_update_times):
            action = self.RL_agent.take_random_action()  ## dormant RL

            if (action != None):
                end_stats = self.RL_env.step(action,virtual=True)  ## perform replay
            else:
                end_stats = stats_dict

            reward = self.RL_env.get_reward(end_stats, stats_dict, )
            self.RL_agent.ExperienceReplayObj.store(state, action, reward, state, done)


        if(self.params.use_ref_model):

            base_action = self.RL_agent.take_base_action()  ## dormant RL


            virtual_end_stats = self.RL_env.step(base_action, virtual=True,use_set_mem=True)  ## perform replay
        else:
            virtual_end_stats = stats_dict

        # reward = self.RL_env.get_reward(end_stats, stats_dict, )
        # self.RL_agent.ExperienceReplayObj.store(state, action, reward, state, done)



        action = self.RL_agent.sample_action(state) ## dormant RL
        if (action != None ):
            end_stats = self.RL_env.step(action,    use_set_mem=True)## perform replay
        else:
            end_stats = stats_dict
        reward = self.RL_env.get_reward(end_stats, stats_dict,virtual_end_stats)


        if(self.RL_agent.greedy == "greedy"):
            self.RL_agent.real_q.append(reward)
            self.RL_agent.greedy_action.append(action)

            self.RL_agent.select_batch_num.append(state[0][0].item())



        [self.state, self.action, self.reward] =  [state,  action,reward, ]
        return end_stats









