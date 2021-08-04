
from RL.actor.continuous_actor import MLPPolicy
from RL.agent.RL_agent_MDP import RL_memIter_agent
import torch
import numpy as np
from utils.utils import maybe_cuda
class RL_pg_agent(RL_memIter_agent):
    def __init__(self,params):
        super().__init__(params)

        self.actor = MLPPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

    def sample_action(self,state):
        action = self.actor.get_action(state)
        return action

    def update_agent(self, reward, state, action, next_state, done, ):
        if (self.params.RL_type == "DormantRL"):
            return
        self.real_reward_list.append(reward)
        self.training_steps += 1
        ## DQN

        # ## add <state, action, reward, next_state> into memory
        self.ExperienceReplayObj.store(state, action, reward, next_state, done)

        ## sample batch from replay memory to train network

        if (self.ExperienceReplayObj.can_sample() == False):
            return
        if (self.training_steps < self.critic_training_start): return
        for i in range(self.RL_training_iters):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.ExperienceReplayObj.sample(
                self.ER_batchsize)
            state_batch = maybe_cuda(torch.from_numpy(state_batch))
            action_batch = maybe_cuda(torch.from_numpy(action_batch))
            reward_batch = maybe_cuda(torch.from_numpy(reward_batch))
            next_state_batch = maybe_cuda(torch.from_numpy(next_state_batch))
            done_batch = maybe_cuda(torch.from_numpy(done_batch))
            rl_loss = self.train_batch(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        self.mse_training_loss.append(rl_loss.item())

        if (self.params.reward_type[:10] == "multi-step" and self.training_steps % self.update_q_target_freq == 0):
            self.update_q_target()

    def save_RL_stats(self, prefix):
        arr = np.array(self.mse_training_loss)
        np.save(prefix + "mse_loss.npy", arr)

        arr = np.array(self.real_reward_list)
        np.save(prefix + "real_reward_list.npy", arr)

        arr = np.array(self.real_action_list)
        np.save(prefix + "real_action_list.npy", arr)

        # save model
        torch.save(self.q_function.state_dict(), prefix + "RLmodel")

        # todo save q functions






