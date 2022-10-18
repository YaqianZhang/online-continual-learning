from torch.utils import data
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from agents.exp_replay_cl import  ExperienceReplay_cl


from RL.RL_replay_base_stop_new import RL_replay

from RL.close_loop_cl import close_loop_cl


class ER_RL_addIter_stop_new(ExperienceReplay_cl):
    def __init__(self, model, opt, params):
        super(ER_RL_addIter_stop_new, self).__init__(model, opt, params)
        #if(self.params.online_hyper_RL or self.params.scr_memIter ):

        self.close_loop_cl = close_loop_cl(self,model, self.memory_manager)

        self.RL_replay = RL_replay(params,self.close_loop_cl)




    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()
        replay_para = None
        ### contextual bandits, memiter as action
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):

                batch_x,batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                #batch_x,batch_y = self.memory_manager.update_before_training(batch_x,batch_y)
                self.set_memIter()
                if (replay_para == None):
                    replay_para = self.replay_para
                current_state = None
                next_state = None
                STOP_FLAG = None
                action = None
                reward = None
                memiter = replay_para['mem_iter']

                for j in range(replay_para['mem_iter']):

                    concat_batch_x, concat_batch_y, mem_num = self.concat_memory_batch(batch_x, batch_y)

                    train_stats, _ = self._batch_update(concat_batch_x, concat_batch_y, losses_batch, acc_batch, i,replay_para,mem_num=mem_num)

                    if(train_stats != None):
                        if (current_state != None and action != None ) :## when mem is empty
                            next_state = self.RL_replay.get_state(train_stats,j)
                            done = 0

                            self.RL_replay.store(current_state, action, reward, next_state,done)
                        current_state = self.RL_replay.get_state(train_stats,j)

                        action = self.RL_replay.make_stop_decision(current_state)
                        STOP_FLAG = action == 0


                        if(action == 0 or j == (replay_para['mem_iter']-1)):
                            ## episode ends

                            reward = self.RL_replay.set_end_reward(train_stats,j)
                            next_state = None
                            done = 1
                            self.RL_replay.store(current_state,action,reward,next_state,done)

                        else:
                            reward = self.RL_replay.set_immediate_reward(train_stats,STOP_FLAG)



                        self.RL_replay.train_agent()


                        if(action == 0):
                            memiter=j+1
                            break


                self.mem_iter_list.append(memiter)
                ## compute reward





                self.memory_manager.update_memory(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
                    print("memIter",memiter,"eps",self.RL_replay.RL_agent.epsilon)

                    #print("replay_para", replay_para,"action:",self.RL_replay.RL_agent.greedy,self.RL_replay.action,)
        self.after_train()

