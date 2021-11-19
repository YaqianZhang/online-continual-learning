
import os
import time
def get_prefix_time(params,):

    folder_path = "results/" + str(params.seed)
    if (not os.path.exists(folder_path)):
        os.mkdir(folder_path)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    trick = str(params.num_val)+"_"+timestamp
    prefix = folder_path + '/' + params.agent +str(params.epoch)+ "_" + params.retrieve[:3] + "_" + params.update[:3] + '_' + trick  + str(
        params.num_tasks) + "_" + str(params.mem_size)+ "_"+params.data+"_"


    print("save file name :" + prefix)

    return prefix

def get_prefix(params,run):

    trick = ""
    if(params.joint_replay_type != "together"):
        trick += "replaySep_"

    if(params.only_task_seen):
        trick+="onlySeen_"
    if(params.frozen_old_fc):
        trick+="frz_"
    if(params.randaug):
        trick+="raug"+str(params.randaug_N)+str(params.randaug_M)+"_"

    if(params.online_hyper_tune):
        trick += "hp"+str(params.online_hyper_freq)+params.online_hyper_lr_list_type+"_"
        if(params.online_hyper_valid_type == "real_data"):
            trick += "real_"
    else:
        if (params.learning_rate != 0.1):
            trick += "lr" + str(params.learning_rate) + "_"
    if(params.agent == "SCR" and params.scr_memIter):
        trick += "memIter_"
        if(params.scr_memIter_type =="MAB"):
            trick += "MAB_"
        trick += params.scr_memIter_state_type +"_"
        trick += "act"+params.scr_memIter_action_type + "_"
    if(params.lambda_ != 100):
        trick += str(params.lambda_)

    if (params.nmc_trick):
        trick += "NMC_"
    if (params.use_test_buffer):
        trick += "tbuf_"
        if(params.test_mem_type == "before"):
            trick += "bf" +"_"
        if(params.close_loop_mem_type == "low_acc"):
            trick+= "memlowacc_"
    if (params.use_tmp_buffer):
        trick += "tmpMem_"

    ### scr relateed: temp, softmax ####
    if(params.agent[:3]=="SCR"):
        trick+= "temp"+str(params.temp)+"_"
    if (params.softmax_type != 'None'):
        trick += "softmax"+str(params.softmax_nsize) \
    +str(params.softmax_nlayers)
        if(params.softmax_membatch != 100):
            trick += str(params.softmax_membatch)
        if(params.softmax_dropout):
            trick += "dp"
        trick+="_"
        if(params.softmaxhead_lr != 0.1):
            trick +="smlr"+str(params.softmaxhead_lr) +"_"

    ### data augumentation ###
    if (params.do_cutmix):
        trick += "cmix"+str(params.cutmix_prob)+"_"+str(params.cutmix_batch)+"_"
        if(params.cutmix_type != "random"):
            trick +=params.cutmix_type +"_"


    if (params.no_aug):
        trick += "noaug_"
    if(params.aug_type != ""):
        trick += params.aug_type
    if (params.single_aug):
        trick += "saug_"


    ### replay_dynamics
    if(params.dyna_mem_iter != "None"):
        if(params.dyna_mem_iter == "dyna"):
            trick += "dMIter_"+str(params.mem_iter_max)+str(params.mem_iter_min)+"_"
        else:
            trick += "dMIter_"+params.dyna_mem_iter
    if (params.mem_iters > 1):
        trick += "mIter" + str(params.mem_iters)+"_"
        if (params.start_mem_iters > -1):
            trick += "s"+str(params.start_mem_iters)+"_"
    if (params.incoming_ratio != 1):
        trick += "iratio" + str(params.incoming_ratio)+"_"
    if (params.mem_ratio != 1):
        trick += "mratio" + str(params.mem_ratio)+"_"
    if(params.dyna_ratio != "None"):
        trick +="dyRatio"+params.dyna_ratio+"_"


    ## switch buffer
    if(params.switch_buffer_type != "one_buffer"):
        if(params.switch_buffer_type == "two_buffer"):
            trick += "2Buff"+"_"
        elif(params.switch_buffer_type == "dyna_buffer"):
            trick += "dBuff"+str(params.switch_buffer_freq)+"_"

        else:
            raise NotImplementedError("undefined switch buffer")

    if (params.test_mem_size != 300):
        trick += "tmem" + str(params.test_mem_size) + "_"
    # if (params.test_mem_batchSize > 10):
    #     trick += "testBch" + str(params.test_mem_batchSize) + "_"



    ### Rl related
    if (params.actor_type == "random"):
        trick += "rndRL_"
    if (params.RL_type != 'NoRL' and params.actor_type != "random"):

        if(params.RL_type == "RL_memIter"):
            trick += "RLmemIter_"+str(params.mem_iter_max)+str(params.mem_iter_min)+"_"
        elif(params.RL_type == "RL_2ratioMemIter" ):
            trick += "RL2rmemIter_"+str(params.mem_iter_max)+str(params.mem_iter_min)+"_"
            trick += str(params.mem_ratio_max)+"_"+str(params.mem_ratio_min)+"_"
        elif (params.RL_type == "RL_ratio_1para"):
            trick += "RLratio1pr_" + str(params.mem_iter_max) + str(params.mem_iter_min) + "_"
        else:
            trick += params.RL_type + "_"

        if(params.virtual_update_times != 0):
            trick += "virtual"+str(params.virtual_update_times)+"_"
        if(params.use_ref_model):
            trick += "ref_"


        #reward
        trick += params.reward_type+"_" ## todo: fix RL_type logic
        trick += str(params.reward_rg)+"_"
        if(params.reward_within_batch):
            trick += "wthin_"

        ## action
        if(params.action_space_type != "sparse"):
            trick += params.action_space_type+"_"

        ## state
        trick += params.state_feature_type+"_"

        ## dynamics
        if(params.episode_type == "multi-step"):
            trick += "Done"+str(params.done_freq)+"_"
            if(params.double_DQN == False):
                trick += "nodouble_"
        if(params.dynamics_type == "next_batch"):
            trick+="nxtBch"+'_'
        if (params.dynamics_type == "within_batch"):
            trick += "wthBch" + '_'


        ## others

        if (params.replay_old_only):
            trick += "oldonly" + "_"
        if (params.split_new_old):
            trick += "splitno" + "_"
        if(params.temperature_scaling):
            trick  += "TS_"

        if((params.RL_agent_update_flag==False)):
            trick +="NoT"+"_"

        if(params.RL_start_batchstep != 0 ):
            trick +="bstart"+str(params.RL_start_batchstep)+"_"
        trick += str(params.task_start_mem_ratio)+str(params.task_start_incoming_ratio)+"_"



        ##critic_training
        trick += params.rl_exp_type+"_"
        trick += "critic"+str(params.critic_layer_size)+"_"+str(params.critic_nlayer)+"_"
        # trick += "ERbch"+str(params.ER_batch_size)+"_"
        if(params.q_function_type != "lstm"):
            trick += "q"+params.q_function_type[:3]+"_"
        if(params.critic_type == "task_critic"):
            trick +="qtype"+params.critic_type[:2]+"_"

            trick+="t"+str(params.critic_task_layer)+"*"+str(params.critic_task_size)
            trick += "l" + str(params.critic_last_layer) + "*" + str(params.critic_last_size)+"_"
        if (params.critic_type == "actor_critic"):
            trick += "qtype" + "actor_"
            if(params.ratio_sigma != 0.01):
                trick += "var"+str(params.ratio_sigma)+"_"
        if(params.ER_batch_size != 20):
            trick +="erb"+str(params.ER_batch_size)+"_"
        if(params.update_q_target_freq != 1000):
            trick+="targetq"+str(params.update_q_target_freq)
        if(params.critic_use_model):
            trick += "Qmodel"+"_"

        trick += params.critic_ER_type+"_"
        if(params.critic_training_start != 80):
            trick += "qstart"+str(params.critic_training_start)+"_"

        if(params.critic_lr_type != "basic"):
            trick +="rllr"+params.critic_lr_type+"_"
        if(params.critic_wd >0):
            trick +="wd-6"+"_"
        if(params.critic_lr != 1e-3):
            trick += str(params.critic_lr)
        # trick += "crtBchSize"+str(params.ER_batch_size)+"_"
        if(params.critic_training_iters != 1):
            trick += "crtitr" + str(params.critic_training_iters) + "_"
        if(params.critic_recent_steps != 100):
            trick += "criticRct"+str(params.critic_recent_steps)+"_"
        if(params.reward_test_type != "None"):
            trick += params.reward_test_type + "_"

    if(params.test == "not_reset"):
        trick += "no_reset"


    if (not params.save_prefix == ""):
        trick += params.save_prefix+"_"
    if (not params.save_prefix_tmp == ""):
        trick += params.save_prefix_tmp+"_"

    if (not params.save_prefix_tmp2 == ""):
        trick += params.save_prefix_tmp2+"_"
    if( not params.eps_mem_batch == 10):
        trick += "memBch"+str(params.eps_mem_batch)+"_"
    if(params.num_runs>1):
        trick += "numRuns"+str(params.num_runs) + "_"


    if(params.dataset_random_type == "order_random"):
        trick += "orderRnd"+"_"
    trick += params.cl_type+"_"


    if(params.new_folder != ""):
        folder_path = "results/" + params.new_folder+"/"+str(params.seed)
        if (not os.path.exists("results/" + params.new_folder)):
            os.mkdir("results/" + params.new_folder)
    else:
        folder_path = "results/" + str(params.seed)
    if (not os.path.exists(folder_path)):
        os.mkdir(folder_path)
    prefix = folder_path + '/' + params.agent +str(params.epoch)+ "_" + params.retrieve[:3] + "_" + params.update[:3] + '_' + trick  + str(
        params.num_tasks) + "_" + str(params.mem_size)+ "_"+params.data+"_"
    print("save file name :"+ prefix)

    return prefix