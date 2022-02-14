import argparse
import random
import numpy as np
import torch
from experiment.run import multiple_run,multiple_RLtrainig_run
from utils.utils import boolean_string
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def main(args):
    print(args)
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.trick = {'labels_trick': args.labels_trick, 'separated_softmax': args.separated_softmax,
                  'kd_trick': args.kd_trick, 'kd_trick_star': args.kd_trick_star, 'review_trick': args.review_trick,
                  'nmc_trick': args.nmc_trick}
    if(args.num_runs>1):
        multiple_RLtrainig_run(args)
    else:
        multiple_run(args)



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--num_runs', dest='num_runs', default=1, type=int,
                        help='Number of runs (default: %(default)s)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed')

    ########################Misc#########################
    parser.add_argument('--val_size', dest='val_size', default=0.1, type=float,
                        help='val_size (default: %(default)s)')
    parser.add_argument('--num_val', dest='num_val', default=3, type=int,
                        help='Number of batches used for validation (default: %(default)s)')
    parser.add_argument('--num_runs_val', dest='num_runs_val', default=3, type=int,
                        help='Number of runs for validation (default: %(default)s)')
    parser.add_argument('--error_analysis', dest='error_analysis', default=False, type=boolean_string,
                        help='Perform error analysis (default: %(default)s)')
    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='print information or not (default: %(default)s)')

    ########################Agent#########################
    parser.add_argument('--agent', dest='agent', default='ER',
                        choices=["ER_offline",'LAMAML','RLER','ER', "ER_RL_ratio","ER_RL_addIter","ER_dyna_iter","ER_RL_iter",'EWC', 'AGEM', 'CNDPM', 'LWF', 'ICARL', 'GDUMB',
                                 'ASER','SCR','SCR_META',
                                 "SCR_RL_ratio","SCR_RL_iter"],
                        help='Agent selection  (default: %(default)s)')
    parser.add_argument('--update', dest='update', default='random', choices=['random', 'GSS', 'ASER','rt','timestamp','rt2'],
                        help='Update method  (default: %(default)s)')
    parser.add_argument('--retrieve', dest='retrieve', default='random', choices=['MIR', 'random', 'ASER','RL','match','mem_match'],
                        help='Retrieve method  (default: %(default)s)')

    ########################Optimizer#########################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1,
                        type=float,
                        help='Learning_rate (default: %(default)s)')
    parser.add_argument('--epoch', dest='epoch', default=1,
                        type=int,
                        help='The number of epochs used for one task. (default: %(default)s)')
    parser.add_argument('--batch', dest='batch', default=10,
                        type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--test_batch', dest='test_batch', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='weight_decay')

    ########################Data#########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=10,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    parser.add_argument('--fix_order', dest='fix_order', default=False,
                        type=boolean_string,
                        help='In NC scenario, should the class order be fixed (default: %(default)s)')
    parser.add_argument('--plot_sample', dest='plot_sample', default=False,
                        type=boolean_string,
                        help='In NI scenario, should sample images be plotted (default: %(default)s)')
    parser.add_argument('--data', dest='data', default="cifar10",
                        help='Path to the dataset. (default: %(default)s)')
    parser.add_argument('--cl_type', dest='cl_type', default="nc", choices=['nc', 'ni','nc_balance',"nc_first_half","nc_second_half"],
                        help='Continual learning type: new class "nc" or new instance "ni". (default: %(default)s)')
    parser.add_argument('--ns_factor', dest='ns_factor', nargs='+',
                        default=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6), type=float,
                        help='Change factor for non-stationary data(default: %(default)s)')
    parser.add_argument('--ns_type', dest='ns_type', default='noise', type=str, choices=['noise', 'occlusion', 'blur'],
                        help='Type of non-stationary (default: %(default)s)')
    parser.add_argument('--ns_task', dest='ns_task', nargs='+', default=(1, 1, 2, 2, 2, 2), type=int,
                        help='NI Non Stationary task composition (default: %(default)s)')
    parser.add_argument('--online', dest='online', default=True,
                        type=boolean_string,
                        help='If False, offline training will be performed (default: %(default)s)')
    parser.add_argument('--offline',default = False,type=boolean_string)
    ########################ER#########################
    parser.add_argument('--mem_size', dest='mem_size', default=5000,
                        type=int,
                        help='Memory buffer size (default: %(default)s)')
    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10,
                        type=int,
                        help='Episode memory per batch (default: %(default)s)')

    ########################EWC##########################
    parser.add_argument('--lambda', dest='lambda_', default=100, type=float,
                        help='EWC regularization coefficient')
    parser.add_argument('--alpha', dest='alpha', default=0.9, type=float,
                        help='EWC++ exponential moving average decay for Fisher calculation at each step')
    parser.add_argument('--fisher_update_after', dest='fisher_update_after', type=int, default=50,
                        help="Number of training iterations after which the Fisher will be updated.")

    ########################MIR#########################
    parser.add_argument('--subsample', dest='subsample', default=50,
                        type=int,
                        help='Number of subsample to perform MIR(default: %(default)s)')

    ########################GSS#########################
    parser.add_argument('--gss_mem_strength', dest='gss_mem_strength', default=10, type=int,
                        help='Number of batches randomly sampled from memory to estimate score')
    parser.add_argument('--gss_batch_size', dest='gss_batch_size', default=10, type=int,
                        help='Random sampling batch size to estimate score')

    ########################ASER########################
    parser.add_argument('--k', dest='k', default=5,
                        type=int,
                        help='Number of nearest neighbors (K) to perform ASER (default: %(default)s)')

    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str, choices=['neg_sv', 'asv', 'asvm'],
                        help='Type of ASER: '
                             '"neg_sv" - Use negative SV only,'
                             ' "asv" - Use extremal values of Adversarial SV and Cooperative SV,'
                             ' "asvm" - Use mean values of Adversarial SV and Cooperative SV')

    parser.add_argument('--n_smp_cls', dest='n_smp_cls', default=2.0,
                        type=float,
                        help='Maximum number of samples per class for random sampling (default: %(default)s)')

    ########################CNDPM#########################
    parser.add_argument('--stm_capacity', dest='stm_capacity', default=1000, type=int, help='Short term memory size')
    parser.add_argument('--classifier_chill', dest='classifier_chill', default=0.01, type=float,
                        help='NDPM classifier_chill')
    parser.add_argument('--log_alpha', dest='log_alpha', default=-300, type=float, help='Prior log alpha')

    ########################GDumb#########################
    parser.add_argument('--minlr', dest='minlr', default=0.0005, type=float, help='Minimal learning rate')
    parser.add_argument('--clip', dest='clip', default=10., type=float,
                        help='value for gradient clipping')
    parser.add_argument('--mem_epoch', dest='mem_epoch', default=70, type=int, help='Epochs to train for memory')

    #######################Tricks#########################
    parser.add_argument('--labels_trick', dest='labels_trick', default=False, type=boolean_string,
                        help='Labels trick')
    parser.add_argument('--separated_softmax', dest='separated_softmax', default=False, type=boolean_string,
                        help='separated softmax')
    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                        help='Knowledge distillation with cross entropy trick')
    parser.add_argument('--kd_trick_star', dest='kd_trick_star', default=False, type=boolean_string,
                        help='Improved knowledge distillation trick')
    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                        help='Review trick')
    parser.add_argument('--nmc_trick', dest='nmc_trick', default=False, type=boolean_string,
                        help='Use nearest mean classifier')
    parser.add_argument('--mem_iters', dest='mem_iters', default=1, type=int,
                        help='mem_iters')

    parser.add_argument('--start_mem_iters',  default=-1, type=int,
                        help='mem_iter for the first task')

    ####################Early Stopping######################
    parser.add_argument('--min_delta', dest='min_delta', default=0., type=float,
                        help='A minimum increase in the score to qualify as an improvement')
    parser.add_argument('--patience', dest='patience', default=0, type=int,
                        help='Number of events to wait if no improvement and then stop the training.')
    parser.add_argument('--cumulative_delta', dest='cumulative_delta', default=False, type=boolean_string,
                        help='If True, `min_delta` defines an increase since the last `patience` reset, '
                             'otherwise, it defines an increase after the last event.')

    ####################SupContrast######################
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--examine_train', type=boolean_string, default=False,
                        )
    parser.add_argument('--no_aug', type=boolean_string, default=False,
                        )
    parser.add_argument('--single_aug', type=boolean_string, default=False,
                        )
    parser.add_argument('--aug_type',default="")
    parser.add_argument('--softmaxhead_lr',type=float,default=0.1)

    parser.add_argument('--buffer_tracker', type=boolean_string, default=False,
                        help='Keep track of buffer with a dictionary')
    parser.add_argument('--warmup', type=int, default=4,
                        help='warmup of buffer before retrieve')
    parser.add_argument('--head', type=str, default='mlp',
                        help='projection head')

    parser.add_argument('--use_softmaxloss',type=boolean_string,default=False)
    parser.add_argument('--softmax_nlayers',type=int,default=1,help="softmax head for scr")
    parser.add_argument('--softmax_nsize',type=int,default=1024,help="softmax head size for scr")
    parser.add_argument('--softmax_membatch', type=int, default=100, help="softmax mem batchsize for scr")
    parser.add_argument('--softmax_dropout', type=boolean_string, default=False, help="whether to use dropout in softmax head")
    parser.add_argument('--softmax_type',type=str,default = 'None',choices=['None','seperate','meta'])

# ######### meta learn lamaml related
#     parser.add_argument('--learn_lr',type=boolean_string,default = False,help="whether to learn softmax head lr")
#     parser.add_argument('--second_order',type=boolean_string,default= False)
#     parser.add_argument('--grad_clip_norm', type=float, default=2.0,
#                         help='Clip the gradients by this value')
#     parser.add_argument('--sync_update', default=False, action='store_true',
#                         help='the LRs and weights should be updated synchronously')
#     parser.add_argument('--xav_init', default=False, action='store_true',
#                         help='Use xavier initialization')
#     parser.add_argument('--opt_lr', type=float, default=1e-1,
#                         help='learning rate for LRs')
#     parser.add_argument('--opt_wt', type=float, default=1e-1,
#                         help='learning rate for weights')
#     parser.add_argument('--alpha_init', type=float, default=1e-3,
#                         help='initialization for the LRs')


    # ################### LAMAML
    # parser.add_argument('--alpha_init', type=float, default=1e-3,
    #                     help='initialization for the LRs')

    # #################### replay dynamics ####################
    # parser.add_argument("--joint_replay_type",default="together",choices=["together","seperate"],
    #                     help="implementation type of joint training of incoming batch and memory batch")
    # parser.add_argument("--online_hyper_tune", default=False, type=boolean_string)
    # parser.add_argument("--online_hyper_valid_type", default="test_data", type=str, choices=["real_data","test_mem"])
    # parser.add_argument("--online_hyper_freq", default=1, type=int)
    # parser.add_argument("--online_hyper_lr_list_type",default="basic",choices=["scr","basic","4lr","5lr"])
    # parser.add_argument("--online_hyper_RL",default=False,type=boolean_string)
    # parser.add_argument("--scr_memIter", default=False, type=boolean_string)
    # parser.add_argument("--scr_memIter_type",default="c_MAB",choices=["c_MAB","MAB"])
    # parser.add_argument("--scr_memIter_state_type", default="4dim", choices=["7dim","6dim","3dim","4dim","train"])
    # parser.add_argument("--scr_memIter_action_type", default="4", choices=["4","8"])
    #
    # parser.add_argument("--temperature_scaling",default=False,type=boolean_string)
    # parser.add_argument("--frozen_old_fc", dest="frozen_old_fc", default=False, type=boolean_string)
    # parser.add_argument("--do_cutmix", dest="do_cutmix", default=False, type=boolean_string)
    # parser.add_argument("--cutmix_prob", default=0.5, type=float)
    # parser.add_argument("--cutmix_batch", default=10, type=int)
    # parser.add_argument("--cutmix_type", default="random", choices=["most_confused","train_mem","random","cross_task","mixed"])
    # parser.add_argument("--close_loop_mem_type", default="random",
    #                     choices=["low_acc",  "random", ])
    #
    # parser.add_argument("--only_task_seen",dest="only_task_seen",default=False,type=boolean_string)
    parser.add_argument("--dyna_mem_iter",dest='dyna_mem_iter',default="None",type=str,choices=["random","STOP","None"],
                        help='If True, adjust mem iter')
    parser.add_argument("--train_acc_max",default=0.95)
    parser.add_argument("--train_acc_min",default=0.85)

    # parser.add_argument("--replay_old_only",dest="replay_old_only",default=False,type=boolean_string,)
    #
    # parser.add_argument("--split_new_old",dest="split_new_old",default=False,type=boolean_string)
    parser.add_argument('--mem_iter_max', dest='mem_iter_max', default=20, type=int,
                        help='')

    parser.add_argument('--mem_iter_min', dest='mem_iter_min', default=1, type=int,
                        help='')
    # parser.add_argument('--mem_iter_std', dest='mem_iter_std', default=0.3, type=int,
    #                     help='')
    #
    #
    # parser.add_argument('--mem_ratio_max', default=1.5,
    #                     help='')
    #
    # parser.add_argument('--mem_ratio_min', default=0.1,
    #                     help='')
    #
    # parser.add_argument('--incoming_ratio', dest='incoming_ratio', default=1.0, type=float,
    #                     help='incoming  gradient update ratio')
    # parser.add_argument('--mem_ratio', dest='mem_ratio', default=1.0, type=float,
    #                     help='mem gradient update ratio')
    #
    # parser.add_argument('--task_start_mem_ratio', dest='task_start_mem_ratio', default=0.5, type=float,
    #                     help='mem gradient update ratio')
    # parser.add_argument('--task_start_incoming_ratio', dest='task_start_incoming_ratio', default=0.1, type=float,
    #                     help='mem gradient update ratio')
    # parser.add_argument("--dyna_ratio", dest='dyna_ratio', type=str, default="None", choices=['dyna','random','None'],
    #                     help='adjust dyna_ratio')
    # parser.add_argument("--rl_exp_type",dest="rl_exp_type",type=str,default="exp",choices=["stb2","l_exp","stb","exp","m_exp3","m_exp","m_exp2"])
    #
    # ################### RL test buffer #####################
    # parser.add_argument('--test_mem_size', dest='test_mem_size', default=300,
    #                     type=int,
    #                     help='Test Memory buffer size (default: %(default)s)')
    # parser.add_argument('--test_mem_batchSize', dest='test_mem_batchSize', default=100,
    #                     type=int,
    #                     help='Test Memory buffer batch size (default: %(default)s)')
    # parser.add_argument("--use_test_buffer",dest='use_test_buffer',default=False,type=boolean_string,
    #                     help='If True, evaluate model on the test buffer during CL training')
    # parser.add_argument("--test_buffer_type",default="class_balance",choices=["class_balance","reservior_sampling"])
    # parser.add_argument("--test_retrieve_num",default=300,type=int)
    parser.add_argument("--dyna_type",default="train_acc",choices=["random","train_acc"])
    # parser.add_argument('--use_tmp_buffer',dest='use_tmp_buffer',default=False,type=boolean_string,
    #                     help='If True, use a tmp buffer to store the to-be-insert samples from new task/replace indices '
    #                          'and insert these into memory at the end of new task')
    #
    # parser.add_argument('--strict_balance', default="False", type=boolean_string,
    #                     help="whether computing state stats on a class balanced sample from train memory and test memory")
    # parser.add_argument("--test_mem_type", dest='test_mem_type', default="after", type=str, choices=["before", "after"],
    #                     help='')
    # parser.add_argument("--test_mem_recycle",default = False, type=boolean_string)
    #
    # #################################### RL basics ####################################
    # # parser.add_argument("--RL_type",dest='RL_type',default="NoRL",type=str,choices=[ "RL_actor","RL_ratio_1para","RL_adpRatio","RL_ratio",
    # #                                                             "RL_memIter","NoRL","DormantRL","RL_ratioMemIter","RL_2ratioMemIter"],#"1dim","2dim",
    # #                     help='RL_memIter dynamic adjust memIteration; 1dim and 2dim employ MAB to adjust coef of retrieve index')
    # #
    #
    # parser.add_argument("--RL_type",dest='RL_type',default="NoRL",type=str,choices=[ "RL_MDP","RL_MAB","NoRL","DormantRL",],#"1dim","2dim",
    #                     help='')
    # ## action
    # parser.add_argument('--action_size', dest='action_size', default=11,
    #                     type=int,
    #                     help='Action size (default: %(default)s)')
    # parser.add_argument('--actor_type', dest='actor_type', default="greedy",
    #                     type=str,)
    # parser.add_argument("--std_trainable",default=False)
    # parser.add_argument("--action_space_type",dest="action_space_type",default="sparse",type=str,choices=["cont","monly_dense","ionly_dense","ionly","upper","posneu","sparse","medium","dense"])
    # parser.add_argument("--hp_action_space",default="ratio_iter",choices=["ratio","ratio_iter","iter","aug_iter"])
    # parser.add_argument("--MAB_reward_len",default="100",type=int)
    # ## reward
    # parser.add_argument("--reward_type", dest='reward_type', default="test_acc", type=str,
    #                     choices=["test_loss_v_rlt","test_alpha01_loss_acc","test_loss_old","test_loss_median","test_loss_acc","acc_diff","test_loss_rlt","test_loss","scaled", "real_reward", "incoming_acc", "mem_acc", "test_acc","test_acc_rlt", "test_acc_rg","relative",
    #                              "multi-step","multi-step-0","multi-step-0-rlt","multi-step-0-rlt-loss"],
    #                     help='')
    # parser.add_argument('--reward_rg',dest='reward_rg',default=0,type=float,help="param to for rward regularization")
    #
    # parser.add_argument('--reward_within_batch',default=False,type=boolean_string)
    #
    #
    # parser.add_argument("--reward_test_type", dest='reward_test_type', default="None", type=str,
    #                     choices=["reverse", "relative", "None"],
    #                     help='')
    #
    # ## state
    # parser.add_argument("--state_feature_type", dest='state_feature_type', default="train_test4", type=str,
    #                     # choices=["new_old6_overall_train","new_old5_overall","new_old5_scale","new_old_old4","new_old_old4_noi","new_old_old","new_old5_4time","new_old5_task","new_old5_incoming","new_old6mn_org","new_old6mn_incoming","new_old3","new_old6mnt","new_old7","new_old6mn","new_old6m","new_old6","new_old11","new_old9","new_old5","new_old5t","new_old4","new_old2","3_dim", "4_dim", "3_loss", "4_loss", "6_dim",
    #                     #          "7_dim","task_dim","8_dim"],
    #                     help='state feature ')
    # ## dynamics
    # parser.add_argument("--done_freq",dest="done_freq",default=249,type=int)
    #
    # parser.add_argument("--virtual_update_times",default=0,type=int)
    # parser.add_argument("--use_ref_model",default = False)
    #
    #
    # parser.add_argument("--episode_type", dest='episode_type', default="batch", type=str, choices=["multi-step", "batch"],
    #                     help='')
    #
    # parser.add_argument("--dynamics_type", dest='dynamics_type', default="next_batch", type=str,
    #                     choices=["same_batch", "next_batch", "within_batch"],
    #                     help='whether the reward and transition dynamics are computed for same incoming batch or not')
    #
    # parser.add_argument("--RL_start_batchstep", dest="RL_start_batchstep", default=0, type=int)
    # parser.add_argument("--RL_agent_update_flag",dest="RL_agent_update_flag",default=True,type=boolean_string)
    # parser.add_argument("--start_task",default=0)
    # parser.add_argument("--ratio_sigma",default=0.01)
    # #################################### critic training####################################
    # parser.add_argument('--q_function_type', type=str, default="mlp")
    # parser.add_argument("--update_q_target_freq",default=250,type=int)
    # parser.add_argument('--double_DQN',default = True,type=boolean_string)
    #
    # parser.add_argument("--critic_type", dest='critic_type', default='critic', type=str,
    #                     choices=["task_critic","critic","actor_critic",])
    # parser.add_argument("--actor_output_activation",default="sigmoid",choices=["sigmoid","relu","identity"])
    # parser.add_argument("--critic_task_layer",default=0,type=int)
    # parser.add_argument("--critic_last_layer", default=0, type=int)
    # parser.add_argument("--critic_task_size",default=10,type=int)
    # parser.add_argument("--critic_last_size", default=10, type=int)
    #
    # parser.add_argument("--critic_ER_type",dest='critic_ER_type',default='recent2',type=str,choices=["recent4","random","recent","recent2","recent3"])
    #
    # parser.add_argument("--ER_batch_size",dest="ER_batch_size",default=50,type=int,) #50
    # parser.add_argument('--critic_nlayer', dest='critic_nlayer', default= 3,
    #                     type=int,
    #                     help='critic network size (default: %(default)s)')
    # parser.add_argument('--critic_layer_size', dest='critic_layer_size', default= 32,
    #                     type=int,
    #                     help='critic network size (default: %(default)s)')
    # parser.add_argument('--critic_training_iters', dest='critic_training_iters', default= 1,
    #                     type=int,
    #                     help="")
    #
    # parser.add_argument('--critic_lr', dest='critic_lr', default=5 * 10 ** (-4),
    #                     type=float,
    #                     help="")
    # parser.add_argument("--critic_lr_type",dest="critic_lr_type",default="static",type=str,choices=["static","basic","large","mid","small"])
    #
    # # parser.add_argument('--critic_wd', dest='critic_wd', default=0,
    # #                     type=int,
    # #                     help="")
    # parser.add_argument('--critic_wd', dest='critic_wd', default=1 * 10 ** (-4),
    #                     type=int,
    #                     help="")
    # parser.add_argument('--critic_training_start', dest='critic_training_start', default= 80,
    #                     type=int,
    #                     help="")
    #
    # parser.add_argument('--critic_recent_steps', dest='critic_recent_steps', default= 250,
    #                     type=int,
    #                     help="")
    # parser.add_argument('--critic_use_model', dest='critic_use_model', default=False,type=boolean_string,
    #                     help="")
    # #################################### multiple buffer idea ####################################
    #
    # parser.add_argument('--test_retrieval_step', dest='test_retrieval_step', default= -1,
    #                     type=int,
    #                     help="")
    #
    parser.add_argument('--dataset_random_type', dest='dataset_random_type', default= "order_random",
                        type=str,choices=["order_random","task_random"],
                        help="")
    #
    # parser.add_argument('--switch_buffer_type', dest='switch_buffer_type', default= "one_buffer",
    #                     type=str,choices=["one_buffer","two_buffer","dyna_buffer"],
    #                     help="whether and how to switch replay buffer")
    #
    # parser.add_argument('--switch_buffer_freq', dest='switch_buffer_freq', default= 1000,
    #                     type=int,
    #                     help="")

    parser.add_argument("--resnet_size",default="reduced",choices=["normal","reduced"])
    parser.add_argument("--randaug",default=False)
    parser.add_argument("--randaug_type",default="static",choices=["dynamic","static"])
    parser.add_argument("--aug_target",default="both",choices=["mem","incoming","both","none"])
    parser.add_argument("--scraug",default=False)
    parser.add_argument("--randaug_N", default=0,type=int)
    parser.add_argument("--randaug_M", default=1,type=int)
    parser.add_argument("--aug_start",default=0,type=int)
    #################################################

    parser.add_argument('--save_prefix', dest='save_prefix', default="",  help='')

    parser.add_argument('--new_folder', dest='new_folder', default="", help='')

    parser.add_argument('--test', dest='test', default=" ", type=str,choices=["not_reset"],
                        help='')
    parser.add_argument('--debug_mode',default=False, type=boolean_string)
    parser.add_argument('--acc_no_aug',default=True)

    parser.add_argument('--GPU_ID', dest='GPU_ID', default= 0,
                        type=int,
                        help="")


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.GPU_ID)#args.GPU_ID

    if(args.data=="cifar100"):
        args.num_tasks = 20
    elif(args.data=="cifar10"):
        args.num_tasks = 5
    elif(args.data=="mini_imagenet"):
        args.num_tasks=10
    elif(args.data=="clrs25"):
        args.num_tasks=5
    elif(args.data=="core50"):
        args.num_tasks=9
    else:
        raise NotImplementedError("not seen dataset",args.data)

    main(args)
