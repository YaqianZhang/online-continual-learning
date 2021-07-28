import argparse
import random
import numpy as np
import torch
from experiment.run import multiple_run,multiple_RLtrainig_run
from utils.utils import boolean_string


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
                        choices=['ER', 'EWC', 'AGEM', 'CNDPM', 'LWF', 'ICARL', 'GDUMB', 'ASER'],
                        help='Agent selection  (default: %(default)s)')
    parser.add_argument('--update', dest='update', default='random', choices=['random', 'GSS', 'ASER','rt','timestamp','rt2'],
                        help='Update method  (default: %(default)s)')
    parser.add_argument('--retrieve', dest='retrieve', default='random', choices=['MIR', 'random', 'ASER','RL'],
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

    ####################Early Stopping######################
    parser.add_argument('--min_delta', dest='min_delta', default=0., type=float,
                        help='A minimum increase in the score to qualify as an improvement')
    parser.add_argument('--patience', dest='patience', default=0, type=int,
                        help='Number of events to wait if no improvement and then stop the training.')
    parser.add_argument('--cumulative_delta', dest='cumulative_delta', default=False, type=boolean_string,
                        help='If True, `min_delta` defines an increase since the last `patience` reset, '
                             'otherwise, it defines an increase after the last event.')

    ################### RL test buffer #####################
    parser.add_argument('--test_mem_size', dest='test_mem_size', default=5000,
                        type=int,
                        help='Test Memory buffer size (default: %(default)s)')
    parser.add_argument('--test_mem_batchSize', dest='test_mem_batchSize', default=100,
                        type=int,
                        help='Test Memory buffer batch size (default: %(default)s)')
    parser.add_argument("--use_test_buffer",dest='use_test_buffer',default=False,type=boolean_string,
                        help='If True, evaluate model on the test buffer during CL training')

    parser.add_argument('--use_tmp_buffer',dest='use_tmp_buffer',default=False,type=boolean_string,
                        help='If True, use a tmp buffer to store the to-be-insert samples from new task/replace indices '
                             'and insert these into memory at the end of new task')

    #################### replay dynamics ####################
    parser.add_argument("--dyna_mem_iter",dest='dyna_mem_iter',default="None",type=str,choices=["random","dyna","None"],
                        help='If True, adjust mem iter')

    parser.add_argument('--mem_iter_max', dest='mem_iter_max', default=2, type=int,
                        help='')

    parser.add_argument('--mem_iter_min', dest='mem_iter_min', default=0, type=int,
                        help='')

    parser.add_argument('--incoming_ratio', dest='incoming_ratio', default=1.0, type=float,
                        help='incoming  gradient update ratio')
    parser.add_argument('--mem_ratio', dest='mem_ratio', default=1.0, type=float,
                        help='mem gradient update ratio')
    parser.add_argument("--dyna_ratio", dest='dyna_ratio', type=str, default="None", choices=['dyna','random','None'],
                        help='adjust dyna_ratio')

    #################################### RL basics ####################################
    parser.add_argument("--RL_type",dest='RL_type',default="NoRL",type=str,choices=[ "RL_ratio_1para","RL_adpRatio","RL_ratio",
                                                                "RL_memIter","NoRL","DormantRL","RL_ratioMemIter","RL_2ratioMemIter"],#"1dim","2dim",
                        help='RL_memIter dynamic adjust memIteration; 1dim and 2dim employ MAB to adjust coef of retrieve index')

    parser.add_argument('--action_size', dest='action_size', default=11,
                        type=int,
                        help='Action size (default: %(default)s)')
    parser.add_argument("--action_space",dest="action_space",default="sparse",type=str,choices=["sparse","medium","dense"])

    parser.add_argument("--reward_type", dest='reward_type', default="test_acc", type=str,
                        choices=["acc_diff","test_loss_rlt","test_loss","scaled", "real_reward", "incoming_acc", "mem_acc", "test_acc","test_acc_rlt", "test_acc_rg","relative",
                                 "multi-step","multi-step-0","multi-step-0-rlt","multi-step-0-rlt-loss"],
                        help='')
    parser.add_argument('--reward_rg',dest='reward_rg',default=0,type=float,help="param to for rward regularization")
    parser.add_argument("--done_freq",dest="done_freq",default=249,type=int)

    parser.add_argument("--reward_test_type", dest='reward_test_type', default="None", type=str,
                        choices=["reverse", "relative", "None"],
                        help='')

    parser.add_argument("--test_mem_type", dest='test_mem_type', default="after", type=str, choices=["before", "after"],
                        help='')

    parser.add_argument("--state_feature_type", dest='state_feature_type', default="6_dim", type=str,
                        choices=["new_old6","new_old11","new_old9","new_old5","new_old5t","new_old4","new_old2","3_dim", "4_dim", "3_loss", "4_loss", "6_dim",
                                 "7_dim","task_dim","8_dim"],
                        help='state feature ')

    parser.add_argument("--dynamics_type",dest='dynamics_type',default="same_batch",type=str,
                        choices=["same_batch","next_batch"],
                        help='whether the reward and transition dynamics are computed for same incoming batch or not')

    parser.add_argument("--episode_type", dest='episode_type', default="task", type=str, choices=["task", "batch"],
                        help='')


    #################################### critic training####################################
    parser.add_argument("--critic_ER_type",dest='critic_ER_type',default='random',type=str,choices=["random","recent","recent2"])

    parser.add_argument("--ER_batch_size",dest="ER_batch_size",default=50,type=int,)
    parser.add_argument('--critic_nlayer', dest='critic_nlayer', default= 2,
                        type=int,
                        help='critic network size (default: %(default)s)')
    parser.add_argument('--critic_layer_size', dest='critic_layer_size', default= 32,
                        type=int,
                        help='critic network size (default: %(default)s)')
    parser.add_argument('--critic_training_iters', dest='critic_training_iters', default= 1,
                        type=int,
                        help="")

    parser.add_argument('--critic_lr', dest='critic_lr', default=5 * 10 ** (-4),
                        type=int,
                        help="")

    parser.add_argument('--critic_wd', dest='critic_wd', default=1 * 10 ** (-4),
                        type=int,
                        help="")
    parser.add_argument('--critic_training_start', dest='critic_training_start', default= 1000,
                        type=int,
                        help="")

    parser.add_argument('--critic_recent_steps', dest='critic_recent_steps', default= 100,
                        type=int,
                        help="")
    parser.add_argument('--critic_use_model', dest='critic_use_model', default=False,type=boolean_string,

                        help="")
    #################################### multiple buffer idea ####################################

    parser.add_argument('--test_retrieval_step', dest='test_retrieval_step', default= -1,
                        type=int,
                        help="")

    parser.add_argument('--dataset_random_type', dest='dataset_random_type', default= "order_random",
                        type=str,choices=["order_random","task_random"],
                        help="")

    parser.add_argument('--switch_buffer_type', dest='switch_buffer_type', default= "one_buffer",
                        type=str,choices=["one_buffer","two_buffer","dyna_buffer"],
                        help="whether and how to switch replay buffer")

    parser.add_argument('--switch_buffer_freq', dest='switch_buffer_freq', default= 1000,
                        type=int,
                        help="")

    #################################################

    parser.add_argument('--save_prefix', dest='save_prefix', default="", type=str,choices=["37_actions","small_rl_lr","large_lr","blcTestMem"],
                        help='')

    parser.add_argument('--test', dest='test', default=" ", type=str,choices=["not_reset"],
                        help='')

    parser.add_argument('--GPU_ID', dest='GPU_ID', default= 0,
                        type=int,
                        help="")


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.GPU_ID)#args.GPU_ID

    main(args)
