from agents.gdumb import Gdumb
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from agents.exp_replay import ExperienceReplay
from agents.rl_exp_replay import RL_ExperienceReplay
from agents.agem import AGEM
from agents.ewc_pp import EWC_pp
from agents.cndpm import Cndpm
from agents.lwf import Lwf
from agents.icarl import Icarl
from agents.lamaml import LAMAML
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
#from utils.buffer.replay_times_update import Replay_times_update
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.rl_retrieve import RL_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update



data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'mini_imagenet': Mini_ImageNet,
    'openloris': OpenLORIS
}

agents = {
    'ER': ExperienceReplay,
    'RLER':RL_ExperienceReplay,
    'LAMAML':LAMAML,
    'EWC': EWC_pp,
    'AGEM': AGEM,
    'CNDPM': Cndpm,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'RL':RL_retrieve
}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'ASER': ASER_update,
    'rt':Reservoir_update,
'rt2':Reservoir_update,
    'timestamp':Reservoir_update,
}

