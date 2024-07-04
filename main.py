import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import wandb
import os
import copy
import time
import random
import numpy as np
from util.config import *
from util.utils import *
import json
from on_tal_task import on_tal

def main(args):

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if args.code_testing:
        exp_name = '%s_[%s]_[%s]_ON_TAL_<%s>' % ('test', args.task, args.dataset, time_str)
    else:
        exp_name = '[%s]_[%s]_ON_TAL_<%s>' % (args.task, args.dataset, time_str)
    save_path = './result/%s' % exp_name
    proj_name = 'ON_TAL-' + args.task
    print_log(save_path, exp_name)
    _args = copy.deepcopy(args)
    opt = vars(_args)
    opt_path = open(os.path.join(save_path, 'opts.json'), 'w')
    json.dump(opt, opt_path)
    opt_path.close()
    
    if args.wandb:
        wandb.init(project=proj_name, name=exp_name)
        config = wandb.config
        config.update(args)

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    args.save_path = save_path
    # task select 
    if args.task == "ontal":
        on_tal(args)
        
if __name__ == "__main__": 
    args = make_parser()    
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    args.num_workers = torch.cuda.device_count() * 4
    
    main(args)