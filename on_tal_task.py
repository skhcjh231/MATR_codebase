import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import wandb
import os
import json
import math
import sys
import copy
import time
import random
import numpy as np
from util.utils import *
import util.misc as utils
import util.logger as loggers
import torchvision.transforms as transforms
from models import build_model
from collections import defaultdict
from dataset import THUMOS14Dataset
from criterion import build_criterion
import cv2
from eval import evaluation_detection
from pathlib import Path
import glob
from collections import OrderedDict

def on_tal(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
        
def train(args):
    save_path = args.save_path
    train_dataset = THUMOS14Dataset(args, subset='train')
    test_dataset = THUMOS14Dataset(args, subset='test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.batch, shuffle= False,
                                            num_workers=args.num_workers, pin_memory=True,drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=args.batch, shuffle= False,
                                            num_workers=args.num_workers, pin_memory=True,drop_last=False) 
    
    model = build_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)

    criterion = build_criterion(args, device)
    
    max_mAP = 0
    
    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # optimizer and scheduler        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.min_lr, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.lr_Tcycle, T_mult=1, eta_max=args.max_lr,
                                                T_up=args.lr_Tup, gamma=args.lr_gamma)
    
    if args.load_model:
        checkpoint = torch.load(args.model_path)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict)
        criterion.load_state_dict(checkpoint['criterion_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1
    
    if args.wandb:
        wandb.watch(model)
    
    # training
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- %s at epoch #%d' % ('Train', epoch))
        train_log = train_one_epoch(args, train_dataset, train_loader, model, criterion, optimizer, epoch, device)
        
        if epoch % args.train_eval_step == 0:
            print_log(save_path, 'mAP: %.2f' % (train_log['mAP_train']))
            print_log(save_path, 'mAP@.3: %.2f' % (train_log['mAP_03_train']))
            print_log(save_path, 'mAP@0.4: %.2f' % (train_log['mAP_04_train']))
            print_log(save_path, 'mAP@.5: %.2f' % (train_log['mAP_05_train']))
            print_log(save_path, 'mAP@.6: %.2f' % (train_log['mAP_06_train']))
            print_log(save_path, 'mAP@.7: %.2f' % (train_log['mAP_07_train']))
        scheduler.step()
        
        if args.wandb:
            wandb.log(train_log)
        if epoch % args.test_freq == 0:
            print_log(save_path, '----- %s at epoch #%d' % ('Test', epoch))
            test_log = test_one_epoch(args, test_dataset, test_loader, model, criterion, optimizer, epoch, device)
               
            if epoch % args.test_eval_step == 0:
                print_log(save_path, 'mAP: %.2f' % (test_log['mAP_test']))
                print_log(save_path, 'mAP@.3: %.2f' % (test_log['mAP_03_test']))
                print_log(save_path, 'mAP@0.4: %.2f' % (test_log['mAP_04_test']))
                print_log(save_path, 'mAP@.5: %.2f' % (test_log['mAP_05_test']))
                print_log(save_path, 'mAP@.6: %.2f' % (test_log['mAP_06_test']))
                print_log(save_path, 'mAP@.7: %.2f' % (test_log['mAP_07_test']))
            if args.wandb:
                wandb.log(test_log)

        if max_mAP < test_log['mAP_test']:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'criterion_dict': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            result_path = save_path + '/best_epoch%d.pth' % epoch
            
            # remove previous epoch model
            pth_files = [file for file in os.listdir(save_path) if '.pth' in file]
            
            for f in pth_files:
                f = os.path.join(save_path, f)
                os.remove(f)                        
            torch.save(state, result_path)
            max_mAP = test_log['mAP_test']
            
@torch.no_grad()
def eval(args):
    args.training = False
    save_path = args.save_path
    test_dataset = THUMOS14Dataset(args, subset='test')
                
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True,drop_last=False) 
        
    model = build_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)
    criterion = build_criterion(args, device)
    
    checkpoint = torch.load(args.model_path)

    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_dict) 

    epoch = checkpoint['epoch']

    model.eval()
    criterion.eval()
    
    memory_initialize(model, args)

    metric_logger = loggers.MetricLogger(mode="evaluation", delimiter="  ")
    header = 'Evaluation Inference: '
    
    print_freq = len(test_loader)
    
    # output path
    proposal_file = args.proposal_path.format({}, 'eval', str(epoch))
    proposal_txt_path = os.path.join(save_path, (proposal_file+'.txt'))
    att_cnt = 0
    for i, (inputs, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        inputs, targets, infos = parrallel_collate_fn(inputs, targets, infos, args.p_videos)
        inputs = inputs.to(device) # batch x seq lens x feature size
        bs, seq_len, _ = inputs.shape
        targets = {k: v.to(device) for k, v in targets.items()}
        inputs = {
            "inputs": inputs,
            "infos": infos
        }
        
        # compute output
        outputs = model(inputs, device)
        _loss_dict = criterion(outputs, targets, infos, device)
        loss_dict = {k: v for k,v in _loss_dict.items()}
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        
        if args.make_output:
            make_txt(args, infos, outputs, proposal_txt_path, test_dataset.label_name)
    
    proposal_json_path = os.path.join(args.save_path, (proposal_file+'.json')).format('pred')
    proposal_pred_txt_path = proposal_txt_path.format('pred')
    
    result_dict = online_nms(args, proposal_pred_txt_path, test_dataset)
    output_dict={"version":"VERSION 1", "results": result_dict, "external_data": {}}
    
    outfile=open(proposal_json_path, "w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    tiou_thresholds=np.linspace(0.3,0.70,5)
    
    mAP = evaluation_detection(args, proposal_json_path, subset='test', 
                                tiou_thresholds=tiou_thresholds, verbose=True)
    
    metric_logger.update(mAP=mAP.mean())
    metric_logger.update(mAP_03=mAP[0])
    metric_logger.update(mAP_04=mAP[1])
    metric_logger.update(mAP_05=mAP[2])
    metric_logger.update(mAP_06=mAP[3])
    metric_logger.update(mAP_07=mAP[4])
    
    print("Averaged stats:", metric_logger)
    
    eval_log = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    print_log(save_path, 'mAP: %.2f' % (eval_log['mAP']))
    print_log(save_path, 'mAP@.3: %.2f' % (eval_log['mAP_03']))
    print_log(save_path, 'mAP@0.4: %.2f' % (eval_log['mAP_04']))
    print_log(save_path, 'mAP@.5: %.2f' % (eval_log['mAP_05']))
    print_log(save_path, 'mAP@.6: %.2f' % (eval_log['mAP_06']))
    print_log(save_path, 'mAP@.7: %.2f' % (eval_log['mAP_07']))

def train_one_epoch(args, train_dataset, train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    criterion.train()
    
    memory_initialize(model, args)
    
    args.training = True
    # logger
    metric_logger = loggers.MetricLogger(mode="train", delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=2, fmt='{value:.6f}'))
    # 최대 epoch 크기에 맞게 space padding
    space_fmt =str(len(str(args.epochs)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch, end_epoch=args.epochs,
                                                                 fill=space_fmt)
    print_freq = len(train_loader)
    
    # output path
    proposal_file = args.proposal_path.format({},'train', str(epoch))
    proposal_txt_path = os.path.join(args.save_path, (proposal_file+'.txt'))

    for i, (inputs, targets, infos) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        inputs, targets, infos = parrallel_collate_fn(inputs, targets, infos, args.p_videos)
        inputs = inputs.to(device) # batch x seq lens x feature size
        bs, seq_len, _ = inputs.shape
        targets = {k: v.to(device) for k, v in targets.items()}
            
        inputs = {
            "inputs": inputs,
            "infos": infos
        }

        outputs = model(inputs, device)
        _loss_dict = criterion(outputs, targets, infos, device)
        loss_weight = criterion.weight_dict
        
        loss = sum(_loss_dict[k] * loss_weight[k] for k in _loss_dict.keys() if k in loss_weight)

        loss_dict = {k: v for k,v in _loss_dict.items()}
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * loss_weight[k] for k, v in loss_dict_reduced.items() if k in loss_weight}
        losses_reduced_scaled =sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        
        if args.wandb:
            wandb.log(loss_dict_reduced_scaled)
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        # compute gradient and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if args.make_output:
            make_txt(args, infos, outputs, proposal_txt_path, train_dataset.label_name)
    
    if epoch % args.train_eval_step == 0:
        proposal_json_path = os.path.join(args.save_path, (proposal_file+'.json')).format('pred')
        proposal_pred_txt_path = proposal_txt_path.format('pred')

        result_dict = online_nms(args, proposal_pred_txt_path, train_dataset)
        output_dict={"version":"VERSION 1", "results": result_dict, "external_data": {}}
        
        outfile=open(proposal_json_path, "w")
        json.dump(output_dict,outfile, indent=2)
        outfile.close()
        
        tiou_thresholds=np.linspace(0.3,0.70,5)
        mAP = evaluation_detection(args, proposal_json_path, subset='train', 
                                    tiou_thresholds=tiou_thresholds, verbose=True)        
    else:        
        mAP = np.array([0,0,0,0,0], dtype=np.float)

    metric_logger.synchronize_between_processes()  

    metric_logger.update(mAP_train=mAP.mean())
    metric_logger.update(mAP_03_train=mAP[0])
    metric_logger.update(mAP_04_train=mAP[1])
    metric_logger.update(mAP_05_train=mAP[2])
    metric_logger.update(mAP_06_train=mAP[3])
    metric_logger.update(mAP_07_train=mAP[4])
    
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(args, test_dataset, test_loader, model, criterion, optimizer, epoch, device):
    model.eval()
    criterion.eval()
    
    memory_initialize(model, args)
    
    args.training = False
    metric_logger = loggers.MetricLogger(mode="test", delimiter="   ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=2, fmt='{value:.6f}'))
    
    space_fmt =str(len(str(args.epochs)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch, end_epoch=args.epochs,
                                                                 fill=space_fmt)
    print_freq = len(test_loader)
    
    # output path
    proposal_file = args.proposal_path.format({},'test', str(epoch))
    proposal_txt_path = os.path.join(args.save_path, (proposal_file+'.txt'))
    for i, (inputs, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        
        inputs, targets, infos = parrallel_collate_fn(inputs, targets, infos, args.p_videos)
        inputs = inputs.to(device) # batch x seq lens x feature size
        bs, seq_len, _ = inputs.shape
        targets = {k: v.to(device) for k, v in targets.items()}
        inputs = {
            "inputs": inputs,
            "infos": infos
        }
        
        # compute output
        outputs = model(inputs, device)
        _loss_dict = criterion(outputs, targets, infos, device)
        loss_weight = criterion.weight_dict
        
        loss = sum(_loss_dict[k] * loss_weight[k] for k in _loss_dict.keys() if k in loss_weight)
        
        loss_dict = {k: v for k,v in _loss_dict.items()}
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * loss_weight[k] for k, v in loss_dict_reduced.items() if k in loss_weight}
        losses_reduced_scaled =sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        
        if args.wandb:
            wandb.log(loss_dict_reduced_scaled)
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
            
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if args.make_output:
            make_txt(args, infos, outputs, proposal_txt_path, test_dataset.label_name)
    
    if epoch % args.test_eval_step == 0:
        proposal_json_path = os.path.join(args.save_path, (proposal_file+'.json')).format('pred')
        proposal_pred_txt_path = proposal_txt_path.format('pred')
        
        result_dict = online_nms(args, proposal_pred_txt_path, test_dataset)
        output_dict={"version":"VERSION 1", "results": result_dict, "external_data": {}}

        outfile=open(proposal_json_path, "w")
        json.dump(output_dict,outfile, indent=2)
        outfile.close()
        
        tiou_thresholds=np.linspace(0.3,0.70,5)
        mAP = evaluation_detection(args, proposal_json_path, subset='test', 
                                    tiou_thresholds=tiou_thresholds, verbose=True)
        # import pdb;pdb.set_trace()
        ###`
    else:
        mAP = np.array([0,0,0,0,0], dtype=np.float)
    
    metric_logger.synchronize_between_processes()  
    
    metric_logger.update(mAP_test=mAP.mean())
    metric_logger.update(mAP_03_test=mAP[0])
    metric_logger.update(mAP_04_test=mAP[1])
    metric_logger.update(mAP_05_test=mAP[2])
    metric_logger.update(mAP_06_test=mAP[3])
    metric_logger.update(mAP_07_test=mAP[4])
    
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}