import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import _LRScheduler
import json

COLOR_MAP = ['#f03e3e', '#7048e8', '#1c7ed6', '#0ca678', '#f59f00', '#ae3ec9', '#24f7ef']

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def calc_iou(a, b):
    st = a[0]-a[1]
    ed = a[0]
    target_st = b[0]-b[1]
    target_ed = b[0]
    sst = min(st, target_st)
    led = max(ed, target_ed)
    lst = max(st, target_st)
    sed = min(ed, target_ed)

    iou = (sed-lst) / max(led-sst,1)
    return iou

def box_include(y, target): #is target is the larger box than y?
    st = y[0]-y[1]
    ed = y[0]
    target_st = target[0]-target[1]
    target_ed = target[0]
    
    detection_point = target_st #(target_st+target_ed)/2.0
    
    if ed > detection_point and target_st < st and target_ed > ed:
        return True
    return False    

def frame_include(ed, target):
    target_st = target[0]-target[1]
    target_ed = target[0]
    if ed > target_st and target_ed > ed:
        return True
    return False

def print_log(result_path, *args):
    os.makedirs(result_path, exist_ok=True)

    print(*args)
    file_path = result_path + '/log.txt'
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)

def sel_regoffset(args, regs, stcls):
    stsel = torch.argmax(stcls, dim=2)
    stregs = regs[:,:,:args.max_memory_len+2]
    streg = torch.gather(stregs, 2, stsel.unsqueeze(-1)).squeeze()
    edreg = regs[:,:,-1]
    return torch.stack([streg, edreg], dim=-1)

def make_txt(args, infos, outputs, file_path, label_map):
    pred_path = file_path.format('pred')
    
    pred_inst_clses = outputs['pred_cls'] # batch x queries x num classes
    pred_inst_reges = outputs['pred_reg'] # batch x queries x (memory len +2)
    pred_inst_stcls = outputs['pred_stcls'] # batch x queries x memory len + 2
    
    act_func = nn.Softmax(dim=1).to(outputs['pred_cls'].device)
    
    Path(pred_path).touch()
    
    pred_inst_reges = sel_regoffset(args, pred_inst_reges, pred_inst_stcls)
    
    for b in range(len(outputs['pred_cls'])):
        vid = infos['video_name'][b]
        fid = infos['current_frame'][b]
        frame_to_time = infos['frame_to_time'][b]
        f_inst_clses = pred_inst_clses[b] # queries x num classes
        f_inst_reges = pred_inst_reges[b] # queries x 2
        f_inst_stcls = pred_inst_stcls[b] # queries x (memory len + 2) 
        f_cls_prob = act_func(f_inst_clses)
        num_queries = args.num_queries
        f_preds = []
        for idx in range(num_queries):
            cls = torch.argmax(f_cls_prob[idx][:-1], dim=0).reshape(-1)
            if f_cls_prob[idx][cls]<args.cls_threshold:
                continue
            st_reg, ed_reg = f_inst_reges[idx]
            stcls= f_inst_stcls[idx]
            stsel = torch.argmax(stcls, dim=0).reshape(-1)
            st_offset = (stsel+st_reg)*args.num_frame
            ed_offset = ed_reg*args.num_frame
            
            st = fid - st_offset
            ed = fid - ed_offset
            
            st = st * frame_to_time
            ed = ed * frame_to_time
            cur = fid * frame_to_time
            
            cls_label = label_map[cls]
            cls_prob = f_cls_prob[idx][cls]
            
            f_preds.append([str(vid), round(float(cur),2), round(float(st),2), round(float(ed),2), str(cls_label), round(float(cls_prob),4)])
        if len(f_preds) == 0:
            continue
        
        f_preds = non_max_suppression(f_preds, args.nms_threshold)
        
        for f_pred in f_preds:
            sp = "   "
            pred_list = f_pred
            str_to_be_added = [str(k) for k in pred_list]
            str_to_be_added = (sp.join(str_to_be_added))
            f = open(pred_path, "a+")
            f.write(str_to_be_added + "\r\n")
            f.close()

def non_max_suppression(proposal_dict, nms_threshold):
    final_proposal_dict = []
    sorted_proposals = sorted(proposal_dict, key=lambda proposal:float(proposal[5]), reverse=True)
    idx=0
    total_proposal=len(sorted_proposals)
    while idx < total_proposal:
        proposal = sorted_proposals[idx]
        
        st = float(proposal[2])
        ed = float(proposal[3])
        label = proposal[4]
        
        delete_item = []
        for j in range(idx+1, total_proposal):
            target_proposal = sorted_proposals[j]
            target_st = float(target_proposal[2])
            target_ed = float(target_proposal[3])
            target_label = target_proposal[4]
            
            if label == target_label:
                sst = np.minimum(st, target_st)
                led = np.maximum(ed, target_ed)
                lst = np.maximum(st, target_st)
                sed = np.minimum(ed, target_ed)
                
                tiou = (sed-lst) / max(led-sst,1)
                if tiou > nms_threshold:
                    delete_item.append(target_proposal)
        
        for item in delete_item:
            sorted_proposals.remove(item)
        total_proposal=len(sorted_proposals)
        idx+=1
    
    return sorted_proposals

def online_nms(args, pred_path, dataset):
    pred_anns = []
    sp = "   "
    with open(pred_path, 'r') as file:
        for line in file.readlines():
            row = line[:-1].split(sp)
            pred_anns.append(row)
    
    result_dict={}
    
    threshold = args.nms_threshold
    
    for video_name in dataset.video_list:
        video_proposal_dict = []
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = video_time/duration
        
        v_pred_proposals = np.array([line for line in pred_anns if line[0] == video_name])
        
        if len(v_pred_proposals) == 0:
            result_dict[video_name] =[]
            continue
        
        v_pred_proposals_valid = [line for line in v_pred_proposals if np.float(line[1])>= np.float(line[3])]

        if len(v_pred_proposals_valid) == 0:
            result_dict[video_name] =[]
            continue
            
        v_pred_proposals = np.array(v_pred_proposals_valid).tolist()
        
        # non max suppression
        sorted_proposals = sorted(v_pred_proposals, key=lambda proposal:(float(proposal[1]), -float(proposal[5])))
        idx=0
        total_proposal=len(sorted_proposals)
        while idx < total_proposal:
            proposal = sorted_proposals[idx]
            st = float(proposal[2])
            ed = float(proposal[3])
            label = proposal[4]
            
            delete_item = []
            for j in range(idx+1, total_proposal):
                target_proposal = sorted_proposals[j]
                target_st = float(target_proposal[2])
                target_ed = float(target_proposal[3])
                target_label = target_proposal[4]
                
                if label == target_label:
                    sst = np.minimum(st, target_st)
                    led = np.maximum(ed, target_ed)
                    lst = np.maximum(st, target_st)
                    sed = np.minimum(ed, target_ed)
                    
                    tiou = (sed-lst) / max(led-sst,1)
                    if tiou > threshold:
                        delete_item.append(target_proposal)
            
            for item in delete_item:
                sorted_proposals.remove(item)
            total_proposal=len(sorted_proposals)
            idx+=1
        
        for proposal in sorted_proposals:
            tmp_dict = {}
            tmp_dict['segment'] = [float(proposal[2]), float(proposal[3])]
            tmp_dict['score'] = float(proposal[5])
            tmp_dict['label'] = proposal[4]
            tmp_dict['gentime'] = float(proposal[1])
            video_proposal_dict.append(tmp_dict)
        
        result_dict[video_name]=video_proposal_dict
    
    return result_dict
    
def memory_initialize(model, args):
    model.module.memory_queue = None
    model.module.memory_queue_index = None
        
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
def parrallel_collate_fn(inputs, targets, infos, p_videos):
    inputs = torch.stack(inputs, dim=0).squeeze()
    inputs = inputs.reshape(-1,inputs.size(-2), inputs.size(-1))
    if p_videos == 1:
        targets = targets[0]
        infos = infos[0]
    else:
        _targets = {}
        _infos = {}
        for key in targets[0].keys():    
            _targets[key] = targets[0][key]
            for i in range(1, p_videos):
                _targets[key] = torch.cat([_targets[key], targets[i][key]], dim=0)
        for key in infos[0].keys():
            if str(type(infos[0][key])) != "<class 'list'>":
                _infos[key] = infos[0][key].tolist()
            else:
                _infos[key] = infos[0][key]
            for i in range(1, p_videos):
                if str(type(infos[0][key])) != "<class 'list'>":
                    infos[i][key] = infos[i][key].tolist()                    
                _infos[key] = _infos[key] + infos[i][key]
            if str(type(infos[0][key])) != "<class 'list'>":
                _infos[key] = torch.tensor(_infos[key])
        targets = _targets
        infos = _infos
    return inputs, targets, infos