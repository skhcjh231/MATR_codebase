import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment
from .matcher import HungarianMatcher

class CrossEntropyLoss(nn.Module):
    def __init__(self, focal=False, weight=None, reduce=True):
        super(CrossEntropyLoss, self).__init__()
        self.focal = focal
        self.weight= weight
        self.reduce = reduce

    def forward(self, input, target, alpha=0.25, gamma=2):
        #IN: input: unregularized logits [B, C] target: multi-hot representaiton [B, C]
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        if not self.focal:
            if self.weight is None:
                output = torch.sum(-target * logsoftmax(input), 1)
            else:
                output = torch.sum(-target * logsoftmax(input) /self.weight, 1)
        else:
            softmax = nn.Softmax(dim=1).to(input.device)
            p = softmax(input)
            output = alpha*torch.sum(-target * (1 - p)**gamma * logsoftmax(input), 1)
            
        if self.reduce:
            return torch.mean(output)
        else:
            return output

def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()

    sp, ep = -input_offsets[:, 0], input_offsets[:, 1]
    sg, eg = -target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    skis = torch.max(sp, sg)
    ekis = torch.min(ep, eg)
    
    # iou
    intsctk = ekis - skis
    unionk = (ep-sp) + (eg - sg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)
    
    # smallest enclosing box
    sc = torch.min(sp, sg)
    ec = torch.max(ep, eg)
    len_c = ec - sc
    
    # offset between centers
    rho = abs(0.5 * (ep+sp-eg-sg))
    
    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
    
class CriterionMATR(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, args=None):
        """ Create the criterion.
        Parameters:
        num_classes: number of object categories, omitting the special no-object category
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.reduce = args.reduce
        self.segment_size = args.num_frame
        self.reduction = 'mean' if self.reduce == 1 else 'none'
        self.matcher = matcher
        self.use_flag = args.use_flag
        self.num_queries = args.num_queries
        self.anti_len = args.anti_len
        self.max_memory_len = args.max_memory_len
        
        empty_weight = None
        if args.use_empty_weight:
            empty_weight = torch.ones(self.num_classes)
            empty_weight[-1] = args.eos_coef
                
        self.register_buffer('empty_weight', empty_weight)
        self.cls_loss_func = CrossEntropyLoss(focal=args.use_focal, weight=empty_weight, reduce=self.reduce)
    
    def sel_regoffset(self, fg_src_regs, fg_src_stcls):
        fg_src_stsel = torch.argmax(fg_src_stcls, dim=1)
        fg_src_stregs = fg_src_regs[:,:self.max_memory_len+2]
        fg_src_streg = torch.gather(fg_src_stregs, 1, fg_src_stsel.unsqueeze(-1)).squeeze()
        fg_src_edreg = fg_src_regs[:,-1].squeeze()
        fg_src_regs = torch.stack([fg_src_streg, fg_src_edreg], dim=-1)
        return fg_src_regs
    
    def edrstc2r2offset(self, regs, stcls):
        stsel = torch.argmax(stcls, dim=1)
        st_offset = (regs[:,0]+stsel)*self.segment_size
        ed_offset = regs[:,1]*self.segment_size
        return torch.stack([st_offset, -ed_offset], dim=-1)
    
    def loss_cls(self, outputs, targets, infos, indices, log=True):
        """ classification loss """
        assert 'pred_cls' in outputs
        src_logits = outputs['pred_cls'] # batch x num_queries x num_class
        bs, num_queries, _ = src_logits.shape
        targets = torch.cat([targets['cls_label'][b][indices[b][1]] for b in range(bs)]).reshape(bs, num_queries, -1)
        size = targets.size()
        src_logits = src_logits.reshape(-1, src_logits.size(-1))
        targets = targets.reshape(-1, targets.size(-1))
            
        loss = self.cls_loss_func(src_logits, targets)
        
        if not self.reduce:
            loss = loss.reshape(size[:-1])
            
        if (loss.isnan()):
            loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        
        losses = {'loss_cls': loss}
        
        if self.use_flag:
            src_logits = outputs['pred_flag'].squeeze() # batch
            targets = infos['segment_flag'].float().to(self.device) # batch
            loss = F.binary_cross_entropy(src_logits, targets, reduction=self.reduction)
            losses['loss_flag'] = loss
        return losses

    def loss_reg(self, outputs, targets, infos, indices, log=True):
        """ regression loss """
        assert 'pred_reg' in outputs
        src_regs = outputs['pred_reg']
        bs, num_queries, _ = src_regs.shape
        src_stcls = outputs['pred_stcls']
        
        tgt_regs = torch.cat([targets['reg_label'][b][indices[b][1]] for b in range(bs)]).reshape(bs, num_queries, -1)
        tgt_stcls = torch.cat([targets['stcls_label'][b][indices[b][1]] for b in range(bs)]).reshape(bs, num_queries, -1)
        
        src_regs = src_regs.reshape(-1, src_regs.size(-1))
        src_stcls = src_stcls.reshape(-1, src_stcls.size(-1))
        tgt_regs = tgt_regs.reshape(-1, tgt_regs.size(-1))
        tgt_stcls = tgt_stcls.reshape(-1, tgt_stcls.size(-1))
        
        bgmask = tgt_regs[:,1] < -1e3
        fg_src_regs, bg_src_regs = src_regs[~bgmask], src_regs[bgmask]
        fg_src_stcls, bg_src_stcls = src_stcls[~bgmask], src_stcls[bgmask]
        fg_tgt_regs, bg_tgt_regs = tgt_regs[~bgmask], tgt_regs[bgmask]
        fg_tgt_stcls, bg_tgt_stcls = tgt_stcls[~bgmask], tgt_stcls[bgmask]
        
        loss_stcls = self.cls_loss_func(fg_src_stcls, fg_tgt_stcls)
        
        fg_src_regs = self.sel_regoffset(fg_src_regs, fg_src_stcls)
        fg_src_regs, fg_tgt_regs = fg_src_regs.reshape(-1,2), fg_tgt_regs.reshape(-1,2)
        loss_l1 = F.l1_loss(fg_src_regs, fg_tgt_regs)
        fg_src_offsets = self.edrstc2r2offset(fg_src_regs, fg_src_stcls)
        fg_tgt_offsets = self.edrstc2r2offset(fg_tgt_regs, fg_tgt_stcls)
        
        loss_diou = ctr_diou_loss_1d(fg_src_offsets, fg_tgt_offsets, reduction=self.reduction)            
                
        if(loss_l1.isnan()):
            loss_l1 = torch.tensor([0.0], requires_grad=True).to(self.device)

        if(loss_diou.isnan()):
            loss_diou = torch.tensor([0.0], requires_grad=True).to(self.device)
            
        losses = {'loss_reg_l1': loss_l1,
                  'loss_reg_diou': loss_diou}
        if(loss_stcls.isnan()):
            loss_stcls = torch.tensor([0.0], requires_grad=True).to(self.device)

        losses['loss_reg_stcls'] = loss_stcls
        return losses
    
    def get_loss(self, loss, outputs, targets, infos, indices, **kwargs):
        loss_map = {
            'cls_loss': self.loss_cls,
            'reg_loss': self.loss_reg
        }
        
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, infos, indices, **kwargs)
    
    def forward(self, outputs, targets, infos, device, log=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self.device = device
        indices = self.matcher(outputs, targets, self.device)        
        
        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, infos, indices))
            
        return losses 