from .criterion import CriterionMATR
from .matcher import build_matcher

def build_criterion(args, device):
    losses = ['cls_loss', 'reg_loss']
    weight_dict = {}
    weight_dict['loss_cls'] = args.cls_coef
    weight_dict['loss_reg_l1'] = args.reg_l1_coef
    weight_dict['loss_reg_diou'] = args.reg_diou_coef
    if args.use_flag:
        weight_dict['loss_flag'] = args.flag_coef
    weight_dict['loss_reg_stcls'] = args.reg_stcls_coef

        
    matcher = build_matcher(args)
    criterion = CriterionMATR(num_classes=args.num_of_class, matcher=matcher,
                                    weight_dict=weight_dict, losses=losses, args=args)
    criterion.to(device)
    
    return criterion