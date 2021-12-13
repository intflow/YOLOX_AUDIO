#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2019-2021 Intflow Inc. All rights reserved.
# --Based on YOLOX made by Megavii Inc.--

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

##class RIOUloss(nn.Module):
##    def __init__(self, reduction="none", loss_type="giou"):
##        super(RIOUloss, self).__init__()
##        self.reduction = reduction
##        self.loss_type = loss_type
##
##    def forward(self, pred, target):
##        assert pred.shape[0] == target.shape[0]
##
##        rad_pred = torch.atan2(pred[:,4],pred[:,5]).unsqueeze(-1)
##        rad_target = torch.atan2(target[:,4],target[:,5]).unsqueeze(-1)
##        _pred = torch.cat((pred[:,:4],rad_pred),dim=-1)
##        _target = torch.cat((target[:,:4],rad_target),dim=-1)
##        _pred = _pred.view(1,-1, 5)
##        _target = _target.view(1,-1, 5)
##        
##        if self.loss_type == "iou": #defualt as DIoU
##            loss, _ = cal_diou(_pred, _target)
##        elif self.loss_type == "giou":
##            loss, _ = cal_giou(_pred, _target)
##
##        if self.reduction == "mean":
##            loss = loss.mean()
##        elif self.reduction == "sum":
##            loss = loss.sum()
##        else:
##            loss = loss[0]
##
##        return loss

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

# Focal loss implementation inspired by
# https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
# https://github.com/doiken23/pytorch_toolbox/blob/master/focalloss2d.py
class MultiClassBCELoss(nn.Module):
    def __init__(self,
                 use_focal_weights=True,
                 focus_param=2,
                 balance_param=0.25,
                 reduction='none'
                 ):
        super().__init__()

        #self.use_weight_mask = use_weight_mask
        self.bcewl = nn.BCEWithLogitsLoss(reduction=reduction)
        self.use_focal_weights = use_focal_weights
        self.focus_param = focus_param
        self.balance_param = balance_param
        
    def forward(self,
                outputs,
                targets):
        ### inputs and targets are assumed to be BatchxClasses
        assert len(outputs.shape) == len(targets.shape)
        assert outputs.size(0) == targets.size(0)
        assert outputs.size(1) == targets.size(1)
        
        ## weights are assumed to be BatchxClasses
        #assert outputs.size(0) == weights.size(0)
        #assert outputs.size(1) == weights.size(1)

        bcewl_loss = self.bcewl(input=outputs,
                                 target=targets)
        
        if self.use_focal_weights:
            logpt = - bcewl_loss
            pt    = torch.exp(logpt)

            focal_loss = -((1 - pt) ** self.focus_param) * logpt
            balanced_focal_loss = self.balance_param * focal_loss
            
            return balanced_focal_loss
        else:
            return bce_loss 


# Focal loss for PSS
class PSSBCELoss(nn.Module):
    def __init__(self,
                 use_focal_weights=True,
                 focus_param=2,
                 balance_param=0.25,
                 reduction='none'
                 ):
        super().__init__()

        #self.use_weight_mask = use_weight_mask
        self.sigm = nn.Sigmoid()
        self.bcewl = nn.BCEWithLogitsLoss(reduction=reduction)
        self.use_focal_weights = use_focal_weights
        self.focus_param = focus_param
        self.balance_param = balance_param
        
    def forward(self,
                outputs,
                targets):

        outputs1, outputs2 = outputs

        ### inputs and targets are assumed to be BatchxClasses
        assert len(outputs1.shape) == len(targets.shape)
        assert outputs1.size(0) == targets.size(0)
        assert outputs1.size(1) == targets.size(1)
        
        
        outputs = torch.logit(torch.sqrt(self.sigm(outputs1) * self.sigm(outputs2) + 1e-6))

        bcewl_loss = self.bcewl(input=outputs,
                                 target=targets)
        
        if self.use_focal_weights:
            logpt = - bcewl_loss
            pt    = torch.exp(logpt)

            focal_loss = -((1 - pt) ** self.focus_param) * logpt
            balanced_focal_loss = self.balance_param * focal_loss
            
            return balanced_focal_loss
        else:
            return bce_loss 