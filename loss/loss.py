# TODO add multibox loss
import torch
import math
import torch.nn.functional as F


def multibox_loss(preds_labels, preds_bbxs, target_labels,
                  target_bbxs, bx_loss=torch.nn.SmoothL1Loss(
        reduction='sum'),
        class_loss=torch.nn.CrossEntropyLoss(reduction='none')):
    pos_mask = torch.where(target_labels != 0, 1, 0).unsqueeze(-1)
    num_not_neg = torch.sum(pos_mask)
    if num_not_neg == 0:
        reg_loss = torch.tensor([0]).cuda()
        cls_loss = torch.tensor([0]).cuda()
    else:
        reg_loss = bx_loss(
            preds_bbxs*pos_mask, target_bbxs*pos_mask)/num_not_neg
        cls_loss = class_loss(
            preds_labels.permute(0, 2, 1), target_labels)  # Bxn_anchorxC -> BxCxn_anchors
        pos_cls = (pos_mask.squeeze(-1)*cls_loss).sum()
        neg_cls, _ = ((1-pos_mask.squeeze(-1)) *
                      cls_loss).sort(descending=True, dim=-1)  # hard negative minings with (num_pos/num_negs)=1/3
        num_neg = min(3*num_not_neg, len(neg_cls)-num_not_neg)
        neg_cls = neg_cls[:, :num_neg]
        cls_loss = (pos_cls + neg_cls.sum())/num_not_neg
    return reg_loss + cls_loss, reg_loss.detach(), cls_loss.detach()
