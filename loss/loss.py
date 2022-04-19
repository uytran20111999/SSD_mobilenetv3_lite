# TODO add multibox loss
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
from data.box_utils import log_sum_exp,decode_new,IoG
import torch.nn as nn



def multibox_loss(preds_labels, preds_bbxs, target_labels,
                  target_bbxs, priors,bx_loss=torch.nn.SmoothL1Loss(
        reduction='mean'),
        class_loss=torch.nn.CrossEntropyLoss(reduction='none')):
        loc_t = Variable(target_bbxs, requires_grad=False)
        conf_t = Variable(target_labels, requires_grad=False)
        num = preds_bbxs.size(0)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(preds_bbxs)
        loc_p = preds_bbxs[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = preds_labels.view(-1, 3)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(preds_labels)
        neg_idx = neg.unsqueeze(2).expand_as(preds_labels)
        conf_p = preds_labels[(pos_idx+neg_idx).gt(0)].view(-1, 3)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l+loss_c,loss_l, loss_c


class RepulsionLoss(nn.Module):

    def __init__(self, use_gpu=True, sigma=0.):
        super(RepulsionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.variance = [0.1,0.2]
        self.sigma = sigma
        
    # TODO 
    def smoothln(self, x, sigma=0.):        
        pass

    def forward(self, loc_data, ground_data, prior_data):
        
        decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
        iog = IoG(ground_data, decoded_boxes)
        # sigma = 1
        # loss = torch.sum(-torch.log(1-iog+1e-10))  
        # sigma = 0
        loss = torch.sum(iog)          
        return loss


        return loss_l, loss_l_repul, loss_c