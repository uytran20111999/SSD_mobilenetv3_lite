# TODO: Add mAP metrics
# use sklearn lib


from torchvision.ops import box_iou, box_convert, nms
import torch
from collections import Counter
# i get the idea from: https://www.youtube.com/watch?v=FppOzcDvaDI&ab_channel=AladdinPersson
import numpy as np



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:  #VOC在2010之后换了评价方法，所以决定是否用07年的
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):  #  07年的采用11个点平分recall来计算
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 取一个recall阈值之后最大的precision
            ap = ap + p / 11.  # 将11个precision加和平均
    else:  # 这里是用2010年后的方法，取所有不同的recall对应的点处的精度值做平均，不再是固定的11个点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))  #recall和precision前后分别加了一个值，因为recall最后是1，所以
        mpre = np.concatenate(([0.], prec, [0.])) # 右边加了1，precision加的是0

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #从后往前，排除之前局部增加的precison情况

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 这里巧妙的错位，返回刚好TP的位置，
                                                                                      # 可以看后面辅助的例子

        # and sum (\Delta recall) * prec   用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def FP_TP_batch(preds_bbxs, preds_labels, truth_bbxs, truth_labels, anchors, decode_box, num_classes, iou_threshold=0.5, in_fmt='cxcywh', device='cuda:0'):

    #preds_bbxs: [B x num_anchors x 4] decode cx cy w h
    #preds_labels: [B x num_anchors x num_class]
    #truth_bbxs: [[num_truth_box(i) x 4] for i in B] cx cy w h
    #truth_labels: [[num_truth_box(i)] for i in B]

    assert preds_bbxs.shape[0] == len(truth_bbxs)
    anchors = anchors.to(device)
    out_bbxs = []
    preds_map = []
    prop_clss, pred_clss = preds_labels.max(dim=-1) # B x num_anchors
    pos_pred_clss = pred_clss != 0

    #for each predicted batch
    for i in range(preds_bbxs.shape[0]):
        #convert the predicted boxes to true coordinates (xyxy)
        converted_bx = box_convert(decode_box(preds_bbxs[i],anchors
             ), in_fmt, 'xyxy').clamp_(0, 1)
        #if there is any predicted class is positive (eg. not back ground)
        #if torch.any(pos_pred_clss[i]):
            # cleaned background preds
        after_nms = nms(
            converted_bx[pos_pred_clss[i]], prop_clss[i][pos_pred_clss[i]], 0.6)
        out_bbxs.append([converted_bx[pos_pred_clss[i]][after_nms, :]])
        preds_map.append([prop_clss[i][pos_pred_clss[i]][after_nms],pred_clss[i][pos_pred_clss[i]][after_nms]])

    #out_bbxs as well as preds_map has the same len as truth_bbxs
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(1,num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        # for (detection,predict_label) in zip(out_bbxs,pred_clss):
        #     if predict_label == c:
        #         detections.append(detection)
        for i, (cur_img_pred_bbxs, cur_img_pred_label) in enumerate(list(zip(out_bbxs,preds_map))):
            for idx,cur_pred_bbxs in enumerate(cur_img_pred_bbxs):
                if cur_pred_bbxs.shape[0] and cur_img_pred_label[1][idx]==c:
                    detections.append([i,cur_img_pred_label[0][idx],cur_img_pred_label[1][idx],cur_pred_bbxs])



        # for (true_box,true_label) in zip(truth_bbxs,truth_labels):
        #     if true_label == c:
        #         ground_truths.append(true_box)

        for i, (cur_img_true_bbxs, cur_img_true_label) in enumerate(list(zip(truth_bbxs,truth_labels))):
            for cur_true_bbxs,cur_true_label in zip(cur_img_true_bbxs,cur_img_true_label):
                if cur_true_label==c:
                    ground_truths.append([i,cur_true_label,box_convert(cur_true_bbxs,in_fmt=in_fmt,out_fmt='xyxy').clamp_(0, 1)])

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                # tmp1 = detection[3].unsqueeze(0)
                # tmp2 = gt[2].unsqueeze(0)
                iou = box_iou(
                    detection[3].to(device=device),
                    gt[2].unsqueeze(0).to(device=device),
                )[0,0]

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(voc_ap(recalls,precisions))

    return sum(average_precisions) / len(average_precisions)

    


        


def mAP(dglob, gt_glob):
    ret_map = {}
    eps = 0.0000001
    for k in dglob.keys():
        ret_map[k] = None
    for i in dglob.keys():
        if len(dglob[i]) == 0:
            if gt_glob[i] != 0:
                ret_map[k] = 0
            continue
        conf = dglob[i][:, 0]
        _, idx_sorted = torch.sort(conf, descending=True)
        # if len(idx_sorted) == 0:
        #     ret_map[k] = 0
        #     continue
        TP_FP = dglob[i][:, 1:]  # num_preds x 2
        TP_FP = TP_FP[idx_sorted, :]
        TP_FP_cumsum = torch.cumsum(TP_FP, dim=0)
        num_preds = TP_FP_cumsum[-1, :].sum()
        num_truth = gt_glob[i]
        recall = TP_FP_cumsum[:, 0]/(num_truth+eps)
        precision = TP_FP_cumsum[:, 0]/(num_preds+eps)
        recall = torch.cat([torch.tensor([0]).cuda(), recall])
        precision = torch.cat([torch.tensor([1]).cuda(), precision])

        ret_map[i] = torch.trapz(precision, recall)
    i = 0
    my_s = 0
    for k in ret_map.keys():
        if ret_map[k] is not None:
            my_s += ret_map[k]
            i += 1
    return my_s/(i)


def wrapper_mAP(preds_bbxs, preds_labels, truth_bbxs, truth_labels, anchors, decode_box, nums_class, iou_threshold=0.5, in_fmt='cxcywh'):
    return FP_TP_batch(preds_bbxs, preds_labels, truth_bbxs, truth_labels,
                                 anchors, decode_box, nums_class, iou_threshold=iou_threshold, in_fmt=in_fmt)
    # return mAP(dglob, gt_glob)


if __name__ == "__main__":
    a = torch.tensor([[1, 2], [3, 4]])
    print(a)
    print(a[[False, True]])

