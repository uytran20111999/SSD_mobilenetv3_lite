# TODO: Add mAP metrics
# use sklearn lib


from torchvision.ops import box_iou, box_convert, nms
import torch
# i get the idea from: https://www.youtube.com/watch?v=FppOzcDvaDI&ab_channel=AladdinPersson


def FP_TP_one_img(preds_bbxs, preds_labels, truth_bbxs, truth_labels, nums_class, iou_threshold=0.5, device='cuda:0'):
    # return dict {class for each class}
    # input format must be xyxy
    d = {}
    gt_d = {}
    max_preds, class_preds = preds_labels.max(dim=1)
    for c in range(1, nums_class):  # 0 denote background
        box_of_c = preds_bbxs[class_preds == c, :].clamp(min=0, max=1)
        TP = torch.zeros(box_of_c.shape[0])
        FP = torch.zeros(box_of_c.shape[0])
        truth_of_c = truth_bbxs[truth_labels == c, :]
        gt_d[c] = truth_of_c.shape[0]
        if truth_of_c.shape[0] == 0:
            if box_of_c.shape[0] == 0:
                d[c] = []
                continue
            FP[:] = 1
            d[c] = torch.stack([max_preds[class_preds == c],
                               TP.to(device), FP.to(device)], dim=1)
        else:
            if box_of_c.shape[0] == 0:
                d[c] = []
                continue
            iou = box_iou(box_of_c, truth_of_c)  # num_pred_c x num_truth_c
            max_iou, max_iou_idx = iou.max(dim=0)
            true_pt = max_iou >= iou_threshold
            if len(true_pt) > 0:  # remove overlaps
                TP[max_iou_idx[true_pt]] = 1
            FP[TP == 0] = 1
            d[c] = torch.stack([max_preds[class_preds == c],
                               TP.to(device), FP.to(device)], dim=1)
    return d, gt_d


def FP_TP_batch(preds_bbxs, preds_labels, truth_bbxs, truth_labels, anchors, decode_box, nums_class, iou_threshold=0.5, in_fmt='cxcywh', device='cuda:0'):
    assert preds_bbxs.shape[0] == len(truth_bbxs)
    anchors = anchors.to(device)
    out_bbxs = []
    preds_map = []
    prop_clss, pred_clss = preds_labels.max(dim=2)
    pos_pred_clss = pred_clss != 0
    for i in range(preds_bbxs.shape[0]):
        converted_bx = box_convert(decode_box(
            anchors, preds_bbxs[i]), in_fmt, 'xyxy').clamp(0, 1)
        if torch.any(pos_pred_clss[i]):
            # cleaned background preds
            after_nms = nms(
                converted_bx[pos_pred_clss[i]], prop_clss[i][pos_pred_clss[i]], 0.5)
            out_bbxs.append(converted_bx[pos_pred_clss[i]][after_nms, :])
            preds_map.append(preds_labels[i][pos_pred_clss[i]][after_nms, :])
        else:
            out_bbxs.append(converted_bx)
            preds_map.append(preds_labels[i])

    dglob = {}
    gt_glob = {}
    for i in range(1, nums_class):
        dglob[i] = []
        gt_glob[i] = 0
    for i in range(len(truth_bbxs)):
        truth_bbxs_cvt = box_convert(
            truth_bbxs[i], in_fmt, 'xyxy').clamp(0, 1).to(device)
        d, gt_d = FP_TP_one_img(
            out_bbxs[i], preds_map[i], truth_bbxs_cvt, truth_labels[i], nums_class, iou_threshold, device=device)
        for k in d.keys():
            if len(d[k]) > 0:
                dglob[k].append(d[k])
            gt_glob[k] += gt_d[k]
    for i in range(1, nums_class):
        if len(dglob[i]) == 0:
            continue
        dglob[i] = torch.cat(dglob[i])
    return dglob, gt_glob


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
    dglob, gt_glob = FP_TP_batch(preds_bbxs, preds_labels, truth_bbxs, truth_labels,
                                 anchors, decode_box, nums_class, iou_threshold=iou_threshold, in_fmt=in_fmt)
    return mAP(dglob, gt_glob)


if __name__ == "__main__":
    a = torch.tensor([[1, 2], [3, 4]])
    print(a)
    print(a[[False, True]])
