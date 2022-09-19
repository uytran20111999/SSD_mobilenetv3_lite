import torch
import itertools
from torchvision.ops import box_iou, box_convert

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

class AnchorBox(object):
    def __init__(self, s_min=0.2, s_max=0.95, img_size=300,
                 devided_windows=[19, 10, 5, 3, 2, 1], num_boxes=[3, 6, 6, 6, 6, 6],
                 ratio=[1, 2, 3]) -> None:
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.img_size = img_size
        self.devided_windows = devided_windows
        self.num_boxes = num_boxes
        self.ratio = ratio
        # gen_anchor
        next_add = (s_max-s_min)/(len(devided_windows)-1)
        sk = s_min
        bxs = []
        for f_size, num_bbxs in zip(devided_windows, num_boxes):
            step = img_size/f_size
            s_next = sk + next_add
            for i, j in itertools.product(range(f_size), range(f_size)):
                fk = img_size/step
                cx = (j + 0.5)/fk
                cy = (i + 0.5)/fk
                bxs += [[cx, cy, sk, sk]]
                if num_bbxs == 3:
                    ratio = self.ratio[1:-1]
                else:
                    ratio = self.ratio[1:]
                for t in ratio:
                    bxs += [[cx, cy, sk*(t**0.5), sk*(t**-0.5)]]
                    bxs += [[cx, cy, sk*((1/t)**0.5), sk*((1/t)**-0.5)]]
                if num_bbxs == 6:
                    bxs += [[cx, cy, (sk*s_next)**0.5, (sk*s_next)**0.5]]
            sk = s_next
        self.anchor_box = torch.Tensor(bxs).clamp_(min = 0,max = 1)

    def get_anchor(self):
        return self.anchor_box.clamp_(max=1,min = 0)


def encode(matched, priors, variances=[0.1,0.2]):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = matched[:,:2] - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:]) / priors[:, 2:]
    g_wh = torch.log(g_wh + 1e-10) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1) 

def decode_new(loc, priors, variances=[0.1,0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    return boxes

def encode_coordinate(default_boxes, match_boxes):
    # bbxs: Num_anchorsx4 tensor with format (x,y,w,h)
    xy = (match_boxes[:, 0:2]-default_boxes[:, 0:2])/default_boxes[:, 2:]
    wh = torch.log(match_boxes[:, 2:]/default_boxes[:, 2:] + 0.000000001)
    return torch.cat([xy, wh], dim=1)


def decode_coordinate(default_boxes, match_boxes):
    xy = ((match_boxes[:, 0:2]) * default_boxes[:, 2:] +
          default_boxes[:, 0:2])
    wh = (torch.exp(match_boxes[:, 2:])*default_boxes[:, 2:])
    return torch.cat([xy, wh], dim=1)

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def match_one_img(anchor, one_img_bbxs, one_img_label, result_tensor_box, result_tensor_class, index, input_format='cxcywh'):
    # the input must be [x,y,w,h]
    anchor_temp = box_convert(
        anchor, in_fmt=input_format, out_fmt='xyxy').clamp_(min=0, max=1)
    one_img_bbx = box_convert(
        one_img_bbxs, in_fmt=input_format, out_fmt='xyxy').clamp_(min=0, max=1)
    IoUs = box_iou(anchor_temp, one_img_bbx)  # num_anchors x num_bbxs
    #one_img_cls, one_img_offset = [], []
    idx1 = torch.argmax(IoUs, dim=0)  # idx cua defaul_bxs
    # asure every groundtruth has a matching default box
    idx2 = torch.argmax(IoUs, dim=1)  # idx cua truthbxs
    for j, t in enumerate(idx1):
        idx2[t] = j
    IoUbest = IoUs[range(anchor_temp.shape[0]), idx2]
    truth_lab_temp = one_img_label.expand(anchor_temp.shape[0], -1)
    truth_bx_temp = one_img_bbxs.expand(
        anchor_temp.shape[0], -1, -1)  # ndef x nbox x 4
    result_tensor_class[index] = (truth_lab_temp[
        range(anchor_temp.shape[0]), idx2]).to(torch.int64)  # ndef
    matched_bbxs = truth_bx_temp[range(anchor_temp.shape[0]), idx2]
    result_tensor_box[index] = encode(matched_bbxs,anchor)
    one_img_cls_idx = IoUbest <= 0.5
    one_img_cls_idx[idx1] = False
    result_tensor_class[index][one_img_cls_idx] = 0
    result_tensor_class[index][idx1] = one_img_label

    # output: num_anchors x 4: regressor label, num_anchors: class_label

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def match(threshold, truths, priors, labels, loc_t, conf_t, idx,input_format = 'cxcywh'):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    priors = box_convert(
        priors, in_fmt=input_format, out_fmt='xyxy').clamp_(min=0, max=1)
    truths = box_convert(
        truths, in_fmt=input_format, out_fmt='xyxy').clamp_(min=0, max=1)
    overlaps = jaccard(
        point_form(truths),
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap <= threshold] = 0  # label as background
    loc = encode(matches,priors)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

def match_batch(anchor, img_bbxs, img_labels, input_format='cxcywh', device='cuda:0'):
    assert len(img_bbxs) == len(img_labels)
    batch = len(img_bbxs)
    result_bbxs = torch.zeros((batch, anchor.shape[0], 4))
    result_labels = torch.zeros((batch, anchor.shape[0]), dtype=torch.int64)
    for i in range(len(img_bbxs)):
        # match(0.5,img_bbxs[i],anchor, img_labels[i],
        #               result_bbxs, result_labels, i,input_format)
        match_one_img(anchor, img_bbxs[i], img_labels[i], result_bbxs, result_labels, i, input_format='cxcywh')
    ret = {"bbxs": result_bbxs.to(device), "clss": result_labels.to(device),'anchor':anchor}
    return ret

def IoG(box_a, box_b):
    """Compute the IoG of two sets of boxes.  
    E.g.:
        A ∩ B / A = A ∩ B / area(A)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_objects,4]
    Return:
        IoG: (tensor) Shape: [num_objects]
    """
    inter_xmin = torch.max(box_a[:, 0], box_b[:, 0])
    inter_ymin = torch.max(box_a[:, 1], box_b[:, 1])
    inter_xmax = torch.min(box_a[:, 2], box_b[:, 2])
    inter_ymax = torch.min(box_a[:, 3], box_b[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)  
    I = Iw * Ih
    G = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    return I / G

if __name__ == "__main__":
    #pass
    import pandas as pd
    from dataprep import ImageData
    import torchvision
    import numpy as np
    import PIL.Image as Image
    from torch_snippets import *
    ROOT = '../my_data'
    dataset = {
        'imgs_path': ROOT+'/images/images',
        'annotations': ROOT + '/df.csv',
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'candidate_boxes_path': ROOT + '/images/candidates',
        'candidate_boxes_class': ROOT + '/images/classes',
        'candidate_boxes_delta': ROOT + '/images/delta',
        'num_workers': 8,
        'IoU_threshold': 0.35,
        'train_ratio': 0.7,
        'test_ratio': 0.15,
        'weight_decay': 0.0005,
    }
    a = AnchorBox()
    default_bxs = a.get_anchor()
    df_raw = pd.read_csv(dataset['annotations'])
    test_ds = ImageData(df_raw, df_raw, dataset['imgs_path'], phase='train')
    img_tensor1, a1, b1, img_path1 = test_ds[1]
    img_tensor2, a2, b2, img_path2 = test_ds[128]
    b_im = torch.stack([img_tensor1, img_tensor2])
    a = [a1, a2]
    b = [b1, b2]
    result = match_batch(default_bxs, a, b)

    deltas = result['bbxs'].squeeze()[1]
    label = result['clss'].squeeze()[1]
    idx = torch.where(label != 0)[0]
    match_bbxs = decode_new(deltas.cuda(),default_bxs.cuda(),)
    match_bbxs = torchvision.ops.box_convert(
        match_bbxs[idx], in_fmt='cxcywh', out_fmt='xyxy')
    match_bbxs = match_bbxs
    label = label[idx]
    #h, w, _ = np.array(Image.open(img_path1)).shape
    _,h, w = img_tensor2.shape
    fin = match_bbxs.clamp_(0, 1)*torch.tensor([w, h, w, h]).cuda()
    show(img_tensor2.permute(1, 2, 0), bbs=fin, texts=[
         test_ds.idx2class[i.item()] for i in label])
    plt.savefig('test_img/test_match_box.png')