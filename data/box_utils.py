import torch
import itertools
from torchvision.ops import box_iou, box_convert


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
                cx = (i + 0.5)/fk
                cy = (j + 0.5)/fk
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
        self.anchor_box = torch.Tensor(bxs)

    def get_anchor(self):
        return self.anchor_box


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


def match_one_img(anchor, one_img_bbxs, one_img_label, result_tensor_box, result_tensor_class, index, input_format='cxcywh'):
    # the input must be [x,y,w,h]
    anchor_temp = box_convert(
        anchor, in_fmt=input_format, out_fmt='xyxy').clamp_(min=0, max=1)
    one_img_bbxs = box_convert(
        one_img_bbxs, in_fmt=input_format, out_fmt='xyxy').clamp_(min=0, max=1)
    IoUs = box_iou(anchor_temp, one_img_bbxs)  # num_anchors x num_bbxs
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
    result_tensor_box[index] = encode_coordinate(anchor_temp, matched_bbxs)
    one_img_cls_idx = IoUbest <= 0.5
    one_img_cls_idx[idx1] = False
    result_tensor_class[index][one_img_cls_idx] = 0
    result_tensor_class[index][idx1] = one_img_label

    # output: num_anchors x 4: regressor label, num_anchors: class_label


def match_batch(anchor, img_bbxs, img_labels, input_format='cxcywh', device='cuda:0'):
    assert len(img_bbxs) == len(img_labels)
    batch = len(img_bbxs)
    result_bbxs = torch.zeros((batch, anchor.shape[0], 4))
    result_labels = torch.zeros((batch, anchor.shape[0]), dtype=torch.int64)
    for i in range(len(img_bbxs)):
        match_one_img(anchor, img_bbxs[i], img_labels[i],
                      result_bbxs, result_labels, i, input_format)
    ret = {"bbxs": result_bbxs.to(device), "clss": result_labels.to(device)}
    return ret


if __name__ == "__main__":
    pass
    # import pandas as pd
    # from dataprep import ImageData
    # import torchvision
    # import numpy as np
    # import PIL.Image as Image
    # from torch_snippets import *
    # ROOT = '../fastRCNNdata'
    # dataset = {
    #     'imgs_path': ROOT+'/images/images',
    #     'annotations': ROOT + '/df.csv',
    #     'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    #     'candidate_boxes_path': ROOT + '/images/candidates',
    #     'candidate_boxes_class': ROOT + '/images/classes',
    #     'candidate_boxes_delta': ROOT + '/images/delta',
    #     'num_workers': 8,
    #     'IoU_threshold': 0.35,
    #     'train_ratio': 0.7,
    #     'test_ratio': 0.15,
    #     'weight_decay': 0.0005,
    # }
    # a = AnchorBox()
    # default_bxs = a.get_anchor()
    # df_raw = pd.read_csv(dataset['annotations'])
    # test_ds = ImageData(df_raw, df_raw, dataset['imgs_path'], phase='test')
    # img_tensor1, a1, b1, img_path1 = test_ds[12]
    # img_tensor2, a2, b2, img_path2 = test_ds[128]
    # b_im = torch.stack([img_tensor1, img_tensor2])
    # a = [a1, a2]
    # b = [b1, b2]
    # result = match_batch(default_bxs, a, b)

    # deltas = result['bbxs'].squeeze()[1]
    # label = result['clss'].squeeze()[1]
    # idx = torch.where(label != 0)[0]
    # match_bbxs = decode_coordinate(default_bxs, deltas)
    # match_bbxs = torchvision.ops.box_convert(
    #     match_bbxs[idx], in_fmt='cxcywh', out_fmt='xyxy')
    # match_bbxs = match_bbxs
    # label = label[idx]
    # h, w, _ = np.array(Image.open(img_path2)).shape
    # fin = match_bbxs.clamp_(0, 1)*torch.tensor([w, h, w, h])
    # show(Image.open(img_path2), bbs=fin, texts=[
    #      test_ds.idx2class[i.item()] for i in label])
    # plt.savefig('test_img/test_match_box.png')
