import warnings
from sklearn.model_selection import train_test_split
import torchvision
import albumentations as A
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('svg')
warnings.filterwarnings("ignore", category=UserWarning)


class RandomExpand(object):
    def __init__(self, p, mean=(104, 117, 123)) -> None:
        super().__init__()
        self.p = p
        self.mean = mean

    def __call__(self, img, bbxs):
        prop = np.random.choice([0, 1], p=[1-self.p, self.p])
        if prop == 0:
            return img, bbxs
        else:
            height, width, _ = img.shape
            ratio = np.random.uniform(1, 3)
            left = np.random.uniform(0, ratio*width - width)
            top = np.random.uniform(0, ratio*height-height)
            blanc_img = np.zeros(
                shape=(int(ratio*height), int(ratio*width), 3), dtype=np.uint8)
            blanc_img[:, :, :] = np.mean(img, axis=(0, 1))
            blanc_img[int(top):int(top+height), int(left):int(left+width), :] = img
            bx = bbxs.clone()
            bx = bx*torch.tensor([width, height, width, height]) + \
                torch.tensor([left, top, 0, 0])
            new_w = int(ratio*width)
            new_h = int(ratio*height)
            bx /= torch.tensor([new_w, new_h, new_w, new_h])
            return blanc_img, bx


class RandomFlip(object):
    def __init__(self, p) -> None:
        self.p = p

    def __call__(self, img, bbxs):
        prop = np.random.choice([0, 1], p=[1-self.p, self.p])
        if prop == 0:
            return img, bbxs
        else:
            img = img[:, ::-1, :]
            bbxs[:, 0] = 1-bbxs[:, 0]
            return img, bbxs


class RandomCrop(object):
    def __init__(self, p, max_step=50) -> None:
        super().__init__()
        self.p = p
        self.max_step = max_step

    def __call__(self, img, bbxs, labels):
        prop = np.random.choice([0, 1], p=[1-self.p, self.p])
        if prop == 0:
            return img, bbxs, labels
        else:
            success_flag = 0
            height, width, _ = img.shape
            min_ious = [0.1, 0.3, 0.7, 0.9]
            i = 0
            min_iou = np.random.choice(min_ious)
            while success_flag == 0 and i < self.max_step:
                w = np.random.uniform(0.3*width, width)
                h = np.random.uniform(0.3*height, height)
                if h/w < 0.5 or h/w > 2:
                    i += 1
                    continue
                left = np.random.uniform(0, width-w)
                top = np.random.uniform(0, height-h)
                reg = torch.tensor(
                    [left, top, w, h])/torch.tensor([width, height, width, height]).view(-1, 4)
                reg_tem = torchvision.ops.box_convert(
                    reg, 'xywh', 'xyxy').clamp(min=0, max=1)
                bx_tem = torchvision.ops.box_convert(
                    bbxs, 'cxcywh', 'xyxy').clamp(min=0, max=1)
                IoUmat = torchvision.ops.box_iou(reg_tem, bx_tem)
                if torch.amax(IoUmat) > min_iou:
                    center_x = bbxs[:, 0]
                    center_y = bbxs[:, 1]
                    con_x = (reg_tem[0, 0] <= center_x) & (
                        center_x <= reg_tem[0, 2])
                    con_y = (reg_tem[0, 1] <= center_y) & (
                        center_y <= reg_tem[0, 3])
                    idx = con_x & con_y
                    get_bbxs = bbxs[idx, :]
                    tem_label = []
                    for i in range(idx.shape[0]):
                        if idx[i].item() is True:
                            tem_label.append(labels[i])
                    if get_bbxs.shape[0] != 0:
                        success_flag = 1
                        origin_wh = get_bbxs * \
                            torch.tensor([width, height, width, height])
                        reg_tem = reg_tem * \
                            torch.tensor([width, height, width, height])
                        lw = torch.maximum(
                            origin_wh[:, 0]-origin_wh[:, 2]/2, torch.tensor(reg_tem[:, 0]))
                        rw = torch.minimum(
                            origin_wh[:, 0]+origin_wh[:, 2]/2, torch.tensor(reg_tem[:, 2]))
                        th = torch.maximum(
                            origin_wh[:, 1]-origin_wh[:, 3]/2, torch.tensor(reg_tem[:, 1]))
                        bh = torch.minimum(
                            origin_wh[:, 1]+origin_wh[:, 3]/2, torch.tensor(reg_tem[:, 3]))
                        cx = ((lw+rw)/2 - left)
                        cy = ((bh+th)/2 - top)
                        cx = cx/w
                        cy = cy/h
                        nw = (rw - lw)/w
                        nh = (bh - th)/h
                        cropped_bbxs = torch.stack([cx, cy, nw, nh], dim=1)
                        return img[int(top):int(top+h), int(left):int(left+w), :], cropped_bbxs, tem_label
                i += 1
            if success_flag == 0:
                return img, bbxs, labels


def prep_process_img_train(img, bbxs, labels):

    transf = transforms.Compose([transforms.ColorJitter(0.3, 0.2, 0.1, 0.1)])

    trans = transforms.Compose([transforms.Resize((300, 300)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    rd_ex = RandomExpand(0.5)
    rd_crop = RandomCrop(0.5)
    rd_flip = RandomFlip(0.5)
    transed = np.array(transf(img))
    transed, bx = rd_ex(transed, bbxs)
    transed, bx, ret_labels = rd_crop(transed, bx, labels)
    transed, bx = rd_flip(transed, bx)

    return trans(Image.fromarray(transed)), bx, ret_labels


def prep_process_img_test(img):
    trans = transforms.Compose([transforms.Resize((300, 300)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    return trans(img)


def class_idxs_convert(df):
    class2idx = df['LabelName'].unique().squeeze()
    class2idx = {i: t+1 for t, i in enumerate(class2idx)}
    class2idx['background'] = 0
    return class2idx


def split_dataset(df, seed=None, valid_ratio=None):
    unique_id = df['ImageID'].unique().tolist()
    trainsize = round(len(unique_id)*0.7)
    testsize = round(len(unique_id)*0.15)
    validsize = len(unique_id)-(trainsize+testsize)
    if valid_ratio:
        validsize = round(len(unique_id)*valid_ratio)
    train_id, test_id = train_test_split(
        unique_id, random_state=seed, train_size=trainsize, test_size=testsize+validsize)
    valid_id, test_id = train_test_split(
        test_id, random_state=seed, train_size=validsize, test_size=testsize)
    df_train = df[df['ImageID'].isin(train_id)]
    df_valid = df[df['ImageID'].isin(valid_id)]
    df_test = df[df['ImageID'].isin(test_id)]
    return df_train, df_valid, df_test


class ImageData(Dataset):
    def __init__(self, df, df_raw, img_folder, phase='train'):
        self.img_folder = img_folder
        self.unique_id = df['ImageID'].unique().tolist()
        self.clss2idx = class_idxs_convert(df_raw)
        self.idx2class = {i: t for t, i in self.clss2idx.items()}
        self.df = df
        self.phase = phase
        if phase == 'train':
            self.augmet = prep_process_img_train
        else:
            self.augmet = prep_process_img_test

    def __len__(self,):
        return len(self.unique_id)

    def __getitem__(self, index):
        unique_id = self.unique_id[index]
        img_path = self.img_folder+'/' + unique_id + '.jpg'
        bbxs = torch.Tensor(self.df[self.df['ImageID'] ==
                                    unique_id]['XMin,YMin,XMax,YMax'.split(',')].values)  # Numbox x4
        _bbxs = torchvision.ops.box_convert(
            bbxs, in_fmt='xyxy', out_fmt='cxcywh')
        class_name = self.df[self.df['ImageID']
                             == unique_id]['LabelName'].values
        classes = torch.tensor([self.clss2idx[t] for t in class_name])
        if self.phase == 'train':
            img, _bbxs, classes = self.augmet(Image.open(
                img_path).convert('RGB'), _bbxs, classes.numpy())
            classes = torch.Tensor(classes)
        else:
            img = self.augmet(Image.open(img_path).convert('RGB'))
        return img, _bbxs, classes.to(torch.int64), img_path

    def collate_fn(self, batch):
        img, _bbxs, classes, _ = zip(*batch)
        b_img = torch.stack(img)
        return b_img, _bbxs, classes


if __name__ == '__main__':
    pass
    import pandas as pd
    from torch_snippets import *
    ROOT = './archive'
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

    df_raw = pd.read_csv(dataset['annotations'])
    df_train, df_valid, df_test = split_dataset(df_raw, 10)
    train_ds = ImageData(df_train, df_raw, dataset['imgs_path'], phase='train')
    test_ds = ImageData(df_test, df_raw, dataset['imgs_path'], phase='train')
    img1, bx, clss, _ = test_ds[7]
    bx = torchvision.ops.box_convert(bx, 'cxcywh', 'xyxy')*300
    show(img1.permute(1, 2, 0), bbs=bx, texts=[
         train_ds.idx2class[i.item()] for i in clss])
    plt.savefig('test_img/my_data_prep.png')
