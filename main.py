from models.ssd import SSDLite
from loss.loss import *
from loss.metric import *
from scripts.train import *
from data.dataprep import *
from config.config import ROOT, dataset_configs
from data.box_utils import *
import pandas as pd
from functools import partial
from ranger import Ranger
if __name__ == "__main__":
    df_raw = pd.read_csv(dataset_configs['annotations'])
    df_train, df_valid, df_test = split_dataset(df_raw, 10)
    my_model = SSDLite(num_class=3).cuda()
    optimizer = Ranger(my_model.parameters(),
                          lr=0.001333333)
    n_epochs = 300
    train_ds = ImageData(
        df_train, df_raw, dataset_configs['imgs_path'], phase='train')
    val_ds = ImageData(
        df_valid, df_raw, dataset_configs['imgs_path'], phase='valid')
    train_loader = DataLoader(train_ds, 32, num_workers=8,
                              pin_memory=True, collate_fn=train_ds.collate_fn, shuffle=True)
    valid_loader = DataLoader(val_ds, 16, num_workers=4,
                              pin_memory=True, collate_fn=val_ds.collate_fn, shuffle=False)
    default_anchors = AnchorBox().get_anchor()
    wrapper_mAP1 = partial(wrapper_mAP,
                           decode_box=decode_new, nums_class=3, anchors=default_anchors)
    # train(my_model, optimizer, default_anchors, train_loader=train_loader,
    #       valid_loader=valid_loader, n_epochs=n_epochs,
    #       bx_match_func=match_batch, loss=multibox_loss, evaluate_metric=wrapper_mAP1,
    #       save_path='./test8.pth', device='cuda:0', save_plot_path='./train_plot.png')
    # -------------------------------------------------test inference---------------------------------------------------
    my_model.load_state_dict(torch.load('test8.pth'))
    my_model = my_model.eval().cuda()
    test_ds = ImageData(
        df_test, df_raw, dataset_configs['imgs_path'], phase='test')
    img_tensor, a, b, img_path = test_ds[2]
    img_tensor = img_tensor[None]
    out = my_model(img_tensor.cuda())
    _deltas, _clss = out['regression'], out['classification'].softmax(dim=-1)
    default_anchors = default_anchors.cuda()
    coord = decode_new(_deltas.squeeze(),default_anchors)
    def_box = torchvision.ops.box_convert(
        coord, in_fmt='cxcywh', out_fmt='xyxy')  # def * numclss
    conf, idxs = _clss.squeeze().max(dim=-1)
    pos = idxs != 0
    conf_pos = conf[pos]
    def_pos = def_box[pos]
    idx = nms(def_pos, conf_pos, 0.3)
    h, w, _ = np.array(Image.open(img_path)).shape
    fin = def_pos[idx].clamp_(min=0, max=1)*torch.tensor([w, h, w, h]).cuda()
    k = 5
    text = ['{}:{:.2f}'.format(train_ds.idx2class[i.item()], j)
            for i, j in zip(idxs[pos][idx], conf_pos[idx])]
    show(Image.open(img_path), bbs=fin, texts=text)
    plt.savefig('test_img/test_predict.png')
