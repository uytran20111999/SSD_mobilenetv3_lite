import matplotlib.pyplot as plt
from scripts.train_one_batch import train_batch, valid_batch
from torch_snippets import *
import matplotlib
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
matplotlib.use('svg')


def train(model, optimizer, anchors, train_loader,
          valid_loader, n_epochs,
          bx_match_func, loss, evaluate_metric,
          save_path, device, save_plot_path=None):
    max_metric = 0
    scheduler = CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.0001)
    log = Report(n_epochs)
    model = model.to(device)
    for epoch in range(n_epochs):
        model = model.train()
        _n = len(train_loader)
        for ix, data in enumerate(train_loader):
            imgs, bbxs, clss = data
            imgs = imgs.to(device)
            with torch.no_grad():
                matched_label = bx_match_func(anchors, bbxs, clss, device=device)
            train_data = (imgs, matched_label)
            cur_loss, bbxs_loss, clss_loss = train_batch(
                model, loss, train_data, optimizer)
            pos = (epoch + (ix+1)/_n)
            log.record(pos, trn_loss=cur_loss,
                       trn_conf_loss=clss_loss,
                       trn_regr_loss=bbxs_loss, end='\r')
            scheduler.step(epoch + ix / _n)
        _n = len(valid_loader)
        avg_eval_val = 0
        with torch.no_grad():
            model = model.eval()
            for ix, data in enumerate(valid_loader):
                imgs, bbxs, clss = data
                matched_label = bx_match_func(
                    anchors, bbxs, clss, device=device)
                valid_data = (imgs.to(device), matched_label, bbxs, clss)
                cur_loss, bbxs_loss, clss_loss, step_map = valid_batch(
                    model, loss, valid_data, evaluate_metric)
                pos = (epoch + (ix+1)/_n)
                log.record(pos, val_loss=cur_loss,
                           val_conf_loss=clss_loss,
                           val_regr_loss=bbxs_loss,
                           step_map=step_map, end='\r')
                avg_eval_val += step_map
            if max_metric < avg_eval_val/_n:
                max_metric = avg_eval_val/_n
                print("update best weight")
                torch.save(model.state_dict(), save_path)
            log.report_avgs(epoch+1)
        if epoch == 1:
            model.feature_extractor.unfreeze_base()
    if save_plot_path is not None:
        log.plot_epochs('trn_loss,val_loss,step_map'.split(','))
        plt.savefig(save_plot_path)


def test_model(model, anchors,device,test_loader,bx_match_func,evaluate_metric,loss):
    with torch.no_grad():
        preds_bxs = []
        preds_lb = []
        true_bxs = []
        true_label = []
        model = model.eval()
        for ix, data in enumerate(test_loader):
            imgs, bbxs, clss = data
            matched_label = bx_match_func(
                anchors, bbxs, clss, device=device)
            imgs, target, truth_bbxs, truth_labels = (imgs.to(device=device), matched_label, bbxs, clss)
            preds = model(imgs)
            preds_bbxs, preds_labels = preds['regression'], preds['classification']
            target_bbxs, target_labels,priors = target['bbxs'], target['clss'],target['anchor']
            #print(evaluate_metric(preds_bbxs, preds_labels, truth_bbxs, truth_labels))
            preds_bxs.append(preds_bbxs)
            preds_lb.append(preds_labels)
            true_bxs.extend(truth_bbxs)
            true_label.extend(truth_labels)
        preds_bbxs = torch.cat(preds_bxs,dim=0)
        preds_labels = torch.cat(preds_lb,dim = 0)
        eval_val = evaluate_metric(preds_bbxs, preds_labels, true_bxs, true_label)
        return eval_val
