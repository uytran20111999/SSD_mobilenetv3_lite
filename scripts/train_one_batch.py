def train_batch(model, loss, data, optimizer):
    optimizer.zero_grad()
    imgs, target = data
    preds = model(imgs)
    preds_bbxs, preds_labels = preds['regression'], preds['classification']
    target_bbxs, target_labels,priors = target['bbxs'], target['clss'],target['anchor']
    cur_loss, box_loss, class_loss = loss(
        preds_labels, preds_bbxs, target_labels,  target_bbxs,priors)
    cur_loss.backward()
    optimizer.step()
    return cur_loss.cpu().detach().item(), box_loss.detach().cpu().item(), class_loss.detach().cpu().item()


def valid_batch(model, loss, data, evaluate_metric):
    # data comprise of [imgs,{match_bbxs,match_labels},ground truth bbxs,ground_truth_labels]
    imgs, target, truth_bbxs, truth_labels = data
    preds = model(imgs)
    preds_bbxs, preds_labels = preds['regression'], preds['classification']
    target_bbxs, target_labels,priors = target['bbxs'], target['clss'],target['anchor']
    cur_loss, box_loss, class_loss = loss(
        preds_labels, preds_bbxs, target_labels,  target_bbxs,priors)
    eval_val = evaluate_metric(
        preds_bbxs, preds_labels, truth_bbxs, truth_labels)
    return cur_loss.cpu().detach().item(), box_loss.detach().cpu().item(), class_loss.detach().cpu().item(), eval_val
