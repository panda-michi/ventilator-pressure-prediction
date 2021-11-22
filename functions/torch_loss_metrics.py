import torch.nn.functional as F

def mask_huber_loss(predict, truth, m, delta=0.1):
    loss = F.huber_loss(predict[m], truth[m], delta=delta)
    return loss

def mask_l1_loss(predict, truth, m):
    loss = F.l1_loss(predict[m], truth[m])
    return loss

def mask_smooth_l1_loss(predict, truth, m, beta=0.1):
    loss = F.smooth_l1_loss(predict[m], truth[m], beta=beta)
    return loss