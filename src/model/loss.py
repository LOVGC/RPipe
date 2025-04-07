import torch
import torch.nn.functional as F


def make_loss(output, input):
    if 'target' in input:
        # # 这里 output['pred'] 是指 core model 的输出，input['target'] 是指 ground truth。
        loss = loss_fn(output['pred'], input['target'])  
    else:
        return
    return loss


def loss_fn(output, target, reduction='mean'):
    """
    这里 output 是 core model 的输出，target 是 ground truth。
    """
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = kld_loss(output, target, reduction=reduction)
    return loss


def cross_entropy_loss(output, target, reduction='mean'):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction=reduction)
    return ce


def kld_loss(output, target, reduction='batchmean'):
    kld = F.kl_div(F.log_softmax(output, dim=-1), target, reduction=reduction)
    return kld
