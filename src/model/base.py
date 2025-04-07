import torch
import torch.nn as nn
from .loss import make_loss


class Base(nn.Module):
    """
    这里的 model 就是指计算从输入到 prediction 的那个模型。还没有包含 loss 的计算。
    这个 Base 就是把 loss 给加入到计算图里面，i.e. Base 这个类就是把 loss 和 model 结合在一起。
        我们可以看出来， 计算 loss 的入口是在这个 make_loss 上。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = make_loss

    def forward(self, **input):
        output = {}
        output['pred'] = self.model(input['data'])
        output['loss'] = self.loss(output, input)
        return output


def base(model):
    model = Base(model)
    return model
