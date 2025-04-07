from config import cfg
from .stats import make_stats


def process_control():
    """
    这个 process_control() 就是在动态地修改这个 cfg 参数。
    这个函数加载的配置参数有：
        关于模型架构的
        关于 training protocol 的
        关于 optimizer 的
        这些参数具体是做什么的，还要慢慢读。
    """
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']

    cfg['batch_size'] = 250
    cfg['step_period'] = 1 # 每次训练多少个 step 就做一次操作，至于是什么操作，可能是更新 learning rate, 也可能是更新统计数据
    cfg['num_steps'] = 80000 # 一个 step 就是模型在一个 batch 数据上完成一次前向传播和后向传播的过程。

    cfg['eval_period'] = 200
    cfg['eval'] = {}
    cfg['eval']['num_steps'] = -1
    cfg['num_epochs'] = 400
    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet10'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['model']['stats'] = make_stats(cfg['control']['data_name'])

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'SGD'
    cfg[tag]['optimizer']['lr'] = 1e-1
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['nesterov'] = True # Nesterov 是一种改进的动量优化方法。
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
