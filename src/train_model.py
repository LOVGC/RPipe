import argparse
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
from module import check, resume, to_device, process_control

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
# 这个 for loop 就是把原来 cfg 里面的东西都给复制过来加到 parser 里面。
# 这个原来的 cfg 就是存在 src/config.yml 里面的值。
# 这个 for loop 其实就是一堆 parser.add_argument(...)
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
# 这个process_args(args)函数的作用：
# 如果有针对这个实验的 <data name>_<model name>.yml 文件（在 output/config/ 文件夹下）,
# 就 Load 针对这个实验的 <data name>_<model name>.yml 文件，并覆盖全局变量 cfg. 
# 如果用户通过 args 传进来的 configs 跟 <data name>_<model name>.yml 的 configs 不同，那就继续覆盖 <data name>_<model name>.yml 的 configs. 
# 所以优先级是：用户传进来的 args (实验配置参数) > <data name>_<model name>.yml(output/config/ 文件下) > 默认的 config.yml（src/config.yml)
# 总结：这个函数的作用是让用户可以自定义实验的配置参数
# 1. 如果用户没有声明特殊的配置，那么就用 <data name>_<model name>.yml(output/config/ 文件下) 的配置
# 2. 如果用户声明了特殊的配置，那么就用用户声明的配置覆盖 <data name>_<model name>.yml(output/config/ 文件下) 的配置
process_args(args)
# 这里我们弄明白了这些配置文件的关系。
# 这里一个重要的知识是：cfg 这个变量是程序一直在用的。程序有可能根据情况去动态调整它。


def main():
    # 为每一次实验设置一个不同的 seed. 所以这里是一个 list. 生成 seed 的方式也很简单，就是一个 range.
    # 如果只有一个实验，那就只有一个 seed.
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        # tag_list = [seed, control_name]
        tag_list = [str(seeds[i]), cfg['control_name']]

        # cfg['tag'] = '<seed>_<dataset name>_<model_name>', 这个变量存的就是这个实验的名字，用来区分不用实验。
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control() # 对每一个实验进行实验参数配置：模型架构，training protocol, optimizer
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    # 准备实验参数：seed, path, logger 
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    cfg['logger_path'] = os.path.join(cfg['tag_path'], 'logger', 'train')

    # prepare dataset  
    dataset = make_dataset(cfg['data_name'])

    # 这里 process_dataset, 也是在动态地去改变这个 global 的 cfg. 
    # 这里是根据具体的 dataset 的参数去更新这个 cfg 的参数。
    dataset = process_dataset(dataset)

    # prepare model
    model = make_model(cfg['model'])

    # whether to resume training or train from scratch
    result = resume(cfg['checkpoint_path'], resume_mode=cfg['resume_mode'])
    if result is None: # train from scratch
        cfg['step'] = 0
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['tag']]['optimizer'])
        scheduler = make_scheduler(optimizer, cfg[cfg['tag']]['optimizer'])
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
    else:  # resume training
        cfg['step'] = result['cfg']['step']
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['tag']]['optimizer'])
        scheduler = make_scheduler(optimizer, cfg[cfg['tag']]['optimizer'])
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
        model.load_state_dict(result['model'])
        optimizer.load_state_dict(result['optimizer'])
        scheduler.load_state_dict(result['scheduler'])
        logger.load_state_dict(result['logger'])
        logger.reset()
    data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], cfg['num_steps'],
                                   cfg['step'], cfg['step_period'], cfg['pin_memory'], cfg['num_workers'],
                                   cfg['collate_mode'], cfg['seed'])
    data_iterator = enumerate(data_loader['train'])
    while cfg['step'] < cfg['num_steps']:
        # train over the data_iterator once, 就是一个 epoch
        train(data_iterator, model, optimizer, scheduler, logger)
        test(data_loader['test'], model, logger)
        result = {'cfg': cfg, 'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                  'logger': logger.state_dict()}
        check(result, cfg['checkpoint_path'])
        if logger.compare('test'):
            shutil.copytree(cfg['checkpoint_path'], cfg['best_path'], dirs_exist_ok=True)
        logger.reset()
    return


def train(data_loader, model, optimizer, scheduler, logger):
    """ 
    train over the data_loader once.
    """
    model.train(True)
    start_time = time.time()
    with logger.profiler:
        for i, input in data_loader:
            if i % cfg['step_period'] == 0 and cfg['profile']:
                logger.profiler.step()
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(**input)
            loss = 1 / cfg['step_period'] * output['loss']
            loss.backward()
            if (i + 1) % cfg['step_period'] == 0:  # 由此可见，step_period 指的是参数更新的周期。
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            evaluation = logger.evaluate('train', 'batch', input, output)
            logger.append(evaluation, 'train', n=input_size)
            idx = cfg['step'] % cfg['eval_period']
            if idx % max(int(cfg['eval_period'] * cfg['log_interval']), 1) == 0 and (i + 1) % cfg['step_period'] == 0:
                step_time = (time.time() - start_time) / (idx + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(
                    seconds=round((cfg['eval_period'] - (idx + 1)) * step_time))
                exp_finished_time = datetime.timedelta(
                    seconds=round((cfg['num_steps'] - (cfg['step'] + 1)) * step_time))
                info = {'info': ['Model: {}'.format(cfg['tag']),
                                 'Train Epoch: {}({:.0f}%)'.format((cfg['step'] // cfg['eval_period']) + 1,
                                                                   100. * idx / cfg['eval_period']),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time),
                                 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                logger.append(info, 'train')
                print(logger.write('train'))
            if (i + 1) % cfg['step_period'] == 0:
                cfg['step'] += 1
            if (idx + 1) % cfg['eval_period'] == 0 and (i + 1) % cfg['step_period'] == 0:
                break
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        model.train(False)
        num_steps = len(data_loader) if cfg['eval']['num_steps'] == -1 else cfg['eval']['num_steps']
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(**input)
            evaluation = logger.evaluate('test', 'batch', input, output)
            logger.append(evaluation, 'test', input_size)
            logger.add('test', input, output)
            if (i + 1) == num_steps:
                break
        evaluation = logger.evaluate('test', 'full')
        logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Test Epoch: {}({:.0f}%)'.format(cfg['step'] // cfg['eval_period'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test'))
        logger.save(True)
    return


if __name__ == "__main__":
    main()
