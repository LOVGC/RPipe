import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model
from module import save, resume, to_device, process_control

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]

        # cfg['tag'] = '<seed>_<dataset name>_<model_name>', 这个变量存的就是这个实验的名字，用来区分不用实验。
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp') # output/exp 存的是 train and test 的实验结果。
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag']) # output/exp/<seed>_<dataset name>_<model_name> 存的是用这个 seed,dataset,model 做实验的结果。
    
    # 存的是这个实验<seed>_<dataset name>_<model_name>的 checkpoint
    #     这个 checkpoint 好像是一个 temp checkpoint, 用来暂时存一下 cfg, logger, model, optimizer, scheduler 的状态。
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint') 
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best') # 存的是这个实验<seed>_<dataset name>_<model_name>的 best
    cfg['logger_path'] = os.path.join(cfg['tag_path'], 'logger', 'test') # 存的是这个实验<seed>_<dataset name>_<model_name>的 test log
    cfg['result_path'] = os.path.join('output', 'result', cfg['tag']) # 存的是这个实验<seed>_<dataset name>_<model_name>的result
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset) # 根据 dataset 长度，更新 cfg 里面的参数。
    model = make_model(cfg['model'])
    result = resume(cfg['best_path'])  # load the result of the best model.
    if result is None:
        raise ValueError('No valid model, please train model first')
    cfg['step'] = result['cfg']['step']
    model = model.to(cfg['device'])
    model.load_state_dict(result['model'])
    data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'])
    test_logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
    test(data_loader['test'], model, test_logger)
    result = resume(cfg['checkpoint_path']) 
    result = {'cfg': cfg, 'logger': {'train': result['logger'],
                                     'test': test_logger.state_dict()}}
    save(result, cfg['result_path']) # 存的是 test result, 在 output/result/0_MNIST_linear
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(**input)
            
            # 这里 evaluation 是在算在这个 batch 上的平均 loss 和 accuracy
            evaluation = logger.evaluate('test', 'batch', input, output)

            # 然后把这个 batch 的 loss 和 accuracy 加到 logger 里面去
            logger.append(evaluation, 'test', input_size)
            logger.add('test', input, output)
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
