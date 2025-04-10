import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    # data_names = ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']
    # data_names = ['MNIST', 'CIFAR10']
    data_names = ['MNIST']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    process_control()
    with torch.no_grad():
        for data_name in data_names:
            dataset = make_dataset(data_name, transform=False, process=True) # 核心
            dataset['train'].transform = Compose([transforms.ToTensor()])
            process_dataset(dataset) # 对 dataset 没有什么改变，就是根据 dataset，在 cfg 变量里面加入信息。
            cfg['step'] = 0
            data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], shuffle=False)
            stats = Stats(dim=dim) # 生成关于这个 dataset 的统计信息。
            for i, input in enumerate(data_loader['train']):
                stats.update(input['data'])
            print(data_name, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}'.format(data_name)), 'torch')
