import dataset
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from config import cfg


def make_dataset(data_name, transform=True, process=False, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)

    if data_name in ['MNIST', 'FashionMNIST']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", process=process, '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        if transform:
            data_stats = (cfg['model']['stats'].mean.tolist(), cfg['model']['stats'].std.tolist())
            dataset_['train'].transform = dataset.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats)])
            dataset_['test'].transform = dataset.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats)])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", process=process, '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        if transform:
            data_stats = (cfg['model']['stats'].mean.tolist(), cfg['model']['stats'].std.tolist())
            dataset_['train'].transform = dataset.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats)])
            dataset_['test'].transform = dataset.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats)])
    elif data_name in ['SVHN']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", process=process, '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        if transform:
            data_stats = (cfg['model']['stats'].mean.tolist(), cfg['model']['stats'].std.tolist())
            dataset_['train'].transform = dataset.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats)])
            dataset_['test'].transform = dataset.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*data_stats)])
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    first = input[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in input])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in input]))
            else:
                batch[k] = torch.tensor([f[k] for f in input])
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=num_samples,
                                                         generator=generator)
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
        else:
            if k == 'train':
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=shuffle,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
            else:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=False,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset):
    processed_dataset = dataset
    cfg['num_samples'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    cfg['model']['data_size'] = dataset['train'].data_size
    cfg['model']['target_size'] = dataset['train'].target_size
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']
    return processed_dataset
