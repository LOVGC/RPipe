import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts


class SVHN(Dataset):
    data_name = 'SVHN'
    file = [('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', 'e26dedcc434d2e4c54c9b2d4a06d8373'),
            ('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', 'eb5a983be6a315427106f1b164d9cef3'),
            ('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat', 'a93ce644f1a588dc4d68dda5feec44a7')]

    def __init__(self, root, split, process=False, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder) or process:
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, self.split))
        self.other = {}
        self.classes_counts = make_classes_counts(self.target)
        self.data_size, self.target_size, self.classes_to_labels = load(os.path.join(self.processed_folder, 'meta'))

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), Image.fromarray(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**input, **other}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, extra_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train'))
        save(test_set, os.path.join(self.processed_folder, 'test'))
        save(extra_set, os.path.join(self.processed_folder, 'extra'))
        save(meta, os.path.join(self.processed_folder, 'meta'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, os.path.join(self.raw_folder, filename), md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_data, train_target = read_data_file(os.path.join(self.raw_folder, 'train_32x32.mat'))
        test_data, test_target = read_data_file(os.path.join(self.raw_folder, 'test_32x32.mat'))
        extra_data, extra_target = read_data_file(os.path.join(self.raw_folder, 'extra_32x32.mat'))
        train_id, test_id, extra_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(
            np.int64), np.arange(len(extra_data)).astype(np.int64)
        data_size = (3, 32, 32)
        classes = list(map(str, list(range(10))))
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        target_size = len(classes)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (
            extra_id, extra_data, extra_target), (data_size, target_size, classes_to_labels)


def read_data_file(path):
    import scipy.io as sio
    loaded_mat = sio.loadmat(path)
    img = loaded_mat['X']
    label = loaded_mat['y'].astype(np.int64).squeeze()
    img = np.transpose(img, (3, 0, 1, 2))
    label[label == 10] = 0
    return img, label
