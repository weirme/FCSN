# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

import h5py


class VideoData(object):
    """Dataset class"""
    def __init__(self, data_path):
        self.data_file = h5py.File(data_path)

    def __len__(self):
        return len(self.data_file)
        
    def __getitem__(self, index):
        video = self.data_file['video_'+str(index.item()+1)]
        feature = torch.tensor(video['feature'][()]).reshape(1024, -1)
        label = torch.tensor(video['label'][()], dtype=torch.long)
        return feature, label, index
    

def get_loader(path, batch_size=5):
    dataset = VideoData(path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [9*len(dataset)//10, len(dataset)//10])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader, test_dataset


if __name__ == '__main__':
    loader = get_loader('fcsn_dataset.h5')
    import ipdb
    ipdb.set_trace()
