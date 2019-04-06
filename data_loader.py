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
        features = torch.Tensor(video['features'][()]).reshape(1024, -1)
        labels = torch.Tensor(video['gtsummary'][()])
        c, l = features.shape
        if l % 32:
            l1 = (32 - l % 32) // 2
            if not l1:
                tf = torch.zeros(c, 1)
                lf = torch.zeros(1)
                features = torch.cat((features, tf), dim=1)
                labels = torch.cat((labels, lf), dim=0)
            else:
                l2 = (32 - l % 32) - l1
                tf1 = torch.zeros(c, l1)
                tf2 = torch.zeros(c, l2)
                lf1 = torch.zeros(l1)
                lf2 = torch.zeros(l2)
                features = torch.cat((tf1, features, tf2), dim=1)
                labels = torch.cat((lf1, labels, lf2), dim=0)
        return features, labels, index
    

def get_loader(path, batch_size=1):
    dataset = VideoData(path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [9*len(dataset)//10, len(dataset)//10])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader, test_dataset


if __name__ == '__main__':
    loader = get_loader('/Users/sameal/')
    import ipdb
    ipdb.set_trace()
