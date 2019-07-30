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
        index += 1
        video = self.data_file['video_'+str(index)]
        feature = torch.tensor(video['feature'][()]).t()
        label = torch.tensor(video['label'][()], dtype=torch.long)
        return feature, label, index
    

def get_loader(path, batch_size=5):
    dataset = VideoData(path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - len(dataset) // 5, len(dataset) // 5])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader, test_dataset


if __name__ == '__main__':
    loader = get_loader('fcsn_dataset.h5')
