# -*- coding: utf-8 -*-

from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image
from pathlib import Path
import cv2
import h5py
from tqdm import tqdm
from googlenet import googlenet

class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img

transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

video_dir = '/Users/sameal/Downloads/Datasets/TVSum/ydata-tvsum50-v1_1/video'
h5_path = '/Users/sameal/Downloads/Datasets/TVSum/fcsn_data.h5'
eccv16_data = h5py.File('/Users/sameal/Downloads/Datasets/TVSum/datasets/eccv16_dataset_tvsum_google_pool5.h5')
net = googlenet(pretrained=True)
fea_net = nn.Sequential(*list(net.children())[:-2])

def video2fea(video_path, h5_f):
    video = cv2.VideoCapture(video_path)
    idx = video.split('.')[0]
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = length//320
    fea = []
    label = []
    usr_sum = eccv16_data['video_'+idx]['user_summary'][()]
    usr_sum = (usr_sum.sum(axis=0) > 10)
    cps = eccv16_data['video_'+idx]['change_points'][()]
    n_frame_per_seg = eccv16_data['video_'+idx]['n_fram_per_seg'][()]
    i = 0
    success, frame = video.read()
    while success:
        if (i+1) % ratio == 0:
            fea.append(fea_net(transform(Image.fromarray(frame)).unsqueeze(0)).squeeze())
            label.append(usr_sum[i])
        i += 1
        success, frame = video.read()
    fea = torch.stack(fea)
    label = torch.stack(fea)
    l = (fea.shape[0]-320)//2
    fea = fea[l:l+320]
    label = label[l:l+320]
    v_data = h5_f.create_group('video_'+idx)
    v_data['feature'] = fea.numpy()
    v_data['label'] = label.numpy()
    v_data['change_points'] = cps
    v_data['n_frame_per_seg'] = n_frame_per_seg
    print('video', idx, 'saved\n', fea.shape, label.shape)


def make_dataset(video_dir, h5_path):
    video_dir = Path(video_dir)
    video_list = video_dir.glob('mp4')
    with h5py.File(h5_path, 'w') as h5_f:
        for video_path in tqdm(video_list, desc='Video', ncols=80, leave=False):
            video2fea(video_path, h5_f)