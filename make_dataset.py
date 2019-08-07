from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image
from pathlib import Path
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, help='directory containing mp4 file of specified dataset.', default='../data/TVSum_video')
parser.add_argument('--h5_path', type=str, help='save path of the generated dataset, which should be a hdf5 file.', default='../data/fcsn_tvsum.h5')
parser.add_argument('--vsumm_data', type=str, help='preprocessed dataset path from this repo: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce, which should be a hdf5 file. We copy cps and some other info from it.', default='../data/eccv_datasets/eccv16_dataset_tvsum_google_pool5.h5')

args = parser.parse_args()
video_dir = args.video_dir
h5_path = args.h5_path
vsumm_data = h5py.File(args.vsumm_data)


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


net = models.googlenet(pretrained=True).float().cuda()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])


def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8);
        recall = overlap / (true_sum + 1e-8);
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)


def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break
    tqdm.write('Overlap: '+str(overlap_arr))
    tqdm.write('True summary n_key: '+str(true_sum_arr))
    tqdm.write('Oracle smmary n_key: '+str(oracle_sum))
    tqdm.write('Final F-score: '+str(best_fscore))
    return oracle_summary


def video2fea(video_path, h5_f):
    video = cv2.VideoCapture(video_path.as_uri())
    idx = video_path.as_uri().split('.')[0].split('/')[-1]
    tqdm.write('Processing video '+idx)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = length//320
    fea = []
    label = []
    usr_sum_arr = vsumm_data['video_'+idx]['user_summary'][()]
    usr_sum = get_oracle_summary(usr_sum_arr) 
    cps = vsumm_data['video_'+idx]['change_points'][()]
    n_frame_per_seg = vsumm_data['video_'+idx]['n_frame_per_seg'][()]
    i = 0
    success, frame = video.read()
    while success:
        if (i+1) % ratio == 0:
            fea.append(fea_net(transform(Image.fromarray(frame)).cuda().unsqueeze(0)).squeeze().detach().cpu())
            try:
                label.append(usr_sum[i])
            except:
                pdb.set_trace()
        i += 1
        success, frame = video.read()
    fea = torch.stack(fea)
    fea = fea[:320]
    label = label[:320]
    v_data = h5_f.create_group('video_'+idx)
    v_data['feature'] = fea.numpy()
    v_data['label'] = label
    v_data['length'] = len(usr_sum)
    v_data['change_points'] = cps
    v_data['n_frame_per_seg'] = n_frame_per_seg
    v_data['picks'] = [ratio*i for i in range(320)]
    v_data['user_summary'] = usr_sum_arr
    if fea.shape[0] != 320 or len(label) != 320:
        print('error in video ', idx, feashape[0], len(label))


def make_dataset(video_dir, h5_path):
    video_dir = Path(video_dir).resolve()
    video_list = list(video_dir.glob('*.mp4'))
    video_list.sort()
    with h5py.File(h5_path, 'w') as h5_f:
        for video_path in tqdm(video_list, desc='Video', ncols=80, leave=False):
            video2fea(video_path, h5_f)
    
    
if __name__ == '__main__':
    make_dataset(video_dir, h5_path)


vsumm_data.close()
