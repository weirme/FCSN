import json
import csv
import h5py
import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description='Generate keyshots, keyframes and score bar.')
parser.add_argument('--h5_path', type=str, help='path to hdf5 file that contains information of a dataset.', default='../data/fcsn_tvsum.h5')
parser.add_argument('-j', '--json_path', type=str, help='path to json file that stores pred score output by model, it should be saved in score_dir.', default='score_dir/epoch-49.json')
parser.add_argument('-r', '--data_root', type=str, help='path to directory of original dataset.', default='../data/TVSum')
parser.add_argument('-s', '--save_dir', type=str, help='path to directory where generating results should be saved.', default='Results')
parser.add_argument('-b', '--bar', action='store_true', help='whether to plot score bar.')

args = parser.parse_args()
h5_path = args.h5_path
json_path = args.json_path
data_root = args.data_root
save_dir = args.save_dir
bar = args.bar
video_dir = os.path.join(data_root, 'ydata-tvsum50-v1_1', 'video')
anno_path = os.path.join(data_root, 'ydata-tvsum50-v1_1', 'data', 'ydata-tvsum50-anno.tsv')
f_data = h5py.File(h5_path)
with open(json_path) as f:
    json_dict = json.load(f)
    ids = json_dict.keys()


def get_keys(id):
    video_info = f_data['video_' + id]
    video_path = os.path.join(video_dir, id+'.mp4')
    cps = video_info['change_points'][()]
    pred_score = json_dict[id]['pred_score']
    pred_selected = json_dict[id]['pred_selected']

    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    frames = np.array(frames)
    keyshots = []
    for sel in pred_selected:
        for i in range(cps[sel][0], cps[sel][1]):
            keyshots.append(frames[i])
    keyshots = np.array(keyshots)

    write_path = os.path.join(save_dir, id, 'summary.avi')
    video_writer = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc(*'XVID'), 24, keyshots.shape[2:0:-1])
    for frame in keyshots:
        video_writer.write(frame)
    video_writer.release()

    keyframe_idx = [np.argmax(pred_score[cps[sel][0] : cps[sel][1]]) + cps[sel][0] for sel in pred_selected]
    keyframes = frames[keyframe_idx]

    keyframe_dir = os.path.join(save_dir, id, 'keyframes')
    os.mkdir(keyframe_dir)
    for i, img in enumerate(keyframes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(os.path.join(keyframe_dir, '{}.jpg'.format(i)))


def plot_bar():
    with open(anno_path) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        csv_dict = {}
        idx = 0
        for row in csv_reader:
            score = np.array([int(i) for i in row[2].split(',')])
            if str(idx//20+1) in ids:
                if idx % 20 == 0:
                    csv_dict[str(idx//20+1)] = score/20
                else:
                    csv_dict[str(idx//20+1)] += score/20
            idx += 1
    
    sns.set()
    fig, ax = plt.subplots(ncols=1, nrows=len(ids), figsize=(30, 20))
    fig.tight_layout()
    for id, axi in zip(ids, ax.flat):
            scores = csv_dict[id]
            pred_summary = json_dict[id]['pred_summary']
            axi.bar(left=list(range(len(scores))), height=scores, color=['lightseagreen' if i == 0
                else 'orange' for i in pred_summary], edgecolor=None)
            axi.set_title(id)
    save_path = os.path.join(save_dir, 'result-bar.png')
    plt.savefig(save_path)


def gen_summary():
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for id in ids:
        os.mkdir(os.path.join(save_dir, id))
        get_keys(id)
    
    if bar:
        plot_bar()


if __name__ == '__main__':
    plt.switch_backend('agg')
    gen_summary()


f_data.close()