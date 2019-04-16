# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import os
import json
import csv
from sklearn import metrics


def select_keyshots(num_frames, cps, weight, value):
    from knapsack import knapsack
    _, selected = knapsack(value, weight, int(0.15 * num_frames))
    selected = selected[::-1]
    key_labels = np.zeros(shape=(num_frames,))
    for i in range(len(selected)):
        key_labels[cps[selected[i]][0]:cps[selected[i]][1]] = 1
    return selected, key_labels

def upsample(down_arr, N):
    ratio = N // 320
    l = (N - ratio*320) // 2
    up_arr = np.zeros(N)
    i = 0
    while i < 320:
        up_arr[l:l+ratio] = np.ones(ratio, dtype=int) * down_arr[i]
        l += ratio
        i += 1
    return up_arr

class Evaluator():

    def __init__(self, data_path, json_path):
        self.video_info = {}
        self.json_path = json_path
        with open(self.json_path) as fp:
            json_dict = json.load(fp)
            test_id = json_dict.keys()
        for id in test_id:
            f = h5py.File(data_path, 'r')
            info = f['video_' + id]
            self.video_info[id] = info

    def evaluate(self):
        with open(self.json_path) as fp:
            json_dict = json.load(fp)
        res_dict = {}
        self.pred_info = {}
        for id, info in self.video_info.items():
            N = info['length'][()]
            cps = info['change_points'][()]
            weight = info['n_frame_per_seg'][()]
            true_summary_arr = info['user_summary'][()]
            pred_score = upsample(np.array(json_dict[id]), N)
            pred_value = np.array([pred_score[cps[i][0]:cps[i][1]].mean() for i in range(len(cps))])
            pred_selected, pred_summary = select_keyshots(N, cps, weight, pred_value)
            P_arr, R_arr, F_arr = [], [], []
            for true_summary in true_summary_arr:
                P_arr.append(metrics.precision_score(y_pred=pred_summary, y_true=true_summary))
                R_arr.append(metrics.recall_score(y_pred=pred_summary, y_true=true_summary))
                F_arr.append(metrics.f1_score(y_pred=pred_summary, y_true=true_summary))
            res_dict[id] = np.array([np.mean(P_arr), np.mean(R_arr), np.mean(F_arr)])
            info_dict = {}
            info_dict['pred_score'] = np.array(pred_score)
            info_dict['pred_value'] = np.array(pred_value)
            info_dict['pred_selected'] = np.array(pred_selected)
            info_dict['pred_summary'] = np.array(pred_summary)
            self.pred_info[id] = info_dict
        res = np.zeros(shape=(3,))
        for k in res_dict.keys():
            res += res_dict[k] / 5
        return res_dict, res

    def get_keys(self, id, video_path):
        test_info = self.video_info[str(id)]
        N = test_info['length'][()]
        cps = test_info['change_points'][()]
        pred_info = self.pred_info[str(id)]
        pred_score = pred_info['pred_score'][()]
        pred_selected = pred_info['pred_selected'][()]

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
        video_writer = cv2.VideoWriter(str(id) + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, keyshots.shape[2:0:-1])
        for frame in keyshots:
            video_writer.write(frame)
        video_writer.release()

        keyframes_idx = [pred_score[cps[sel][0]:cps[sel][1]].argmax() + cps[sel][0] for sel in pred_selected]
        keyframes = []
        for idx in keyframes_idx:
            keyframes.append(frames[idx])
        keyframes = np.array(keyframes)

        fig, ax = plt.subplots(ncols=keyframes.shape[0], nrows=1, figsize=(30, 10))
        for i, axi in enumerate(ax.flat):
            axi.imshow(keyframes[i])
            axi.axis('off')
        plt.show()

    def plot_bar(self):
        import seaborn as sns
        sns.set()

        test_id = self.video_info.keys()
        with open(path) as f:
            csv_reader = csv.reader(f, delimiter='\t')
            csv_dict = {}
            idx = 0
            for row in csv_reader:
                score = np.array([int(i) for i in row[2].split(',')])
                if str(idx // 20 + 1) in test_id:
                    if idx % 20 == 0:
                        csv_dict[str(idx // 20 + 1)] = score / 20
                    else:
                        csv_dict[str(idx // 20 + 1)] += score / 20
                idx += 1

        fig, ax = plt.subplots(ncols=1, nrows=len(test_id), figsize=(30, 20))
        for i, axi in enumerate(ax.flat):
            scores = csv_dict[test_id[i]]
            pred_summary = self.pred_info[id]['pred_summary']
            axi.bar(left=list(range(len(scores)), height=scores, color=['lightseagreen' if i == 0 else 'orange' for i in
                                                                        pred_summary], edgecolor=None))
            axi.set_title(id)
        plt.show()

