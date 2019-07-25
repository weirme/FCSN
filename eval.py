import torch
import numpy as np

from knapsack import knapsack


def eval_metrics(y_pred, y_true):
    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return [precision, recall, fscore]


def select_keyshots(video_info, pred_score):
    N = video_info['length'][()]
    cps = video_info['change_points'][()]
    weight = video_info['n_frame_per_seg'][()]
    pred_score = np.array(pred_score.cpu().data)
    pred_score = upsample(pred_score, N)
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
    _, selected = knapsack(pred_value, weight, int(0.15 * N))
    selected = selected[::-1]
    key_labels = np.zeros(shape=(N, ))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1
    return pred_score.tolist(), selected, key_labels.tolist()


def upsample(down_arr, N):
    up_arr = np.zeros(N)
    ratio = N // 320
    l = (N - ratio * 320) // 2
    i = 0
    while i < 320:
        up_arr[l:l+ratio] = np.ones(ratio, dtype=int) * down_arr[i]
        l += ratio
        i += 1
    return up_arr
