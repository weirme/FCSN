import torch
import numpy as np

from knapsack import knapsack


def eval_metrics(y_pred: torch.Tensor, y_true: torch.Tensor):
    overlap = (y_pred * y_true).sum().item()
    precision = overlap / (y_pred.sum().item() + 1e-8)
    recall = overlap / (y_true.sum().item() + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return [precision, recall, fscore]


def select_keyshots(num_frames, cps, weight, value):
    _, selected = knapsack(value, weight, int(0.15 * num_frames))
    selected = selected[::-1]
    key_labels = np.zeros(shape=(num_frames,))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1
    return selected, key_labels


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


def eval_single(video_info, pred_score):
    """
    Evaluate F-score of given video and pred_score.

    Args:
        video_info: hdf5 dataset instance, containing necessary infomation for evaluation.
        pred_score: output of FCSN model.
    Returns:
        evaluation result (precision, recall, f-score).
    """
    N = video_info['length'][()]
    cps = video_info['change_points'][()]
    weight = video_info['n_frame_per_seg'][()]
    true_summary_arr = video_info['user_summary'][()]
    pred_score = np.array(pred_score.cpu().data)
    pred_score = upsample(pred_score, N)
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
    pred_selected, pred_summary = select_keyshots(N, cps, weight, pred_value)
    eval_arr = [eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr]
    eval_res = np.mean(eval_arr, axis=0)
    return eval_res.tolist()