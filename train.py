# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import json
from tqdm import tqdm, trange

from fcsn import FCSN
from utils import TensorboardWriter

import pdb


class Solver(object):
    """Class that Builds, Trains FCSN model"""

    def __init__(self, config=None, train_loader=None, test_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # model
        self.model = FCSN(self.config.n_class) # .cuda()

        # optimizer
        if self.config.mode == 'train':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9)
            self.model.train()
            self.writer = TensorboardWriter(self.config.log_dir)

    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def sum_loss(pred_score, gt_labels):
        n_batch, n_class, n_frame = pred_score.shape
        sum_loss = 0
        for score_i, labels_i in zip(pred_score, gt_labels):
            freq = torch.empty((n_class, ))
            for i in range(n_class):
                freq[i] = (labels_i == i).sum()
            median_freq = freq.mean()
            w = median_freq / freq

            loss = 0
            # pdb.set_trace()
            for t in range(n_frame):
                exp_sum = torch.exp(score_i[:, t]).sum()
                label_t = labels_i[t].int().item()
                exp_ct = torch.exp(score_i[label_t, t])
                loss += w[label_t] * torch.log(exp_ct / exp_sum)
            loss *= -1/n_frame
            sum_loss += loss
        sum_loss /= n_batch
        return sum_loss

    def train(self):
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            sum_loss_history = []

            for batch_i, (features, labels, _) in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80, leave=False)):

                # [batch_size, 1024, seq_len]
                # => cuda
                features = Variable(features) # .cuda()

                # ---- Train ---- #
                pred_score = self.model(features)

                # pdb.set_trace()

                # pred_label = torch.argmax(pred_score, dim=1).float()
                loss = Variable(self.sum_loss(pred_score, labels), requires_grad=True)
                loss.backward()

                if (batch_i+1) % 5 == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    sum_loss_history.append(loss)

            # pdb.set_trace()
            mean_loss = torch.stack(sum_loss_history).mean().item()
            tqdm.write(f'\nEpoch {epoch_i}')
            tqdm.write(f'sum loss: {mean_loss:.3f}')
            self.writer.update_loss(mean_loss, epoch_i, 'loss')

            if (epoch_i+1) % 5 == 0:
                ckpt_path = self.config.save_dir + f'/epoch-{epoch_i}.pkl'
                tqdm.write(f'Save parameters at {ckpt_path}')
                torch.save(self.model.state_dict(), ckpt_path)
                self.evaluate(epoch_i)
                self.model.train()

    def evaluate(self, epoch_i):
        self.model.eval()
        out_dict = {}

        for features, _, idx in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):

            features = Variable(features) # .cuda()
            pred_score = self.model(features.unsqueeze(0))
            pred_label = torch.argmax(pred_score, dim=1).squeeze(0).type(dtype=torch.int)
            pred_label = np.array(pred_label.cpu().data).tolist()

            out_dict[idx.item()] = pred_label

        score_save_path = self.config.score_dir + f'/epoch-{epoch_i}.json'
        # pdb.set_trace()
        with open(score_save_path, 'w') as f:
            tqdm.write(f'Saving score at {str(score_save_path)}.')
            json.dump(out_dict, f)


if __name__ == '__main__':
    from config import Config
    from data_loader import get_loader
    train_config = Config()
    test_config = Config(mode='test')
    train_loader, test_loader = get_loader(train_config.data_path)
    solver = Solver(train_config, train_loader, test_loader)
    solver.train()