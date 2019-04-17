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
        self.model = FCSN(self.config.n_class)
        self.model.eval()

        # optimizer
        if self.config.mode == 'train':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9)
            self.model.train()
            self.writer = TensorboardWriter(self.config.log_dir)

        # weight
        self.tvsum_weight = torch.tensor([0.55989996, 4.67362574])

        if self.config.gpu:
            self.model = self.model.cuda()
            self.tvsum_weight = self.tvsum_weight.cuda()

    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def sum_loss(pred_score, gt_labels, weight=None):
        n_batch, n_class, n_frame = pred_score.shape

        log_p = torch.log_softmax(pred_score, dim=1).reshape(-1, n_class)
        gt_labels = gt_labels.reshape(-1)
        criterion = torch.nn.NLLLoss(weight)
        loss = criterion(log_p, gt_labels)

        return loss

    def train(self):
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            sum_loss_history = []

            for batch_i, (feature, label, _) in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80,
                                                                leave=False)):

                # [batch_size, 1024, seq_len]
                feature.requires_grad_()
                # => cuda
                if self.config.gpu:
                    feature = feature.cuda()
                    label = label.cuda()

                # ---- Train ---- #
                pred_score = self.model(feature)

                # pdb.set_trace()

                # pred_label = torch.argmax(pred_score, dim=1)
                # loss = torch.nn.MSELoss(pred_label, label)
                loss = self.sum_loss(pred_score, label, self.tvsum_weight)
                loss.backward()

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

        for feature, _, idx in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):

            if self.config.gpu:
                feature = feature.cuda()
            pred_score = self.model(feature.unsqueeze(0))
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
    train_loader, test_loader = get_loader(train_config.data_path, batch_size=train_config.batch_size)
    solver = Solver(train_config, train_loader, test_loader)
    solver.train()