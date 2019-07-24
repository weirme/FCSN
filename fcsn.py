import torch.nn as nn
from collections import OrderedDict 


class FCSN(nn.Module):
    def __init__(self, n_class=2):
        super(FCSN, self).__init__()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn1_1', nn.BatchNorm1d(1024)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn1_2', nn.BatchNorm1d(1024)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/2

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn2_1', nn.BatchNorm1d(1024)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn2_2', nn.BatchNorm1d(1024)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/4

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_1', nn.BatchNorm1d(1024)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_2', nn.BatchNorm1d(1024)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_3', nn.BatchNorm1d(1024)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/8

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4_1', nn.Conv1d(1024, 2048, 3, padding=1)),
            ('bn4_1', nn.BatchNorm1d(2048)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn4_2', nn.BatchNorm1d(2048)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn4_3', nn.BatchNorm1d(2048)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/16

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv5_1', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_1', nn.BatchNorm1d(2048)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_2', nn.BatchNorm1d(2048)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('conv5_3', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_3', nn.BatchNorm1d(2048)),
            ('relu5_3', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/32

        self.conv6 = nn.Sequential(OrderedDict([
            ('fc6', nn.Conv1d(2048, 4096, 1)),
            ('bn6', nn.BatchNorm1d(4096)),
            ('relu6', nn.ReLU(inplace=True)),
            ('drop6', nn.Dropout())
            ]))
   
        self.conv7 = nn.Sequential(OrderedDict([
            ('fc7', nn.Conv1d(4096, 4096, 1)),
            ('bn7', nn.BatchNorm1d(4096)),
            ('relu7', nn.ReLU(inplace=True)),
            ('drop7', nn.Dropout())
            ]))

        self.conv8 = nn.Sequential(OrderedDict([
            ('fc8', nn.Conv1d(4096, n_class, 1)),
            ('bn8', nn.BatchNorm1d(n_class)),
            ('relu8', nn.ReLU(inplace=True)),
            ]))

        self.conv_pool4 = nn.Conv1d(2048, n_class, 1)
        self.bn_pool4 = nn.BatchNorm1d(n_class)

        self.deconv1 = nn.ConvTranspose1d(n_class, n_class, 4, padding=1, stride=2, bias=False)
        self.deconv2 = nn.ConvTranspose1d(n_class, n_class, 16, stride=16, bias=False)

    def forward(self, x):

        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        pool4 = h

        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.conv8(h)

        h = self.deconv1(h)
        upscore2 = h

        h = self.conv_pool4(pool4)
        h = self.bn_pool4(h)
        score_pool4 = h

        h = upscore2 + score_pool4

        h = self.deconv2(h)

        return h


if __name__ == '__main__':
    import torch
    net = FCSN()
    data = torch.randn((1, 1024, 320))
    print(net(data).shape)
