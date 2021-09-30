# This implementation is based on the DenseNet-BC implementation in torchvision
# Use Nvidia APEX DDP Mode and DeepSeparableConv2d to speed up train and reduce GPU memory consumption

import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn

from apex import amp
from models.swish import Swish
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.DeepSeparableConv import DeepSeparableConv2d


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, drop_rate, BatchNorm, dsconv, act_op, training):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1   = BatchNorm(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2   = BatchNorm(interChannels)

        if dsconv == True :
            self.conv2 = DeepSeparableConv2d(interChannels, growthRate, kernel_size=3, BatchNorm=BatchNorm)
        else :
            self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

        self.act_op    = act_op
        self.drop_rate = drop_rate
        self.training  = training

    def forward(self, x):
        out = self.conv1(self.act_op(self.bn1(x)))
        out = self.conv2(self.act_op(self.bn2(out)))
        out = torch.cat((x, out), 1)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, BatchNorm, dsconv, act_op):
        super(SingleLayer, self).__init__()
        self.bn1    = BatchNorm(nChannels)
        self.act_op = act_op

        if dsconv == True :
            self.conv2 = DeepSeparableConv2d(nChannels, growthRate, kernel_size=3, BatchNorm=BatchNorm)
        else :
            self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.act_op(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, drop_rate, BatchNorm, act_op, training):
        super(Transition, self).__init__()
        self.bn1   = BatchNorm(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

        self.act_op    = act_op
        self.drop_rate = drop_rate
        self.training  = training

    def forward(self, x):
        out = self.conv1(self.act_op(self.bn1(x)))
        out = F.avg_pool2d(out, 2)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class DenseNet(nn.Module):
    def __init__(self, backbone, compression, num_classes, bottleneck, drop_rate, sync_bn, dsconv, training, amp_sel=False, active='ReLU'):
        super(DenseNet, self).__init__()

        if active == 'SWISH' :
            act_op = Swish()
        elif active == 'PReLU' :
            act_op = nn.PReLU()
        else :
            act_op = F.relu

        if sync_bn == True :
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if amp_sel :
            amp.register_float_function(torch, 'sigmoid')
            amp.register_float_function(torch, 'softmax')
            amp.register_float_function(torch, 'relu')

        if backbone == 'net121' :
            growthRate    = 32
            nDenseBlocks1 = 6
            nDenseBlocks2 = 12
            nDenseBlocks3 = 24
            nDenseBlocks4 = 16
        elif backbone == 'net161' :
            growthRate    = 32
            nDenseBlocks1 = 6
            nDenseBlocks2 = 12
            nDenseBlocks3 = 36
            nDenseBlocks4 = 24
        elif backbone == 'net169' :
            growthRate    = 48
            nDenseBlocks1 = 6
            nDenseBlocks2 = 12
            nDenseBlocks3 = 32
            nDenseBlocks4 = 32
        elif backbone == 'net201' :
            growthRate    = 32
            nDenseBlocks1 = 6
            nDenseBlocks2 = 12
            nDenseBlocks3 = 48
            nDenseBlocks4 = 32
        else :
            print('Backbone {} not available.'.format(backbone))
            raise NotImplementedError


        nChannels    = 2*growthRate
        self.conv1   = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)

        self.dense1  = self._make_dense(nChannels, growthRate, nDenseBlocks1, bottleneck, drop_rate, BatchNorm, dsconv, act_op, training)
        nChannels   += nDenseBlocks1 * growthRate
        nOutChannels = int(math.floor(nChannels * compression))
        self.trans1  = Transition(nChannels, nOutChannels, drop_rate, BatchNorm, act_op, training)

        nChannels    = nOutChannels
        self.dense2  = self._make_dense(nChannels, growthRate, nDenseBlocks2, bottleneck, drop_rate, BatchNorm, dsconv, act_op, training)
        nChannels   += nDenseBlocks2 * growthRate
        nOutChannels = int(math.floor(nChannels * compression))
        self.trans2  = Transition(nChannels, nOutChannels, drop_rate, BatchNorm, act_op, training)

        nChannels    = nOutChannels
        self.dense3  = self._make_dense(nChannels, growthRate, nDenseBlocks3, bottleneck, drop_rate, BatchNorm, dsconv, act_op, training)
        nChannels   += nDenseBlocks3 * growthRate
        nOutChannels = int(math.floor(nChannels * compression))
        self.trans3  = Transition(nChannels, nOutChannels, drop_rate, BatchNorm, act_op, training)

        nChannels   = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks4, bottleneck, drop_rate, BatchNorm, dsconv, act_op, training)
        nChannels  += nDenseBlocks4 * growthRate

        self.bn1 = BatchNorm(nChannels)
        self.fc  = nn.Linear(nChannels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, DeepSeparableConv2d) :
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, drop_rate, BatchNorm, dsconv, act_op, training):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, drop_rate, BatchNorm, dsconv, act_op, training))
            else:
                layers.append(SingleLayer(nChannels, growthRate, BatchNorm, dsconv, act_op))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        #out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = F.log_softmax(self.fc(out), dim=1)

        return out


