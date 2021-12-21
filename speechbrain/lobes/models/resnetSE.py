#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetSE(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_filters,
        encoder_out,
        encoder_type="SAP",
        n_mels=40,
        log_input=True,
        **kwargs,
    ):
        super(ResNetSE, self).__init__()

        # print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input
        self.encoder_out = encoder_out

        self.conv1 = nn.Conv1d(
            encoder_out, num_filters[0], kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_filters[0])

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(
            block, num_filters[1], layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, num_filters[2], layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, num_filters[3], layers[3], stride=2
        )

        # self.instancenorm   = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torch.nn.Sequential(
        #         PreEmphasis(),
        #         torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
        #         )

        # outmap_size = int(self.n_mels/8)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(enabled=False):
        #         x = self.torchfb(x)+1e-6
        #         if self.log_input: x = x.log()
        #         x = self.instancenorm(x).unsqueeze(1)

        outputs = []
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        outputs.append(x)

        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)

        # x = x.reshape(x.size()[0],-1,x.size()[-1])

        # w = self.attention(x)

        # if self.encoder_type == "SAP":
        #     x = torch.sum(x * w, dim=2)
        # elif self.encoder_type == "ASP":
        #     mu = torch.sum(x * w, dim=2)
        #     sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        #     x= torch.cat((mu,sg),1)

        # x = x.view(x.size()[0], -1)
        # x = self.fc(x)

        return outputs


class AutoEncoder_SE(nn.Module):
    def __init__(
        self,
        layers=[3, 6, 4, 3],
        num_filters=[256, 128, 64, 32],
        encoder_out=256,
    ):
        super(AutoEncoder_SE, self).__init__()

        self.resnet = ResNetSE_trans(
            block=SEBasicBlock_trans,
            layers=layers,
            num_filters=num_filters,
            encoder_out=encoder_out,
        )

    def forward(self, x):
        return self.resnet(x)


class ResNetSE_trans(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_filters,
        encoder_out,
        encoder_type="SAP",
        n_mels=40,
        log_input=True,
        **kwargs,
    ):
        super(ResNetSE_trans, self).__init__()

        # print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input
        self.encoder_out = encoder_out

        self.conv1 = nn.ConvTranspose1d(
            num_filters[3], encoder_out, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_filters[0])

        self.layer1 = self._make_layer(
            block, num_filters[1], layers[0], stride=2
        )
        self.layer2 = self._make_layer(
            block, num_filters[2], layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, num_filters[3], layers[2], stride=2
        )
        self.layer4 = self._make_layer(block, num_filters[3], layers[3])

        # self.instancenorm   = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torch.nn.Sequential(
        #         PreEmphasis(),
        #         torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
        #         )

        # outmap_size = int(self.n_mels/8)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, Xs):

        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(enabled=False):
        #         x = self.torchfb(x)+1e-6
        #         if self.log_input: x = x.log()
        #         x = self.instancenorm(x).unsqueeze(1)

        x = self.layer1(Xs[-1])
        diff = Xs[3].size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x += Xs[3]

        x = self.layer2(x)
        diff = Xs[2].size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x += Xs[2]

        x = self.layer3(x)
        diff = Xs[1].size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x += Xs[1]

        x = self.layer4(x)
        diff = Xs[0].size()[2] - x.size()[2]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x += Xs[0]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        # x = x.reshape(x.size()[0],-1,x.size()[-1])

        # w = self.attention(x)

        # if self.encoder_type == "SAP":
        #     x = torch.sum(x * w, dim=2)
        # elif self.encoder_type == "ASP":
        #     mu = torch.sum(x * w, dim=2)
        #     sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        #     x= torch.cat((mu,sg),1)

        # x = x.view(x.size()[0], -1)
        # x = self.fc(x)

        return x


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, reduction=8
    ):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBasicBlock_trans(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, reduction=8
    ):
        super(SEBasicBlock_trans, self).__init__()
        self.conv1 = nn.ConvTranspose1d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.ConvTranspose1d(
            planes, planes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, reduction=8
    ):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
