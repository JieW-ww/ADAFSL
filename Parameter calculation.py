import torch
from thop import profile
import torch.nn as nn
import torch.nn.functional as F

class MSSFE(nn.Module):
    def __init__(self, in_channels, kernel_size):  # 12 (1,1,1) 3
        super(MSSFE, self).__init__()
        self.channels = in_channels
        self.k1 = kernel_size[0]
        self.k2 = kernel_size[1]
        self.k3 = kernel_size[2]
        self.conv1 = nn.Conv3d(in_channels, in_channels, (self.k1, self.k2, self.k3), stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False, dilation=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, (self.k1, self.k2, self.k3), stride=(1, 1, 1),
                               padding=(2, 2, 2), bias=False, dilation=2)
        self.Avgpool = nn.AvgPool3d((2, 1, 1), stride=(2, 1, 1))
        self.bn = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        xs = []
        x1 = self.conv1(x)
        x2 = self.bn(x1)
        x3 = self.Avgpool(x2)
        xs.append(x3)

        x4 = self.conv1(x1)
        x5 = self.bn(x4)
        x6 = self.Avgpool(x5)
        xs.append(x6)

        x7 = self.conv2(x4)
        x8 = self.bn(x7)
        x9 = self.Avgpool(x8)
        xs.append(x9)

        out = torch.cat(xs, dim=1)
        return out

class MSSFE1(nn.Module):
    def __init__(self, in_channels, kernel_size):  # 12 (1,1,1) 3
        super(MSSFE1, self).__init__()
        self.channels = in_channels
        self.k1 = kernel_size[0]
        self.k2 = kernel_size[1]
        self.k3 = kernel_size[2]
        self.conv1 = nn.Conv3d(in_channels, in_channels, (self.k1, self.k2, self.k3), stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False, dilation=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, (self.k1, self.k2, self.k3), stride=(1, 1, 1),
                               padding=(2, 2, 2), bias=False, dilation=2)
        self.Avgpool = nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
        self.bn = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        xs = []
        x1 = self.conv1(x)
        x2 = self.bn(x1)
        x3 = self.Avgpool(x2)
        xs.append(x3)

        x4 = self.conv1(x1)
        x5 = self.bn(x4)
        x6 = self.Avgpool(x5)
        xs.append(x6)

        x7 = self.conv2(x4)
        x8 = self.bn(x7)
        x9 = self.Avgpool(x8)
        xs.append(x9)

        out = torch.cat(xs, dim=1)
        return out

class MInet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MInet, self).__init__()
        self.channels = in_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, (1, 1, 1), stride=(1, 1, 1))
        self.conv2 = MSSFE(4, (3, 3, 3))
        self.conv3 = MSSFE1(12, (3, 3, 3))
        self.conv4 = nn.Conv3d(36, 16, (3, 3, 3), stride=(1, 1, 1))
        self.Avgpool = nn.AvgPool3d((3, 1, 1), stride=(2, 1, 1))
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=1)
        self.bn1 = nn.BatchNorm3d(4)
        self.bn2 = nn.BatchNorm3d(16)
        # self.Attention = ChannelSpatialSELayer3D(4)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = F.relu(self.bn1(x1))
        x3 = self.Avgpool(x2)
        x4 = self.conv2(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = F.relu(self.bn2(x6))
        out = self.Avgpool1(x7)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        return out


model = MInet(1,4).cuda()
input = torch.randn(1, 103, 9, 9).cuda()
macs, params = profile(model, inputs=(input, ))

print('FLOPs = ' + str(macs/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
