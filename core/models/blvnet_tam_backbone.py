
import itertools
from collections import OrderedDict

import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ['bLVNet_TAM_BACKBONE', 'blvnet_tam_backbone']

model_urls = {
    'blresnet50': 'pretrained/ImageNet-bLResNet-50-a2-b4.pth.tar',
    'blresnet101': 'pretrained/ImageNet-bLResNet-101-a2-b4.pth.tar'
}


class TAM(nn.Module):

    def __init__(self, duration, channels, blending_frames=3):
        super().__init__()
        self.blending_frames = blending_frames

        if blending_frames == 3:
            self.prev = nn.Conv2d(channels, channels, kernel_size=1,
                                     padding=0, groups=channels, bias=False)
            self.next = nn.Conv2d(channels, channels, kernel_size=1,
                                     padding=0, groups=channels, bias=False)
            self.curr = nn.Conv2d(channels, channels, kernel_size=1,
                                     padding=0, groups=channels, bias=False)
        else:
            self.blending_layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1,
                                                            padding=0, groups=channels, bias=False)
                                                  for i in range(blending_frames)])
        self.relu = nn.ReLU(inplace=True)
        self.duration = duration

    def forward(self, x):
        if self.blending_frames == 3:

            prev_x = self.prev(x)
            curr_x = self.curr(x)
            next_x = self.next(x)
            prev_x = prev_x.view((-1, self.duration) + prev_x.size()[1:])
            curr_x = curr_x.view((-1, self.duration) + curr_x.size()[1:])
            next_x = next_x.view((-1, self.duration) + next_x.size()[1:])

            prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
            next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

            out = torch.stack([prev_x, curr_x, next_x], dim=0)
        else:
            # multiple blending
            xs = [se(x) for se in self.blending_layers]
            xs = [x.view((-1, self.duration) + x.size()[1:]) for x in xs]

            shifted_xs = []
            for i in range(self.blending_frames):
                shift = i - (self.blending_frames // 2)
                x_temp = xs[i]
                n, t, c, h, w = x_temp.shape
                start_index = 0 if shift < 0 else shift
                end_index = t if shift < 0 else t + shift
                padding = None
                if shift < 0:
                    padding = (0, 0, 0, 0, 0, 0, abs(shift), 0)
                elif shift > 0:
                    padding = (0, 0, 0, 0, 0, 0, 0, shift)
                shifted_xs.append(F.pad(x_temp, padding)[:, start_index:end_index, ...]
                                  if padding is not None else x_temp)

            out = torch.stack(shifted_xs, dim=0)
        out = torch.sum(out, dim=0)
        out = self.relu(out)
        # [N, T, C, N, H]
        out = out.view((-1, ) + out.size()[2:])
        return out


def get_frame_list(init_list, num_frames, batch_size):
    if batch_size == 0:
        return []

    flist = list()
    for i in range(batch_size):
        flist.append([k + i * num_frames for k in init_list])
    return list(itertools.chain(*flist))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True,
                 with_tam=False, num_frames=-1, blending_frames=-1):

        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

        self.tam = TAM(num_frames, inplanes, blending_frames) \
            if with_tam else None

    def forward(self, x):
        residual = x

        if self.tam is not None:
            x = self.tam(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out


class bLModule(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, alpha, beta, stride,
                 num_frames, blending_frames=3):
        super(bLModule, self).__init__()
        self.num_frames = num_frames
        self.blending_frames = blending_frames

        self.relu = nn.ReLU(inplace=True)
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1, 2, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha, max(1, blocks // beta - 1))
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.fusion = self._make_layer(block, out_channels, out_channels, 1, stride=stride)
        self.tam = TAM(self.num_frames, in_channels, blending_frames=self.blending_frames)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, stride, downsample))
        else:
            layers.append(block(inplanes, planes, stride, downsample))
            for i in range(1, blocks):
                layers.append(block(planes, planes,
                                    last_relu=last_relu if i == blocks - 1 else True))

        return nn.Sequential(*layers)

    def forward(self, x, big_frame_num, big_list, little_frame_num, little_list):
        n = x.size()[0]
        if self.tam is not None:
            x = self.tam(x)

        big = self.big(x[big_list, ::])
        little = self.little(x[little_list, ::])
        little = self.little_e(little)
        big = torch.nn.functional.interpolate(big, little.shape[2:])

        # [0 1] sum up current and next frames
        bn = big_frame_num
        ln = little_frame_num

        big = big.view((-1, bn) + big.size()[1:])
        little = little.view((-1, ln) + little.size()[1:])
        big += little  # left frame

        # only do the big branch
        big = big.view((-1,) + big.size()[2:])
        big = self.relu(big)
        big = self.fusion(big)

        # distribute big to both
        x = torch.zeros((n,) + big.size()[1:], device=big.device, dtype=big.dtype)
        x[range(0, n, 2), ::] = big
        x[range(1, n, 2), ::] = big

        return x


class bLVNet_TAM_BACKBONE(nn.Module):

    def __init__(self, block, layers, alpha, beta, num_frames, num_classes=1000,
                 blending_frames=3, input_channels=3):

        self.num_frames = num_frames
        self.blending_frames = blending_frames

        self.bL_ratio = 2
        self.big_list = range(self.bL_ratio // 2, num_frames, self.bL_ratio)
        self.little_list = list(set(range(0, num_frames)) - set(self.big_list))

        num_channels = [64, 128, 256, 512]
        self.inplanes = 64

        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2,
                                 padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])

        self.tam = TAM(self.num_frames, num_channels[0], blending_frames=self.blending_frames)

        self.layer1 = bLModule(block, num_channels[0], num_channels[0] * block.expansion,
                               layers[0], alpha, beta, stride=2, num_frames=self.num_frames,
                               blending_frames=blending_frames)
        self.layer2 = bLModule(block, num_channels[0] * block.expansion,
                               num_channels[1] * block.expansion, layers[1], alpha, beta, stride=2,
                               num_frames=self.num_frames,
                               blending_frames=blending_frames)
        self.layer3 = bLModule(block, num_channels[1] * block.expansion,
                               num_channels[2] * block.expansion, layers[2], alpha, beta, stride=1,
                               num_frames=self.num_frames,
                               blending_frames=blending_frames)
        # only half frames are used.
        self.layer4 = self._make_layer(
            block, num_channels[2] * block.expansion, num_channels[3] * block.expansion, layers[3],
            num_frames=self.num_frames // 2, stride=2)

        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, num_frames, stride=1, with_tam=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, with_tam=with_tam,
                            num_frames=num_frames, blending_frames=self.blending_frames))
        for i in range(1, blocks):
            layers.append(block(planes, planes, with_tam=with_tam,
                                num_frames=num_frames, blending_frames=self.blending_frames))

        return nn.Sequential(*layers)

    def _forward_bL_layer0(self, x, big_frame_num, big_list, little_frame_num, little_list):
        n = x.size()[0]
        if self.tam is not None:
            x = self.tam(x)

        bx = self.b_conv0(x[big_list, ::])
        bx = self.bn_b0(bx)

        lx = self.l_conv0(x[little_list, ::])
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)

        bn = big_frame_num
        ln = little_frame_num
        bx = bx.view((-1, bn) + bx.size()[1:])
        lx = lx.view((-1, ln) + lx.size()[1:])
        bx += lx   # left frame

        bx = bx.view((-1,) + bx.size()[2:])

        bx = self.relu(bx)
        bx = self.bl_init(bx)
        bx = self.bn_bl_init(bx)
        bx = self.relu(bx)

        x = torch.zeros((n,) + bx.size()[1:], device=bx.device, dtype=bx.dtype)
        x[range(0, n, 2), ::] = bx
        x[range(1, n, 2), ::] = bx

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        n = x.size()[0]
        batch_size = n // self.num_frames
        big_list = get_frame_list(self.big_list, self.num_frames, batch_size)
        little_list = get_frame_list(self.little_list, self.num_frames, batch_size)

        x = self._forward_bL_layer0(x, len(self.big_list), big_list, len(self.little_list), little_list)
        x = self.layer1(x, len(self.big_list), big_list, len(self.little_list), little_list)
        x = self.layer2(x, len(self.big_list), big_list, len(self.little_list), little_list)
        x = self.layer3(x, len(self.big_list), big_list, len(self.little_list), little_list)

        x = self.layer4(x[big_list, ::])

        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def blvnet_tam_backbone(depth, alpha, beta, num_frames, blending_frames=3, input_channels=3,
                        imagenet_blnet_pretrained=True):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]

    model = bLVNet_TAM_BACKBONE(Bottleneck, layers, alpha, beta, num_frames,
                                blending_frames=blending_frames, input_channels=input_channels)

    if imagenet_blnet_pretrained:
        checkpoint = torch.load(model_urls['blresnet{}'.format(depth)], map_location='cpu')
        # fixed parameter names in order to load the weights correctly
        state_d = OrderedDict()
        if input_channels != 3:  # flow
            print("loading weights from ImageNet-pretrained blnet, blresnet{}".format(depth),
                  flush=True)
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('module.', '')
                if "conv1.weight" in key:
                    o_c, in_c, k_h, k_w = value.shape
                else:
                    o_c, in_c, k_h, k_w = 0, 0, 0, 0
                if k_h == 7 and k_w == 7:
                    # average the weights and expand to all channels
                    new_shape = (o_c, input_channels, k_h, k_w)
                    new_value = value.mean(dim=1, keepdim=True).expand(new_shape).contiguous()
                else:
                    new_value = value
                state_d[new_key] = new_value
        model.load_state_dict(state_d, strict=False)

    return model
