import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, inChannel, num_classes=17):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, inChannel=3, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], inChannel, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(pretrained=False, inChannel=3, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnet50(pretrained=False, inChannel=3, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



def resnet101(pretrained=False, inChannel=3, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



def resnet152(pretrained=False, inChannel=3, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class LeNet(nn.Module):
    def __init__(self, inChannel=4, nbClass = 8):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 256, 3)
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.fc1 = nn.Linear(256 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, nbClass)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv3(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNet_feature_fusion(nn.Module):
    def __init__(self, inChannel_1=4, inChannel_2=4, nbClass = 8):
        super(LeNet_feature_fusion, self).__init__()
        self.conv1_x1 = nn.Conv2d(inChannel_1, 64, 5)
        self.conv1_x2 = nn.Conv2d(inChannel_2, 64, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 256, 3)
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.fc1 = nn.Linear(256 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, nbClass)
        self.fusionFC = nn.Linear(256,128)

    def forward(self, x1, x2):
        x1 = self.pool(F.relu(self.conv1_x1(x1)))
        x1 = self.pool(F.relu(self.conv3(self.conv2(x1))))
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.fc1(x1))

        x2 = self.pool(F.relu(self.conv1_x2(x2)))
        x2 = self.pool(F.relu(self.conv3(self.conv2(x2))))
        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.fc1(x2))

        fus = torch.cat([x1,x2],dim=1)
        fus = F.relu(self.fusionFC(fus))
        fus = self.fc2(fus)
        return fus

class LeNet_decision_fusion(nn.Module):
    def __init__(self, inChannel_1=4, inChannel_2=4, nbClass = 8):
        super(LeNet_decision_fusion, self).__init__()
        self.conv1_x1 = nn.Conv2d(inChannel_1, 64, 5)
        self.conv1_x2 = nn.Conv2d(inChannel_2, 64, 5)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 256, 3)
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.fc1 = nn.Linear(256 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, nbClass)
        self.LogSoftMax_decistion = nn.LogSoftmax(dim=1)

        self.fusionFC = nn.Linear(nbClass*2,nbClass)

    def forward(self, x1, x2):
        x1 = self.pool(F.relu(self.conv1_x1(x1)))
        x1 = self.pool(F.relu(self.conv3(self.conv2(x1))))
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.fc1(x1))
        x1 = self.LogSoftMax_decistion(self.fc2(x1))

        x2 = self.pool(F.relu(self.conv1_x2(x2)))
        x2 = self.pool(F.relu(self.conv3(self.conv2(x2))))
        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.fc1(x2))
        x2 = self.LogSoftMax_decistion(self.fc2(x2))

        fus = torch.cat([x1,x2],dim=1)
        fus = F.relu(self.fusionFC(fus))
        return fus






class Conv_5(nn.Module):
    def __init__(self, inChannel=4, nbClass = 8):
        super(Conv_5, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, 16, 5)
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.conv5 = nn.Conv2d(128, 256, 5)

        self.fc1 = nn.Linear(256 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, nbClass)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


