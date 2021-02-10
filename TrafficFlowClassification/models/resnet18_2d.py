'''
@Author: WANG Maonan
@Date: 2021-02-10 11:01:50
@Description: ResNet 的实现
@LastEditTime: 2021-02-10 11:21:50
'''
import math
import torch
import torch.nn as nn
import torchvision.models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """一个基础的 Residual Block
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample # 是否改变输入图片的通道个数
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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=12, image_width=28):
        """构建一个 ResNet 网络
        
        Args:
            block: ResNet 中基础的 Block, 例如可以是上面定义的 BasicBlock
            layers (list): 每一块中 Block 的个数

        """
        super(ResNet, self).__init__()
        self.image_width = image_width # 处理图像的大小

        self.inplanes = 32 # 这个是经过最上面的「卷积」之后的 pannel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        self.adaptiveAvgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules(): # 对模型的参数进行初始化
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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, statistic):
        x = x.view(x.size(0), 1, self.image_width, self.image_width)
        x = self.conv1(x) # (1,28,28)->(32,28,28)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x) # (32,28,28)->(32,28,28)
        x = self.layer2(x) # (32,28,28)->(64,14,14)
        x = self.layer3(x) # (64,14,14)->(128,7,7)
        x = self.layer4(x) # (128,7,7)->(256,4,4)
        
        x = self.adaptiveAvgpool(x) # (256,4,4)->(256,1,1)
        x = x.view(x.size(0), -1) # (batch_size,256,1,1)->(batch_size,256)
        x = self.fc(x)
        
        return x

def resnet182D(model_path, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == "__main__":
    resnet_model = resnet182D(model_path = '') 
    print(resnet_model) # 打印 ResNet 的结构