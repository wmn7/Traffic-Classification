'''
@Author: WANG Maonan
@Date: 2021-02-02 18:02:52
@Description: 二维卷积模型, 用来进行流量分类
@LastEditTime: 2021-02-03 16:11:50
'''
import os
import torch
import torch.nn as nn

class Cnn2d(nn.Module):
    def __init__(self, num_classes=12, image_width=28):
        super(Cnn2d, self).__init__()
        self.image_width = image_width # 图片长和宽
        # 卷积层+池化层
        self.features = nn.Sequential(
            nn.Conv2d(kernel_size=(5, 5), in_channels=1, out_channels=32, stride=1, padding=2), # (1,28,28)->(32,28,28)
            nn.BatchNorm2d(32), # 加上BN的结果
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1), # (32,28,28)->(32,15,15)
            
            nn.Conv2d(kernel_size=(5, 5), in_channels=32, out_channels=64, stride=1, padding=2), # (32,15,15)->(64,15,15)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=1), # (64,15,15)->(64,8,8)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*8*8, out_features=1024),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), 1, self.image_width, self.image_width) # 这里需要转换为 (batch, channel, width, height)
        x = self.features(x) # 卷积层, 提取特征
        x = x.view(x.size(0), -1) # 展开
        x = self.classifier(x) # 分类层, 用来分类
        return x

def cnn2d(model_path, pretrained=False, **kwargs):
    """
    CNN 2D model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Cnn2d(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        pass
    return model