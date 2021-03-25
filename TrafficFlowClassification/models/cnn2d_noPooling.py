'''
@Author: WANG Maonan
@Date: 2021-02-02 18:02:52
@Description: 二维卷积模型, 包含了 Pooling 操作
@LastEditTime: 2021-03-14 17:38:26
'''
import torch
import torch.nn as nn

class Cnn2d_noPooling(nn.Module):
    def __init__(self, num_classes=12, image_width=28):
        super(Cnn2d_noPooling, self).__init__()
        self.image_width = image_width # 图片长和宽
        # 卷积层+池化层
        self.features = nn.Sequential(
            nn.Conv2d(kernel_size=(5, 5), in_channels=1, out_channels=32, stride=2, padding=1), # (1,28,28)->(32,13,13)
            nn.BatchNorm2d(32), # 加上BN的结果
            nn.ReLU(),
            
            nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=64, stride=2, padding=1), # (32,13,13)->(64,7,7)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=128, stride=2, padding=1), # (64,7,7)->(128,4,4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128*8*8, out_features=1024), # 784:128*4*4, 1024:128*4*4, 4096: 128*8*8
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        
    def forward(self, x, statistic):
        """模型前向传播的函数

        Args:
            x: 裁剪过后的 pcap 数据
            statistic: pcap 的统计信息 (26 个统计特征)
        """
        x = x.view(x.size(0), 1, self.image_width, self.image_width) # 这里需要转换为 (batch, channel, width, height)
        x = self.features(x) # 卷积层, 提取特征
        # print(x.shape)
        x = x.view(x.size(0), -1) # 展开
        x = self.classifier(x) # 分类层, 用来分类
        return x

def cnn2d_noPooling(model_path, pretrained=False, **kwargs):
    """
    CNN 2D no Pooling model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Cnn2d_noPooling(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        pass
    return model