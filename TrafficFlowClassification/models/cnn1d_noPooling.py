'''
@Author: WANG Maonan
@Date: 2021-01-07 10:57:54
@Description: 1 维卷积模型, 包含了 Pooling 操作
@LastEditTime: 2021-03-14 14:28:08
'''
import torch
import torch.nn as nn

class Cnn1d_noPooling(nn.Module):
    def __init__(self, num_classes=12, image_width=28):
        super(Cnn1d_noPooling, self).__init__()
        # 卷积层+池化层
        self.features = nn.Sequential(
            nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=3, padding=1), # (1,784)->(32,254)
            nn.BatchNorm1d(32), # 加上BN的结果
            nn.ReLU(),

            nn.Conv1d(kernel_size=9, in_channels=32, out_channels=64, stride=3, padding=1), # (32,254)->(64,83)
            nn.BatchNorm1d(64), # 加上BN的结果
            nn.ReLU(),
            
            nn.Conv1d(kernel_size=9, in_channels=64, out_channels=128, stride=3, padding=1), # (64,83)->(128,26)
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(kernel_size=9, in_channels=128, out_channels=128, stride=2, padding=1), # (128,26)->(128,10)
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128*72, out_features=1024), # 784:128*10, 1024:128*15, 4096: 128*72
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    def forward(self, x, statistic):
        x = x.view(x.size(0),1,-1) # 将图片摊平
        x = self.features(x) # 卷积层, 提取特征
        # print(x.shape) # 查看经过卷积之后的输出大小
        x = x.view(x.size(0), -1) # 展开
        x = self.classifier(x) # 分类层, 用来分类
        return x

def cnn1d_noPooling(model_path, pretrained=False, **kwargs):
    """
    CNN 1D no Pooling model architecture 

    Args: 
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Cnn1d_noPooling(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model