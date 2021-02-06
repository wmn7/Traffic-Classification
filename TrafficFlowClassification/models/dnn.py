'''
@Author: WANG Maonan
@Date: 2021-02-06 19:24:50
@Description: 全连接网络
@LastEditTime: 2021-02-06 19:35:11
'''
import torch
import torch.nn as nn

class DeepNN(nn.Module):
    def __init__(self, num_classes=12):
        super(DeepNN, self).__init__()
        # 提取特征
        self.features = nn.Sequential(
            nn.Linear(in_features=26, out_features=128),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=512),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=24),
            nn.Dropout(0.7),
            nn.Linear(in_features=24, out_features=num_classes)
        )
    def forward(self, pcap, statistic):
        x = self.features(statistic)
        x = self.classifier(x) # 分类层, 用来分类
        return x

def deepnn(model_path, pretrained=False, **kwargs):
    """
    DNN model architecture 

    Args: 
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = DeepNN(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model