'''
@Author: WANG Maonan
@Date: 2021-01-07 10:57:54
@Description: 1 维卷积模型, 基础模型
@LastEditTime: 2021-01-07 12:10:47
'''
import torch.nn as nn

class Cnn1d(nn.Module):
    def __init__(self, num_classes=12):
        super(cnn1d, self).__init__()
        # 卷积层+池化层
        self.features = nn.Sequential(
            nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12), # (1,784)->(32,784)
            nn.BatchNorm1d(32), # 加上BN的结果
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1), # (32,784)->(32,262)
            
            nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12), # (32,262)->(64,262)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1), # (64,262)->(64*88)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_features=88*64, out_features=1024),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    def forward(self, x):
        x = x.view(x.size(0),1,-1) # 将图片摊平
        x = self.features(x) # 卷积层, 提取特征
        x = x.view(x.size(0), -1) # 展开
        x = self.classifier(x) # 分类层, 用来分类
        return x

def cnn1d(pretrained=False, **kwargs):
    """
    CNN 1D model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Cnn1d(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        # model.load_state_dict(torch.load(os.path.join(models_dir, model_name))) # TODO 增加预训练的模型
        pass
    return model