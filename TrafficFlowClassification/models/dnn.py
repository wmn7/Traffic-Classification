'''
@Author: WANG Maonan
@Date: 2021-02-06 19:24:50
@Description: 全连接网络, 用来直接使用统计特征来分类
@LastEditTime: 2021-02-06 19:35:11
'''
import numpy as np

import torch
import torch.nn as nn

mean_val = np.array([2.86401660e-03, 0.00000000e+00, 3.08146750e-03, 1.17455448e-02,
       5.75561597e-03, 6.91365004e-04, 6.64955585e-02, 2.41380099e-02,
       9.75861990e-01, 0.00000000e+00, 2.89814456e+02, 6.42617944e+01,
       6.89227965e+00, 2.56964887e+02, 1.36799462e+02, 9.32648320e+01,
       7.83185943e+01, 1.32048335e+02, 2.09555592e+01, 1.70122810e-02,
       6.28544986e+00, 3.27195426e-03, 3.60230735e+01, 9.15340653e+00,
       2.17694894e-06, 7.32748605e+01])

std_val = np.array([3.44500263e-02, 0.00000000e+00, 3.09222563e-02, 8.43027570e-02,
       4.87519125e-02, 1.48120354e-02, 2.49138903e-01, 1.53477827e-01,
       1.53477827e-01, 0.00000000e+00, 8.48196659e+02, 1.94163550e+02,
       1.30259798e+02, 7.62370125e+02, 4.16966374e+02, 1.25455838e+02,
       2.30658312e+01, 8.78612984e+02, 1.84367543e+02, 1.13978421e-01,
       1.19289813e+02, 1.45965914e-01, 8.76535415e+02, 1.78680040e+02,
       4.91812227e-04, 4.40298923e+03]) + 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean_val = torch.from_numpy(mean_val).float().to(device)
std_val = torch.from_numpy(std_val).float().to(device)

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
            nn.Linear(in_features=24, out_features=num_classes)
        )
    def forward(self, pcap, statistic):
        statistic = (statistic - mean_val)/std_val # 首先需要对 statistic 的数据进行归一化
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

if __name__ == "__main__":
    random_input = torch.rand(10, 26) # 模拟统计特征
    deepnn_model = deepnn(model_path='')
    deepnn_model(None, random_input) # 随机一个输入, 查看模型的输出