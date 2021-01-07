'''
@Author: WANG Maonan
@Date: 2021-01-07 12:50:18
@Description: 
@LastEditTime: 2021-01-07 12:53:59
'''
import os
import yaml
from easydict import EasyDict

def setup_config():
    """获取配置信息
    """
    with open(os.path.join('./TrafficFlowClassification/entry', 'traffic_classification.yaml'), encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg