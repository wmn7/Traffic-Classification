'''
@Author: WANG Maonan
@Date: 2020-12-15 17:46:21
@Description: 通过该文件, 来运行每一个模块的功能
@LastEditTime: 2021-01-07 20:42:26
'''

import fire
from TrafficFlowClassification.entry.preprocess import preprocess_pipeline
from TrafficFlowClassification.entry.train import train_pipeline
from TrafficFlowClassification.entry.CENTIME_Train import CENTIME_train_pipeline, alpha_experiment_CENTIME

def help():
    """使用的一些简单说明
    """
    data = '''
    => 数据预处理的流程
    python -m TrafficFlowClassification preprocess_pipeline
    => 数据训练的流程 (基础模型训练)
    python -m TrafficFlowClassification train_pipeline
    => 混合模型训练流程
    python -m TrafficFlowClassification CENTIME_train_pipeline
    '''
    return data

if __name__ == "__main__":
    fire.Fire()