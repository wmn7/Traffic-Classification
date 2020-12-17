'''
@Author: WANG Maonan
@Date: 2020-12-15 17:46:21
@Description: 通过该文件, 来运行每一个模块的功能
@LastEditTime: 2020-12-15 17:48:58
'''

import fire
from TrafficFlowClassification.entry.preprocess import preprocess_pipeline

def help():
    """使用的一些简单说明
    """
    data = '''
    => 数据预处理的流程
    python -m TrafficFlowClassification preprocess_pipeline
    
    '''
    return data

if __name__ == "__main__":
    fire.Fire()