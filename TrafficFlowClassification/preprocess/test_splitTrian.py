'''
@Author: WANG Maonan
@Date: 2021-01-05 11:32:31
@Description: 测试 splitTrain.py 文件
@LastEditTime: 2021-01-05 11:55:29
'''
from TrafficFlowClassification.preprocess.splitTrain import get_file_path

def test_get_file_path():
    pcap_dict = get_file_path('D:\Traffic-Classification\data\preprocess_data')
    print(pcap_dict)