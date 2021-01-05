'''
@Author: WANG Maonan
@Date: 2021-01-05 11:32:31
@Description: 测试 splitTrain.py 文件
@LastEditTime: 2021-01-05 12:53:17
'''
from TrafficFlowClassification.preprocess.splitTrain import get_file_path, get_train_test

def test_get_file_path():
    pcap_dict = get_file_path('D:\Traffic-Classification\data\preprocess_data')
    print(pcap_dict)

def test_get_train_test():
    train_dict, test_dict = get_train_test('D:\Traffic-Classification\data\preprocess_data', 0.9)
    print(train_dict, test_dict)