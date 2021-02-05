'''
@Author: WANG Maonan
@Date: 2021-01-05 16:07:22
@Description: 将 pcap 文件保存为 npy, 用作最后的训练
@LastEditTime: 2021-02-05 21:44:40
'''
import os
import json
import binascii
import numpy as np

from TrafficFlowClassification.TrafficLog.setLog import logger

def getIntfrom_pcap(filename):
    """将 pcap 文件读入, 转换为 十进制 的格式

    Args:
        filename (str): pcap 文件的路径

    Returns:
        np.array: 返回 pcap 对应的数组, 如果是 784 bytes, array 的长度就是 784
    """
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)   
    fh = np.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)]) # 16进制转10进制
    fh = np.uint8(fh)
    return fh

def save_pcap2npy(pcap_dict, file_name, statistic_feature_json, label2index = {}):
    """将 pcap 文件分为训练集和测试集, 并进行保存. 
    => pcap 数据从 二进制 转换为 十进制, 调用 getIntfrom_pcap 函数
    => 从 json 文件读取每一个 pcap 的统计特征
    => 将数据保存在 data 中, data = [[pcap, statistic, label], [], ..]
    => 将 data 数据保存为 npy 文件

    Args:
        pcap_dict (dict): 见函数 get_train_test 的输出
        file_name (str): 最后保存的文件的名称
    """
    with open(statistic_feature_json, "r") as json_read:
        statistic_feature_dict = json.load(json_read) # 获取每个 pcap 的统计特征

    data = []
    index = 0
    for label, pcap_file_list in pcap_dict.items():
        logger.info('模式, {}, 正在将 {} 的 pcap 保存为 npy'.format(file_name, label))
        if label not in label2index:
            label2index[label] = index
            index = index + 1
        for pcap_file in pcap_file_list:
            pcap_file_name = os.path.normpath(pcap_file).split('\\')[-1]
            pcap_content = getIntfrom_pcap(pcap_file) # 返回 pcap 的 十进制 内容, 这里 pcap_file 是 pcap 文件的路径名
            statistic_feature = statistic_feature_dict[pcap_file_name] # 得到统计特征
            data.append([pcap_content, statistic_feature, label2index[label]]) # 将 pcap 和 label 添加到 data 中去

    np.random.shuffle(data) # 对数据进行打乱
    
    pcap_data = np.array([i[0] for i in data]) # raw pcap 减裁
    statistic_data = np.array([i[2] for i in data]) # statistic data
    y = np.array([i[2] for i in data]) # label

    logger.info('数据的大小, {}; 统计特征的大小, {}; 标签的大小, {};'.format(pcap_data.shape, statistic_data.shape, y.shape)) # 打印数据大小
    
    np.save('{}-pcap.npy'.format(file_name), pcap_data)
    np.save('{}-statistic.npy'.format(file_name), statistic_data)
    np.save('{}-labels.npy'.format(file_name), y)

    logger.info('将 {} 的数据保存为 npy !'.format(file_name))
    logger.info('==========\n')

    return label2index