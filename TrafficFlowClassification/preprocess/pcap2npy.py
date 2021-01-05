'''
@Author: WANG Maonan
@Date: 2021-01-05 16:07:22
@Description: 将 pcap 文件保存为 npy, 用作最后的训练
@LastEditTime: 2021-01-05 18:39:19
'''
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

def save_pcap2npy(pcap_dict, file_name):
    """将 pcap 文件分为训练集和测试集, 并进行保存. 
    => pcap 数据从 二进制 转换为 十进制, 调用 getIntfrom_pcap 函数
    => 将数据保存在 data 中, data = [[pcap, label], [], ..]
    => 将 data 数据保存为 npy 文件

    Args:
        pcap_dict (dict): 见函数 get_train_test 的输出
        file_name (str): 最后保存的文件的名称
    """
    data = []
    label2index = {} # 保存 label 和 index 之间的关系
    index = 0
    for label, pcap_file_list in pcap_dict.items():
        if label not in label2index:
            logger.info('模式, {}, 正在将 {} 的 pcap 保存为 npy'.format(file_name, label))
            label2index[label] = index
            index = index + 1
        for pcap_file in pcap_file_list:
            pcap_content = getIntfrom_pcap(pcap_file) # 返回 pcap 的 十进制 内容
            data.append([pcap_content, label2index[label]]) # 将 pcap 和 label 添加到 data 中去

    np.random.shuffle(data) # 对数据进行打乱
    
    X = np.array([i[0] for i in data]) # data 的特征(data)
    y = np.array([i[1] for i in data]) # data 的标签(label)

    logger.info('数据的大小, {}; 标签的大小, {};'.format(X.shape, y.shape)) # 打印数据大小
    
    np.save('{}-pcap.npy'.format(file_name), X)
    np.save('{}-labels.npy'.format(file_name), y)

    logger.info('将 {} 的数据保存为 npy !'.format(file_name))
    logger.info('==========\n')