'''
@Author: WANG Maonan
@Date: 2021-02-03 14:35:39
@Description: 将数据转换为 tensor 数据
@LastEditTime: 2021-02-03 14:38:59
'''
import torch
import numpy as np

from TrafficFlowClassification.TrafficLog.setLog import logger


def get_tensor_data(pcap_file, label_file, trimed_file_len):
    """读取处理好的 npy 文件, 并返回 pytorch 训练使用的 dataloader 数据

    Args:
        pcap_file (str): pcap 文件转换得到的 npy 文件的路径
        label_file (str): 上面的 pcap 文件对应的 label 文件的 npy 文件的路径
        trimed_file_len (int): pcap 被裁剪成的长度
    """
    # 载入 npy 数据
    pcap_data = np.load(pcap_file) # 获得 pcap 文件
    label_data = np.load(label_file) # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    X_data = torch.from_numpy(pcap_data.reshape(-1, 1, trimed_file_len)).float()
    Y_data = torch.from_numpy(label_data).long()
    logger.info('导入 Tensor 数据, pcap 文件大小, {}; label 文件大小: {}'.format(X_data.shape, Y_data.shape))

    return X_data, Y_data