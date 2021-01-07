'''
@Author: WANG Maonan
@Date: 2021-01-07 11:06:49
@Description: 用来加载 Pytorch 训练所需要的数据
@LastEditTime: 2021-01-07 12:02:09
'''
import torch
import numpy as np

from TrafficFlowClassification.TrafficLog.setLog import logger


def data_loader(pcap_file, label_file, trimed_file_len, batch_size=256, workers=1, pin_memory=True):
    """读取处理好的 npy 文件, 并返回 pytorch 训练使用的 dataloader 数据

    Args:
        pcap_file ([type]): [description]
        label_file ([type]): [description]
        trimed_file_len ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 256.
        workers (int, optional): [description]. Defaults to 1.
        pin_memory (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # 载入 npy 数据
    pcap_data = np.load(pcap_file) # 获得 pcap 文件
    label_data = np.load(label_file) # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    X_data = torch.from_numpy(pcap_data.reshape(-1, 1, trimed_file_len)).float()
    Y_data = torch.from_numpy(label_data).long()
    logger.info('pcap 文件大小, {}; label 文件大小: {}'.format(X_data.shape, Y_data.shape))
    
    # 将 tensor 数据转换为 Dataset->Dataloader
    res_dataset = torch.utils.data.TensorDataset(X_data, Y_data) # 合并训练数据和目标数据
    res_dataloader = torch.utils.data.DataLoader(
        dataset=res_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=1        # set multi-work num read data
    )

    return res_dataloader