'''
@Author: WANG Maonan
@Date: 2021-01-05 11:08:45
@Description: 将原始的数据划分训练集与测试集, 返回训练集和测试集的路径地址
@LastEditTime: 2021-01-05 12:37:55
'''
import os
from TrafficFlowClassification.TrafficLog.setLog import logger

def get_file_path(folder_path):
    """获得 folder_path 下 pcap 文件的路径, 以 dict 的形式返回. 
    返回的包含每个大类(Chat, Email), 下每个小类(AIMchat1, aim_chat_3a), 中 pcap 的文件路径.
    返回数据类型如下所示:
    {
        'Chat': {
            'AIMchat1': ['D:\\Traffic-Classification\\data\\preprocess_data\\Chat\\AIMchat1\\AIMchat1.pcap.TCP_131-202-240-87_13393_178-237-24-202_443.pcap', ...]
            'aim_chat_3a': [...],
            ...
        },
        'Email': {
            'email1a': [],
            ...
        },
        ...
    }

    Args:
        folder_path (str): 包含 pcap 文件的根目录名称
    """
    pcap_dict = {}
    for (root, _, files) in os.walk(folder_path):
        if len(files) > 0:
            logger.info('正在记录 {} 下的 pcap 文件'.format(root))
            folder_name_list = os.path.normpath(root).split(os.sep) # 将 'D:\Traffic-Classification\data\preprocess_data' 返回为列表 ['D:', 'Traffic-Classification', 'data', 'preprocess_data']
            top_category, second_category = folder_name_list[-2], folder_name_list[-1]
            if top_category not in pcap_dict:
                pcap_dict[top_category] = {}
            if second_category not in pcap_dict[top_category]:
                pcap_dict[top_category][second_category] = []
            for Ufile in files:
                pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
                pcap_dict[top_category][second_category].append(pcapPath)
    logger.info('将所有的 pcap 文件整理为 dict !')
    logger.info('==========\n')
    return pcap_dict