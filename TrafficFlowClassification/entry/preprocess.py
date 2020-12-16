'''
@Author: WANG Maonan
@Date: 2020-12-15 16:53:21
@Description: 对原始流量文件进行预处理
@LastEditTime: 2020-12-16 13:16:44
'''
import os
import yaml

from TrafficFlowClassification.TrafficLog.setLog import logger
from TrafficFlowClassification.preprocess.pcapng2pcap import pcapng_to_pcap
from TrafficFlowClassification.preprocess.pcap2session import pcap_to_session

def setup_config():
    """获取配置信息
    """
    with open(os.path.join(os.path.dirname(__file__), './preprocess.yaml'), encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def preprocess_pipeline():
    """对流量进行预处理, 处理流程为:
    0. 首先将不同的流量新建文件夹, 分成不同的类别
    1. 首先将 pcapng 文件转换为 pcap 文件
    2. 接着将 pcap 文件按照五元组分为不同的 session, 使用 SplitCap 来完成
    """
    cfg = setup_config() # 获取 config 文件
    logger.info(cfg)

    # pcapng_to_pcap(cfg['pcap_path']['raw_pcap_path']) # 将 pcapng 转换为 pcap
    pcap_to_session(cfg['pcap_path']['raw_pcap_path'], cfg['tool_path']['splitcap_path']) # 将 pcap 转换为 session