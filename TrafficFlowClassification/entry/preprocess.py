'''
@Author: WANG Maonan
@Date: 2020-12-15 16:53:21
@Description: 对原始流量文件进行预处理
@LastEditTime: 2020-12-15 17:51:38
'''
import os
import yaml

from TrafficFlowClassification.TrafficLog.setLog import logger
from TrafficFlowClassification.preprocess.pcapng2pcap import pcapng_to_pcap

def setup_config():
    with open(os.path.join(os.path.dirname(__file__), './preprocess.yaml'), encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def preprocess_pipeline():
    cfg = setup_config() # 获取 config 文件
    logger.info(cfg)
    pcapng_to_pcap(cfg['pcap_path']['raw_pcap_path'])