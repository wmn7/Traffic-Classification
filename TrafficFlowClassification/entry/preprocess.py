'''
@Author: WANG Maonan
@Date: 2020-12-15 16:53:21
@Description: 对原始流量文件进行预处理
@LastEditTime: 2021-01-05 10:53:32
'''
import os
import yaml
import shutil
from easydict import EasyDict

from TrafficFlowClassification.TrafficLog.setLog import logger

from TrafficFlowClassification.preprocess.pcapng2pcap import pcapng_to_pcap
from TrafficFlowClassification.preprocess.pcapTransfer import pcap_transfer
from TrafficFlowClassification.preprocess.pcap2session import pcap_to_session
from TrafficFlowClassification.preprocess.anonymizeSession import anonymize

def setup_config():
    """获取配置信息
    """
    with open(os.path.join(os.path.dirname(__file__), './preprocess.yaml'), encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg

def transfer_pcap(before_path, after_path):
    """将 before_path 中的所以文件转移到 after_path 中去

    Args:
        before_path (str): 转移之前的文件路径
        after_path (str): 转移之后的文件路径
    """
    ignore_list = ['youtubeHTML5_1.pcap', 'torFacebook.pcap', 'torGoogle.pcap', 'torTwitter.pcap'] # 不需要转移的 pcap 文件
    os.makedirs(after_path, exist_ok=True) # 新建目标目录
    for file in os.listdir(before_path):
        src_path = os.path.join(before_path, file)
        dst_path = os.path.join(after_path, file)
        if file in ignore_list:
            logger.info('忽略文件 {}'.format(src_path))
        else:
            logger.info('开始转移文件, {} --> {}'.format(src_path, dst_path))
            shutil.copy(src_path, dst_path)
    logger.info('文件全部转移完成')
    logger.info('=============\n')

def preprocess_pipeline():
    """对流量进行预处理, 处理流程为:
    0. 将所有流量文件转移到新的文件夹, 这时候没有细分, 就是所有文件进行转移
    1. 接着将 pcapng 文件转换为 pcap 文件
    2. 接着将不同的流量新建文件夹, 分成不同的类别, pcap transfer
    3. 接着将 pcap 文件按照五元组分为不同的 session, 使用 SplitCap 来完成 (这一步可以选择, 提取 all 或是 L7)
    4. 对 session 进行处理, 匿名化处理, ip, mac, port
    5. 对于每一类的文件, 划分训练集和测试集, 获得每一类的
    6. 将文件切割为指定大小, 最终保存为 npy 的格式
    """
    cfg = setup_config() # 获取 config 文件
    logger.info(cfg)

    # transfer_pcap(cfg.pcap_path.raw_pcap_path, cfg.pcap_path.new_pcap_path) # 转移文件
    # pcapng_to_pcap(cfg.pcap_path.new_pcap_path) # 将 pcapng 转换为 pcap
    # pcap_transfer(cfg.pcap_path.new_pcap_path, cfg.pcap_path.new_pcap_path) # 将文件放在指定文件夹中
    # pcap_to_session(cfg.pcap_path.new_pcap_path, cfg.tool_path.splitcap_path) # 将 pcap 转换为 session
    # anonymize(cfg.pcap_path.new_pcap_path) # 对指定文件夹内的所有 pcap 进行匿名化处理


if __name__ == "__main__":
    preprocess_pipeline()