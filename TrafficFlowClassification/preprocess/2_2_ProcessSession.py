# Maonan Wang(wangmaonan@bupt.edu.cn)
# 
# 匿名化IP地址和MAC地址
# 将IP地址和MAC地址全变为全0
# ==============================================================================

import os
import shutil # 用于文件的复制
import numpy as np
from scapy.all import *

dataPath = """../1.DataSet/3_trainDataset/""" # 原始数据路径
ResultPath = """../1.DataSet/4_AnonymizationDataset/""" # 目标数据文件夹路径

# 进行匿名化的函数
def customAction(pcap):
    # 匿名化信息
    src_ip = "0.0.0.0"
    src_ipv6 = "0:0:0:0:0:0:0:0"
    src_port = 0
    src_mac = "00:00:00:00:00:00"

    dst_ip = "0.0.0.0"
    dst_ipv6 = "0:0:0:0:0:0:0:0"
    dst_port = 0
    dst_mac = "00:00:00:00:00:00"
    
    if 'Ether' in pcap:
        pcap.src = src_mac
        pcap.dst = dst_mac
    if 'IP' in pcap:
        pcap["IP"].src = src_ip
        pcap["IP"].dst = dst_ip
        pcap["IP"].sport = src_port
        pcap["IP"].dport = dst_port
    if 'IPv6' in pcap:
        pcap["IPv6"].src = src_ipv6
        pcap["IPv6"].dst = dst_ipv6
        pcap["IPv6"].sport = src_port
        pcap["IPv6"].dport = dst_port
    if 'TCP' in pcap:
        pcap['TCP'].sport = src_port
        pcap['TCP'].dport = dst_port
    if 'UDP' in pcap:
        pcap['UDP'].sport = src_port
        pcap['UDP'].dport = dst_port
    if 'ARP' in pcap:
        pcap["ARP"].psrc = src_ip
        pcap["ARP"].pdst = dst_ip
        pcap["ARP"].hwsrc = src_mac
        pcap["ARP"].hwdst = dst_mac



def Anonymize(filePath):
    packets = []
    # 逐个packet进行转换
    for sniffed_packet in sniff(offline=filePath, prn=customAction):
        packets.append(sniffed_packet)
    # 返回新的packet
    return packets

# 循环读取文件夹中所有的pcap文件
for (root, dirs, files) in os.walk(dataPath):
    # 打印文件夹名字
    pcapFile = root.split('/')[-1] # 所在文件夹的名字
    print(pcapFile) # 打印所在文件夹
    for Ufile in files:
        pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
        # -------------
        # 进行匿名化处理
        # -------------
        packets = Anonymize(pcapPath)
        # -------------
        # 生成新的文件名
        # -------------
        fileName = pcapPath.split('/')[-1] # 获取test\\Chat\\skype_chat1b\\skype_chat1b.pcap
        dst = '{}{}'.format(ResultPath,fileName).replace('\\','/')
        os.makedirs(os.path.dirname(dst), exist_ok=True) # 没有就创建文件夹
        # --------------
        # 保存新的packet
        # --------------
        wrpcap(dst, packets)