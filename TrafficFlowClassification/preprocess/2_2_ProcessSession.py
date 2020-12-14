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
def customAction(packet):
    # 匿名化MAC地址
    try:
        packet[Ether].dst="00:00:00:00:00:00"
        packet[Ether].src="00:00:00:00:00:00"
        # 匿名化IP地址(需要判断是IPv4还是IPv6)
        if packet[Ether].type==34525: # 0x86dd
            packet[IPv6].src="0:0:0:0:0:0:0:0"
            packet[IPv6].dst="0:0:0:0:0:0:0:0"
        else:
            packet[IP].src="0.0.0.0"
            packet[IP].dst="0.0.0.0"
    except: # 使用VPN无MAC地址
        if packet[0].version==6: # VPN没有链路层, 第一层是IP
            packet[IPv6].src="0:0:0:0:0:0:0:0"
            packet[IPv6].dst="0:0:0:0:0:0:0:0"
        else:
            packet[IP].src="0.0.0.0"
            packet[IP].dst="0.0.0.0"



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