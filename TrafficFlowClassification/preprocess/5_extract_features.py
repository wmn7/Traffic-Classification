# Maonan Wang(wangmaonan@bupt.edu.cn)

# 对pcap文件提取特征
# ===================================

from scapy.all import *
from FeaturesCalc import FeaturesCalc

import os
import csv
import numpy as np
import pandas as pd

dataPath = """../1.DataSet/3_trainDataset/""" # 原始数据路径

# 初始化计算特征的类
featuresCalc = FeaturesCalc(flow_type="legitimate", min_window_size=1)

# 初始化csv文件, 并加入行名称
with open('trafficFeature.csv', 'w', newline='') as csvfile:
    traffic_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    traffic_writer.writerow(featuresCalc.get_features_name()) # 写入行名
    # 逐个文件扫描并写入特征
    for (root, dirs, files) in os.walk(dataPath):
        # -------------------------
        # 打印文件夹名字(查看运行进度)
        # -------------------------
        pcapFile = root.split('/')[-1] # 所在文件夹的名字
        print(pcapFile) # 打印所在文件夹
        # ------------
        # 查看每一个文件
        # ------------
        for Ufile in files:
            pcapPath = os.path.join(root, Ufile) # 需要提取特征的pcap的完整路径
            # -------------------
            # 特征提取, 并写入特征
            # -------------------
            # 得到label的名称
            labelName = root.split('/')[-1].split('\\')[-2] # 文件所在文件夹的名字, 也就是label
            # 得到pcap的名称
            fileName = '_'.join(Ufile.split('.')[:-1]) # 去掉后缀的文件名
            fileName = "{}_{}".format(root.split('/')[-1].split('\\')[-1], fileName) # 再加上一层文件夹的名称, 得到新的pcap的名称
            # 计算特征
            pkts = rdpcap(pcapPath) # 一整个文件中有多个pcap文件, 读取之后会是一个list
            featuresCalc.set_min_window_size(len(pkts)) # 修改窗口大小
            features = featuresCalc.compute_features(packets_list=pkts, filename=fileName, label=labelName) # 计算特征
            traffic_writer.writerow(features) # 写入特征
            
print('Finish')