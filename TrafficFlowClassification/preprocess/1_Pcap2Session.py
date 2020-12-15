'''
@Author: WANG Maonan
@Date: 2020-12-14 18:50:39
@Description: 将pcap文件转为session.
@LastEditTime: 2020-12-15 17:10:07
'''

# 对./data下每一个文件进行转换
# 如将./data/Chat/aim_chat_3a.pcap文件进行转换
# 最终文件保存在文件夹, /1.DataSet/pcap2session/, 中
# ==============================================================================

import os
from TrafficFlowClassification.TrafficLog.setLog import logger

rawPath = """G:/台大机器学习课程学习/28_加密流量检测/Demo/DeepTraffic/2.encrypted_traffic_classification/1.DataSet/data/"""
sessionPath = """G:/台大机器学习课程学习/28_加密流量检测/Demo/DeepTraffic/2.encrypted_traffic_classification/1.DataSet/pcap2session/"""
toolPath = "G:/台大机器学习课程学习/28_加密流量检测/Demo/DeepTraffic/2.encrypted_traffic_classification/2.PreprocessedTools/0_Tool/SplitCap_2-1/SplitCap"

def pcap_to_session()
for (root, dirs, files) in os.walk(rawPath):
    for Ufile in files:
        pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
        pcapFile = root.split('/')[-1] # 所在文件夹的名字
        pcapName = Ufile.split('.')[0] # pcap文件的名字
        pcap2session = "{} -p 100000 -b 100000 -r {} -o {}{}/{}/".format(toolPath, pcapPath, sessionPath, pcapFile, pcapName) # 提取All Layers
        # pcap2session = "{} -p 100000 -b 100000 -r {} -o {}{}/{}/ -y L7".format(toolPath, pcapPath, sessionPath, pcapFile, pcapName) # 提取L7
        print('正在处理文件{}'.format(Ufile))
        os.system(pcap2session)
