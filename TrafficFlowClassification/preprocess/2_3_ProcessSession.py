# Maonan Wang(wangmaonan@bupt.edu.cn)
# 
# 对原始文件进行裁剪(我们需要把需要剪裁的文件放在新的文件夹中)=>需要改进的地方(为了方便之后做实验)
# 裁剪大小为784bytes, 大于这个值进行裁剪, 小于这个值进行填补
# 这一部分思路参考https://github.com/echowei/DeepTraffic的代码
# ==============================================================================

import os
import shutil # 用于文件的复制
import numpy as np

# RawTrainPath = """../1.DataSet/3_trainDataset/train""" # 训练集文件夹路径
# RawTestPath = """../1.DataSet/3_trainDataset/test""" # 测试集文件夹路径

TrimeTrainPath = """../1.DataSet/5_trimDataset/trime_784/train/""" # 修改后的训练集
TrimeTestPath = """../1.DataSet/5_trimDataset/trime_784/test/""" # 修改后的测试集

TRIMED_FILE_LEN = 784 # 裁剪大小

# 循环读取文件夹中所有的pcapng文件
for dataPath in [TrimeTrainPath, TrimeTestPath]: # 依次处理训练集和测试集
    for (root, dirs, files) in os.walk(dataPath):
        fileNum = 0 # 统计文件夹内文件个数
        for Ufile in files:
            pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
            pcapSize = os.path.getsize(pcapPath) # 获得文件的大小, bytes
            fileLength = TRIMED_FILE_LEN - pcapSize # 文件大小与规定大小之间的比较
            fileNum = fileNum + 1 # 统计文件夹内的文件数量
            if fileLength > 0 : # 需要进行填充
                with open(pcapPath, 'ab') as f: # 这里使用with来操作文件
                    f.write(bytes([0]*fileLength)) # 获取文件内容  
            elif fileLength < 0 : # 需要进行裁剪
                with open(pcapPath, 'ab') as f: # 这里使用with来操作文件
                    f.seek(TRIMED_FILE_LEN)
                    f.truncate()
            else:
                pass
        pcapFile = root.split('/')[-1] # 所在文件夹的名字
        print(pcapFile) # 打印所在文件夹
        print('-'*10)
