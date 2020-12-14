# Maonan Wang(wangmaonan@bupt.edu.cn)
# 
# 划分训练集和测试集.
# 对于每一个大的pcap文件中拆分出的若干个session: 
# - 选出前6000个pcap, 其中10%作为测试集, 90%作为训练集
# - 同时对文件进行判断, 删除0byte的文件
# ==============================================================================

import os
import shutil # 用于文件的复制
import numpy as np

trainPath = """../1.DataSet/3_trainDataset/train""" # 训练集文件夹路径
testPath = """../1.DataSet/3_trainDataset/test""" # 测试集文件夹路径 
dataPath = """../1.DataSet/2_pcap2session/""" # 原始数据路径

SESSIONS_COUNT_LIMIT_MAX = 6000 # 一个pcap提取最多的session

# 循环读取文件夹中所有的pcapng文件
for (root, dirs, files) in os.walk(dataPath):
    fileNum = 0 # 统计文件夹内文件个数
    dtype = [('filePath', 'U1000'), ('filesize', 'int64')]
    fileList = [] # 记录文件名和大小
    for Ufile in files:
        pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
        pcapSize = os.path.getsize(pcapPath) # 获得文件的大小, bytes
        if pcapSize > 0 and pcapSize < 104857600: # 需要文件有大小(特别大的文件就不要了, >100MB)
            fileNum = fileNum + 1 # 统计文件夹内的文件数量
            fileList = fileList + [(pcapPath, pcapSize)]
    pcapFile = root.split('/')[-1] # 所在文件夹的名字
    print(pcapFile) # 打印所在文件夹
    fileList = np.array(fileList, dtype=dtype)
    if fileNum > 0: # 文件夹内文件的个数>0
        if fileNum > SESSIONS_COUNT_LIMIT_MAX:
            fileList = np.sort(fileList, order='filesize') # 按照文件size从大到小排序
            fileList = fileList[-6000:] # 只提取前6000个文件
            fileNum = SESSIONS_COUNT_LIMIT_MAX
        else:
            pass # 还是按照原来的顺序保持不变
        # --------------
        # 下面开始转移文件
        # --------------
        inx = np.random.choice(np.arange(fileNum), size=int(fileNum/10), replace=False) # 生成一个[0,fileNum]的数组
        testFiles = fileList[inx] # 选出10%作为测试
        trainFiles = fileList[list(set(np.arange(fileNum))-set(inx))] # 选出90%作为训练
        # 转移测试集
        for testFile in testFiles:
            fileName = testFile[0].split('/')[-1] # 获取chat/qq/xxx.pcap
            dst = '{}/{}'.format(testPath,fileName).replace('\\','/')
            # print(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True) # 没有就创建文件夹
            shutil.copy(testFile[0], dst)
        # 转移训练集
        for trainFile in trainFiles:
            fileName = trainFile[0].split('/')[-1] # 获取chat/qq/xxx.pcap
            dst = '{}/{}'.format(trainPath,fileName).replace('\\','/')
            # print(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True) # 没有就创建文件夹
            shutil.copy(trainFile[0], dst)
        print('-'*10)
