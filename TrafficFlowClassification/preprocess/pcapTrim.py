'''
@Author: WANG Maonan
@Date: 2021-01-05 16:48:42
@Description: 对 pcap 文件进行减裁, 使其转换为指定的大小
@LastEditTime: 2021-01-05 17:02:57
'''

import os

from TrafficFlowClassification.TrafficLog.setLog import logger

def pcap_trim(pcap_folder, trimed_file_len):
    for (root, _ , files) in os.walk(pcap_folder):
        logger.info('正在减裁文件夹 {} 中的 pcap 文件'.format(root))
        for Ufile in files:
            pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
            pcapSize = os.path.getsize(pcapPath) # 获得文件的大小, bytes
            fileLength = trimed_file_len - pcapSize # 文件大小与规定大小之间的比较
            if fileLength > 0 : # 需要进行填充
                with open(pcapPath, 'ab') as f: # 这里使用with来操作文件
                    f.write(bytes([0]*fileLength)) # 获取文件内容  
            elif fileLength < 0 : # 需要进行裁剪
                with open(pcapPath, 'ab') as f: # 这里使用with来操作文件
                    f.seek(trimed_file_len)
                    f.truncate() # 文件进行裁断
            else: # 文件大小正好是 trimed_file_len 的, 就不需要进行处理
                pass
                        
    logger.info('将所有 pcap 减裁为 {} bytes!'.format(trimed_file_len))
    logger.info('==========\n')
