'''
@Author: WANG Maonan
@Date: 2021-01-05 16:48:42
@Description: 对 pcap 文件进行减裁, 使其转换为指定的大小
@LastEditTime: 2021-02-01 13:49:29
'''

import os
import json
from TrafficFlowClassification.TrafficLog.setLog import logger

class collect_pcap_size(object):
    """统计不同类别的 session 的文件长度, 最终 pcaps_length 的输出如下所示:
    {
        'Chat', [700, 691, 100, ...]
        'Email': [100, 102, 201, 20, ...],
        ...
    }
    即为每一个类别的 pcap 中 pcap 文件的大小
    """
    def __init__(self) -> None:
        self.pcaps_length = {}

    def update_pcaps_length(self, pcap_label, pcap_size):
        if pcap_label not in self.pcaps_length:
            self.pcaps_length[pcap_label] = [pcap_size]
        else:
            self.pcaps_length[pcap_label].append(pcap_size)

def pcap_trim(pcap_folder, trimed_file_len):
    """将 pcap 文件裁剪为指定的大小

    Args:
        pcap_folder (str): pcap 文件所在的文件夹
        trimed_file_len (str): 减裁的大小
    """
    get_pcaps_size = collect_pcap_size() # 用来统计每个 pcap 文件的大小
    for (root, _ , files) in os.walk(pcap_folder):
        logger.info('正在减裁文件夹 {} 中的 pcap 文件'.format(root))
        for Ufile in files: # Ufile 具体到了文件名
            pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
            pcapSize = os.path.getsize(pcapPath) # 获得文件的大小, bytes
            
            pcap_label = os.path.normpath(root).split('\\')[2] # 获得 pcap 所属类别
            get_pcaps_size.update_pcaps_length(pcap_label, pcapSize) # 更新每一类的数量

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
    
    # 将不同类别的 pcap 的大小转换为 json 文件存储
    logger.info('==========\n')
    logger.info('存储不同 pcap 文件的大小.')
    with open("./pcaps_size_record.json", "w") as f:
        json.dump(get_pcaps_size.pcaps_length, f)

    logger.info('将所有 pcap 减裁为 {} bytes!'.format(trimed_file_len))
    logger.info('==========\n')