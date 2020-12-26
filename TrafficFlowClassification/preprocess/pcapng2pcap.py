'''
@Author: WANG Maonan
@Date: 2020-12-14 18:50:39
@Description: 将数据集中的pcapng转换为pcap文件.
editcap.exe 的用法, 
==> editcap.exe -F libpcap -T ether file.pcapng file.pcap
==> 需要将 editcap.exe 所在的路径 (C:/Program Files/Wireshark) 添加在环境目录中
@LastEditTime: 2020-12-25 18:45:15
'''

from math import log
import os
import subprocess
from TrafficFlowClassification.TrafficLog.setLog import logger

def pcapng_to_pcap(path):
    """将文件夹 path 中所有的 pcapng 文件转换为 pcap 文件

    Args:
        path (str): pcapng 文件所在的路径
    """
    for files in os.listdir(path):
        if files.split('.')[1]=='pcapng':
            # pcapng2pcap = 'editcap.exe -F libpcap -T ether {}{} {}{}.pcap'.format(path, files, path, files.split('.')[0])
            # os.system(pcapng2pcap) # 直行 cmd 命令
            output_pcap_name = '{}.pcap'.format(files.split('.')[0])
            prog = subprocess.Popen(["editcap.exe", 
                            "-F", "libpcap",
                            "-T", "ether",
                            os.path.abspath(os.path.join(path, files)),
                            os.path.abspath(os.path.join(path, output_pcap_name))],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            _, _ = prog.communicate()
            os.remove(os.path.abspath(os.path.join(path, files))) # pcapng 转换之后, 源文件删除
            logger.info("文件 {}, 修改成功.".format(files))
    
    logger.info('将所以的 pcapng 文件转换为 pcap 文件.')
    logger.info('============\n')