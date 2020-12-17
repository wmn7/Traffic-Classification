'''
@Author: WANG Maonan
@Date: 2020-12-14 18:50:39
@Description: 将数据集中的pcapng转换为pcap文件.
@LastEditTime: 2020-12-16 12:49:23
'''

# editcap.exe -F libpcap -T ether file.pcapng file.pcap
# 修改path(为pcapng存放的位置, 将该文件放在C:/Program Files/Wireshark下, 以管理员身份进行运行),
# ==============================================================================

import os
import subprocess
from TrafficFlowClassification.TrafficLog.setLog import logger

def pcapng_to_pcap(path):
    """将文件夹中所有的 pcapng 文件转换为 pcap 文件

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
            # TODO, pcapng 转换之后, 源文件删除
            logger.info("文件 {}, 修改成功.".format(files))